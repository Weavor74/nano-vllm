"""Training engine for nano-vllm."""

import os
import time
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training.optimizer import create_optimizer, create_scheduler, clip_grad_norm_
from nanovllm.training.data import CausalLMDataset, CausalLMDataCollator, create_dataloader, prepare_datasets
from nanovllm.training.utils import (
    set_seed, get_rank, get_world_size, is_main_process,
    save_checkpoint, load_checkpoint, get_parameter_count,
    format_time, compute_metrics, AverageMeter, reduce_tensor
)

logger = logging.getLogger(__name__)


class Trainer:
    """Training engine for nano-vllm models."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Optional[CausalLMDataset] = None,
        eval_dataset: Optional[CausalLMDataset] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup distributed training
        self.is_distributed = dist.is_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )
        
        # Setup mixed precision
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None
        self.autocast_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        )
        
        # Calculate total training steps
        if train_dataset is not None:
            steps_per_epoch = len(train_dataset) // (config.per_device_train_batch_size * config.gradient_accumulation_steps * self.world_size)
            if config.max_train_steps is not None:
                self.total_steps = config.max_train_steps
            else:
                self.total_steps = steps_per_epoch * config.num_train_epochs
        else:
            self.total_steps = 1000
        
        # Setup warmup steps
        if config.warmup_ratio > 0:
            self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        else:
            self.warmup_steps = config.warmup_steps
        
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.lr_scheduler_type,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup data collator
        self.data_collator = CausalLMDataCollator(
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )
        
        # Log model info
        if is_main_process():
            total_params, trainable_params = get_parameter_count(self.model)
            logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def create_dataloaders(self) -> tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Create training and evaluation dataloaders."""
        train_dataloader = None
        eval_dataloader = None
        
        if self.train_dataset is not None:
            train_dataloader = create_dataloader(
                self.train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                collate_fn=self.data_collator,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                distributed=self.is_distributed,
            )
        
        if self.eval_dataset is not None:
            eval_dataloader = create_dataloader(
                self.eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                distributed=self.is_distributed,
            )
        
        return train_dataloader, eval_dataloader
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.use_amp:
            with autocast(dtype=self.autocast_dtype):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        else:
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
        
        train_dataloader, eval_dataloader = self.create_dataloaders()
        
        if is_main_process():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(self.train_dataset)}")
            logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.per_device_train_batch_size}")
            logger.info(f"  Total train batch size = {self.config.effective_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.total_steps}")
        
        # Training metrics
        train_loss_meter = AverageMeter()
        start_time = time.time()
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            # Training loop
            if is_main_process():
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")
            else:
                pbar = train_dataloader
            
            for step, batch in enumerate(pbar):
                # Training step
                loss = self.training_step(batch)
                train_loss_meter.update(loss.item())
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        if self.use_amp and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        
                        clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clipping
                        )
                    
                    # Optimizer step
                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0 and is_main_process():
                        lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix({
                            "loss": f"{train_loss_meter.avg:.4f}",
                            "lr": f"{lr:.2e}",
                            "step": self.global_step
                        })
                    
                    # Evaluation
                    if (self.config.eval_strategy == "steps" and 
                        self.global_step % self.config.eval_steps == 0 and 
                        eval_dataloader is not None):
                        eval_results = self.evaluate(eval_dataloader)
                        if is_main_process():
                            logger.info(f"Step {self.global_step}: {eval_results}")
                    
                    # Save checkpoint
                    if (self.config.save_strategy == "steps" and 
                        self.global_step % self.config.save_steps == 0):
                        self.save_checkpoint()
                    
                    # Check if training is complete
                    if self.global_step >= self.total_steps:
                        break
            
            # End of epoch evaluation
            if (self.config.eval_strategy == "epoch" and eval_dataloader is not None):
                eval_results = self.evaluate(eval_dataloader)
                if is_main_process():
                    logger.info(f"Epoch {epoch+1}: {eval_results}")
            
            # End of epoch checkpoint
            if self.config.save_strategy == "epoch":
                self.save_checkpoint()
            
            if self.global_step >= self.total_steps:
                break
        
        # Final evaluation and checkpoint
        if eval_dataloader is not None:
            final_eval_results = self.evaluate(eval_dataloader)
            if is_main_process():
                logger.info(f"Final evaluation: {final_eval_results}")
        
        self.save_checkpoint()
        
        total_time = time.time() - start_time
        if is_main_process():
            logger.info(f"Training completed in {format_time(total_time)}")
        
        return {
            "global_step": self.global_step,
            "train_loss": train_loss_meter.avg,
            "total_time": total_time,
        }

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation loop."""
        if is_main_process():
            logger.info("***** Running evaluation *****")

        self.model.eval()
        eval_loss_meter = AverageMeter()

        with torch.no_grad():
            if is_main_process():
                pbar = tqdm(eval_dataloader, desc="Evaluating")
            else:
                pbar = eval_dataloader

            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                if self.use_amp:
                    with autocast(dtype=self.autocast_dtype):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                eval_loss_meter.update(loss.item())

        # Reduce loss across all processes
        eval_loss = torch.tensor(eval_loss_meter.avg, device=self.device)
        eval_loss = reduce_tensor(eval_loss, average=True)

        # Compute metrics
        eval_results = compute_metrics(eval_loss.item(), len(eval_dataloader))

        # Check if this is the best model
        if eval_loss.item() < self.best_eval_loss:
            self.best_eval_loss = eval_loss.item()
            if is_main_process():
                self.save_checkpoint(is_best=True)

        return eval_results

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        if not is_main_process():
            return

        checkpoint_dir = self.config.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get model state dict (unwrap DDP if needed)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
            "config": asdict(self.config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _cleanup_checkpoints(self, checkpoint_dir: str):
        """Remove old checkpoints to save disk space."""
        if self.config.save_total_limit <= 0:
            return

        # Get all checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith("checkpoint-") and file.endswith(".pt"):
                step = int(file.split("-")[1].split(".")[0])
                checkpoint_files.append((step, file))

        # Sort by step number
        checkpoint_files.sort(key=lambda x: x[0])

        # Remove old checkpoints
        while len(checkpoint_files) > self.config.save_total_limit:
            _, file_to_remove = checkpoint_files.pop(0)
            file_path = os.path.join(checkpoint_dir, file_to_remove)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed old checkpoint: {file_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))

        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed training from step {self.global_step}")

    @classmethod
    def from_pretrained(
        cls,
        config: TrainingConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "Trainer":
        """Create trainer from pretrained model or from scratch."""
        if config.train_from_scratch:
            return cls.from_scratch(config, checkpoint_path)

        # Load model configuration
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        # Create model
        model = Qwen3ForCausalLM(model_config)

        # Load tokenizer if not provided
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        # Load datasets if provided
        train_dataset = None
        eval_dataset = None

        if config.dataset_path:
            train_dataset, eval_dataset = prepare_datasets(
                config.dataset_path,
                tokenizer,
                config.max_seq_length,
            )

        # Create trainer
        trainer = cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Load checkpoint if provided
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)

        return trainer

    @classmethod
    def from_scratch(
        cls,
        config: TrainingConfig,
        checkpoint_path: Optional[str] = None,
    ) -> "Trainer":
        """Create trainer for from-scratch training."""
        from nanovllm.training.from_scratch import create_model_from_scratch
        from nanovllm.training.data import prepare_multi_document_datasets

        if not config.train_from_scratch:
            raise ValueError("train_from_scratch must be True for from_scratch training")

        if not config.dataset_path:
            raise ValueError("dataset_path required for from-scratch training")

        # Create model and tokenizer from scratch
        model, tokenizer, creation_info = create_model_from_scratch(
            documents_path=config.dataset_path,
            model_size=config.model_size,
            vocab_size=config.vocab_size,
            save_path=os.path.join(config.output_dir, "model_artifacts"),
        )

        if is_main_process():
            logger.info("Created model from scratch:")
            logger.info(f"  Model size: {creation_info['model_size']}")
            logger.info(f"  Parameters: {creation_info['parameter_count']:,}")
            logger.info(f"  Vocabulary size: {creation_info['vocab_stats']['vocab_size']:,}")
            logger.info(f"  Vocabulary coverage: {creation_info['vocab_stats']['coverage']:.2%}")

        # Prepare datasets with custom tokenizer
        train_dataset, eval_dataset = prepare_multi_document_datasets(
            data_path=config.dataset_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

        # Create trainer
        trainer = cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Store creation info
        trainer.creation_info = creation_info

        # Load checkpoint if provided
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)

        return trainer
