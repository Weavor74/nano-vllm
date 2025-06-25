"""Advanced checkpoint management for distributed training."""

import os
import json
import shutil
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer, AutoConfig

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training.utils import get_rank, is_main_process

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Advanced checkpoint management with support for distributed training."""
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        save_strategy: str = "steps",
        save_steps: int = 1000,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            save_strategy: When to save checkpoints ("steps" or "epoch")
            save_steps: Save every N steps (if save_strategy is "steps")
        """
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        
        # Create output directory
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        tokenizer: PreTrainedTokenizer,
        step: int,
        epoch: int,
        loss: float,
        config: TrainingConfig,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        is_best: bool = False,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a comprehensive checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            tokenizer: Tokenizer
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            config: Training configuration
            scaler: Mixed precision scaler
            is_best: Whether this is the best checkpoint
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        if not is_main_process():
            return ""
        
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Get model state dict (unwrap DDP if needed)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save model state
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(model_to_save.state_dict(), model_path)
        
        # Save model config
        if hasattr(model_to_save, 'config'):
            model_config = model_to_save.config
        else:
            # Try to get config from the model class
            model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        
        config_path = checkpoint_dir / "config.json"
        model_config.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "training_config": config.__dict__,
        }
        
        if additional_data:
            training_state.update(additional_data)
        
        training_state_path = checkpoint_dir / "training_state.pt"
        torch.save(training_state, training_state_path)
        
        # Save training arguments
        training_args_path = checkpoint_dir / "training_args.json"
        with open(training_args_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        # Create a marker file to indicate successful save
        marker_path = checkpoint_dir / "checkpoint_complete.txt"
        with open(marker_path, 'w') as f:
            f.write(f"Checkpoint saved at step {step}\n")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Save as best model if specified
        if is_best:
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            logger.info(f"Best model saved to {best_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_dir)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            scaler: Scaler to load state into
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dictionary with training state information
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Check if checkpoint is complete
        marker_path = checkpoint_dir / "checkpoint_complete.txt"
        if not marker_path.exists():
            logger.warning(f"Checkpoint may be incomplete: {checkpoint_dir}")
        
        # Load model state
        model_path = checkpoint_dir / "pytorch_model.bin"
        if model_path.exists():
            model_state_dict = torch.load(model_path, map_location="cpu")
            
            # Handle DDP wrapper
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(model_state_dict, strict=strict)
            logger.info(f"Model state loaded from {model_path}")
        else:
            logger.warning(f"Model state file not found: {model_path}")
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.pt"
        training_state = {}
        
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location="cpu")
            
            # Load optimizer state
            if optimizer is not None and "optimizer_state_dict" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                logger.info("Optimizer state loaded")
            
            # Load scheduler state
            if scheduler is not None and "lr_scheduler_state_dict" in training_state:
                scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])
                logger.info("Scheduler state loaded")
            
            # Load scaler state
            if scaler is not None and "scaler_state_dict" in training_state:
                scaler.load_state_dict(training_state["scaler_state_dict"])
                logger.info("Scaler state loaded")
        else:
            logger.warning(f"Training state file not found: {training_state_path}")
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        return training_state
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = []
        
        if not self.output_dir.exists():
            return checkpoints
        
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Check if checkpoint is complete
                marker_path = item / "checkpoint_complete.txt"
                if marker_path.exists():
                    checkpoints.append(str(item))
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(Path(x).name.split("-")[1]))
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint."""
        best_dir = self.output_dir / "best_model"
        if best_dir.exists():
            marker_path = best_dir / "checkpoint_complete.txt"
            if marker_path.exists():
                return str(best_dir)
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.save_total_limit <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        
        # Remove old checkpoints
        while len(checkpoints) > self.save_total_limit:
            checkpoint_to_remove = checkpoints.pop(0)
            checkpoint_path = Path(checkpoint_to_remove)
            
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    def export_model(
        self,
        checkpoint_path: str,
        export_dir: str,
        export_format: str = "huggingface",
    ):
        """
        Export model in different formats.
        
        Args:
            checkpoint_path: Path to checkpoint
            export_dir: Directory to export to
            export_format: Export format ("huggingface", "onnx", "torchscript")
        """
        if not is_main_process():
            return
        
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = Path(checkpoint_path)
        
        if export_format == "huggingface":
            # Copy model files
            for file_name in ["pytorch_model.bin", "config.json", "tokenizer.json", 
                             "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]:
                src_path = checkpoint_dir / file_name
                if src_path.exists():
                    dst_path = export_path / file_name
                    shutil.copy2(src_path, dst_path)
            
            logger.info(f"Model exported to {export_path} in HuggingFace format")
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")


def auto_resume_training(
    output_dir: str,
    config: TrainingConfig,
) -> Optional[str]:
    """
    Automatically find and return the latest checkpoint for resuming training.
    
    Args:
        output_dir: Training output directory
        config: Training configuration
        
    Returns:
        Path to checkpoint to resume from, or None if no checkpoint found
    """
    if config.resume_from_checkpoint:
        # Explicit checkpoint path provided
        if os.path.exists(config.resume_from_checkpoint):
            return config.resume_from_checkpoint
        else:
            logger.warning(f"Specified checkpoint not found: {config.resume_from_checkpoint}")
    
    # Look for latest checkpoint in output directory
    checkpoint_manager = CheckpointManager(output_dir)
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    
    if latest_checkpoint:
        logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")
        return latest_checkpoint
    
    logger.info("No checkpoint found for resuming training")
    return None
