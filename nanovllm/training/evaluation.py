"""Evaluation utilities for training and inference."""

import math
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm

from nanovllm.training.utils import get_rank, is_main_process, reduce_tensor, AverageMeter

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    loss: float
    perplexity: float
    accuracy: Optional[float] = None
    num_samples: int = 0
    num_tokens: int = 0


class Evaluator:
    """Evaluation engine for language models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        use_amp: bool = False,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Device to run evaluation on
            use_amp: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp
        self.autocast_dtype = autocast_dtype
    
    def evaluate(
        self,
        dataloader: DataLoader,
        max_eval_samples: Optional[int] = None,
        compute_accuracy: bool = False,
        return_predictions: bool = False,
    ) -> EvaluationMetrics:
        """
        Run evaluation on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            max_eval_samples: Maximum number of samples to evaluate
            compute_accuracy: Whether to compute token-level accuracy
            return_predictions: Whether to return model predictions
            
        Returns:
            EvaluationMetrics object with computed metrics
        """
        if is_main_process():
            logger.info("***** Running evaluation *****")
            logger.info(f"  Num examples = {len(dataloader.dataset)}")
            logger.info(f"  Batch size = {dataloader.batch_size}")
        
        self.model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter() if compute_accuracy else None
        total_samples = 0
        total_tokens = 0
        predictions = [] if return_predictions else None
        
        with torch.no_grad():
            if is_main_process():
                pbar = tqdm(dataloader, desc="Evaluating")
            else:
                pbar = dataloader
            
            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached max samples
                if max_eval_samples and total_samples >= max_eval_samples:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch["input_ids"].size(0)
                
                # Forward pass
                if self.use_amp:
                    with autocast(dtype=self.autocast_dtype):
                        outputs = self._forward_step(batch)
                else:
                    outputs = self._forward_step(batch)
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                # Update loss
                loss_meter.update(loss.item(), batch_size)
                
                # Compute accuracy if requested
                if compute_accuracy and accuracy_meter is not None:
                    accuracy = self._compute_accuracy(logits, batch["labels"])
                    accuracy_meter.update(accuracy, batch_size)
                
                # Store predictions if requested
                if return_predictions:
                    batch_predictions = self._extract_predictions(logits, batch)
                    predictions.extend(batch_predictions)
                
                # Update counters
                total_samples += batch_size
                total_tokens += batch["attention_mask"].sum().item()
                
                # Update progress bar
                if is_main_process():
                    pbar.set_postfix({
                        "loss": f"{loss_meter.avg:.4f}",
                        "ppl": f"{math.exp(loss_meter.avg):.2f}",
                    })
        
        # Reduce metrics across all processes
        eval_loss = torch.tensor(loss_meter.avg, device=self.device)
        eval_loss = reduce_tensor(eval_loss, average=True)
        
        eval_accuracy = None
        if accuracy_meter is not None:
            eval_accuracy = torch.tensor(accuracy_meter.avg, device=self.device)
            eval_accuracy = reduce_tensor(eval_accuracy, average=True)
            eval_accuracy = eval_accuracy.item()
        
        # Compute perplexity
        perplexity = math.exp(eval_loss.item())
        
        # Create metrics object
        metrics = EvaluationMetrics(
            loss=eval_loss.item(),
            perplexity=perplexity,
            accuracy=eval_accuracy,
            num_samples=total_samples,
            num_tokens=total_tokens,
        )
        
        if is_main_process():
            logger.info(f"Evaluation results: {metrics}")
        
        if return_predictions:
            return metrics, predictions
        return metrics
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform forward pass and compute loss."""
        outputs = self.model(**batch)
        
        if isinstance(outputs, dict):
            return outputs
        else:
            # Handle case where model returns only logits
            logits = outputs
            loss = self._compute_loss(logits, batch["labels"])
            return {"loss": loss, "logits": logits}
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            reduction="mean"
        )
        
        return loss
    
    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute token-level accuracy."""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get predictions
        predictions = shift_logits.argmax(dim=-1)
        
        # Create mask for valid tokens (not -100)
        valid_mask = (shift_labels != -100)
        
        if valid_mask.sum() == 0:
            return 0.0
        
        # Compute accuracy
        correct = (predictions == shift_labels) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
        
        return accuracy.item()
    
    def _extract_predictions(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Extract text predictions from logits."""
        predictions = []
        
        # Get predicted token IDs
        predicted_ids = logits.argmax(dim=-1)
        
        for i in range(predicted_ids.size(0)):
            # Get sequence length (excluding padding)
            attention_mask = batch["attention_mask"][i]
            seq_len = attention_mask.sum().item()
            
            # Extract predicted tokens
            pred_tokens = predicted_ids[i, :seq_len]
            
            # Decode to text
            try:
                pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                predictions.append(pred_text)
            except Exception as e:
                logger.warning(f"Failed to decode prediction: {e}")
                predictions.append("")
        
        return predictions
    
    def compute_perplexity(
        self,
        texts: List[str],
        max_length: int = 1024,
        batch_size: int = 1,
    ) -> float:
        """
        Compute perplexity for a list of texts.
        
        Args:
            texts: List of texts to evaluate
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            
        Returns:
            Average perplexity across all texts
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(dtype=self.autocast_dtype):
                        outputs = self.model(input_ids=input_ids, labels=input_ids)
                else:
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                
                # Accumulate loss and token count
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        return perplexity


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    eval_dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    max_eval_samples: Optional[int] = None,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        autocast_dtype: Data type for autocast
        max_eval_samples: Maximum number of samples to evaluate
        
    Returns:
        EvaluationMetrics object
    """
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
    )
    
    return evaluator.evaluate(
        dataloader=eval_dataloader,
        max_eval_samples=max_eval_samples,
        compute_accuracy=True,
    )
