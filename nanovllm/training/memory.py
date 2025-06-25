"""Memory optimization utilities for training large models."""

import gc
import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

logger = logging.getLogger(__name__)


class GradientCheckpointing:
    """Gradient checkpointing utilities for memory-efficient training."""
    
    @staticmethod
    def enable_gradient_checkpointing(
        model: nn.Module,
        checkpoint_layers: Optional[list] = None,
        use_reentrant: bool = False,
    ):
        """
        Enable gradient checkpointing for specified layers.
        
        Args:
            model: Model to enable checkpointing for
            checkpoint_layers: List of layer names to checkpoint (None for all transformer layers)
            use_reentrant: Whether to use reentrant checkpointing
        """
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        # Find transformer layers to checkpoint
        layers_to_checkpoint = []
        
        if checkpoint_layers is None:
            # Auto-detect transformer layers
            for name, module in model.named_modules():
                if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'decoder']):
                    if hasattr(module, 'forward') and len(list(module.children())) > 0:
                        layers_to_checkpoint.append((name, module))
        else:
            # Use specified layers
            for name, module in model.named_modules():
                if name in checkpoint_layers:
                    layers_to_checkpoint.append((name, module))
        
        # Apply checkpointing
        for name, module in layers_to_checkpoint:
            original_forward = module.forward
            
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(
                    create_custom_forward(module),
                    *args,
                    use_reentrant=use_reentrant,
                    **kwargs
                )
            
            module.forward = checkpointed_forward
            logger.info(f"Enabled gradient checkpointing for layer: {name}")
    
    @staticmethod
    def disable_gradient_checkpointing(model: nn.Module):
        """Disable gradient checkpointing by restoring original forward methods."""
        for name, module in model.named_modules():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
                logger.info(f"Disabled gradient checkpointing for layer: {name}")


class MemoryManager:
    """Memory management utilities for training."""
    
    def __init__(self, device: torch.device):
        """
        Initialize memory manager.
        
        Args:
            device: Device to manage memory for
        """
        self.device = device
        self.peak_memory = 0
        self.initial_memory = 0
        
        if device.type == 'cuda':
            self.initial_memory = torch.cuda.memory_allocated(device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            max_reserved = torch.cuda.max_memory_reserved(self.device)
            
            return {
                "allocated_gb": allocated / 1024**3,
                "reserved_gb": reserved / 1024**3,
                "max_allocated_gb": max_allocated / 1024**3,
                "max_reserved_gb": max_reserved / 1024**3,
                "utilization": allocated / reserved if reserved > 0 else 0.0,
            }
        else:
            return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0, "max_reserved_gb": 0, "utilization": 0.0}
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    @contextmanager
    def memory_profiling(self, operation_name: str = "operation"):
        """Context manager for profiling memory usage of an operation."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            start_memory = torch.cuda.memory_allocated(self.device)
            
            yield
            
            torch.cuda.synchronize(self.device)
            end_memory = torch.cuda.memory_allocated(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            
            memory_diff = (end_memory - start_memory) / 1024**3
            peak_diff = (peak_memory - start_memory) / 1024**3
            
            logger.info(f"{operation_name} - Memory delta: {memory_diff:.2f}GB, Peak: {peak_diff:.2f}GB")
        else:
            yield
    
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        usage = self.get_memory_usage()
        logger.info(f"{prefix}Memory usage - Allocated: {usage['allocated_gb']:.2f}GB, "
                   f"Reserved: {usage['reserved_gb']:.2f}GB, "
                   f"Utilization: {usage['utilization']:.1%}")


class CPUOffloader:
    """CPU offloading utilities for large model training."""
    
    def __init__(self, model: nn.Module, offload_params: bool = True, offload_gradients: bool = True):
        """
        Initialize CPU offloader.
        
        Args:
            model: Model to offload
            offload_params: Whether to offload parameters to CPU
            offload_gradients: Whether to offload gradients to CPU
        """
        self.model = model
        self.offload_params = offload_params
        self.offload_gradients = offload_gradients
        self.param_cpu_storage = {}
        self.grad_cpu_storage = {}
    
    def offload_parameters(self):
        """Offload model parameters to CPU."""
        if not self.offload_params:
            return
        
        for name, param in self.model.named_parameters():
            if param.device.type == 'cuda':
                self.param_cpu_storage[name] = param.data.cpu()
                param.data = torch.empty_like(param.data, device='cpu')
    
    def restore_parameters(self):
        """Restore model parameters from CPU."""
        if not self.offload_params:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.param_cpu_storage:
                param.data = self.param_cpu_storage[name].cuda()
                del self.param_cpu_storage[name]
    
    def offload_gradients(self):
        """Offload gradients to CPU."""
        if not self.offload_gradients:
            return
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.grad.device.type == 'cuda':
                self.grad_cpu_storage[name] = param.grad.cpu()
                param.grad = None
    
    def restore_gradients(self):
        """Restore gradients from CPU."""
        if not self.offload_gradients:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.grad_cpu_storage:
                param.grad = self.grad_cpu_storage[name].cuda()
                del self.grad_cpu_storage[name]


def optimize_memory_usage(
    model: nn.Module,
    enable_gradient_checkpointing: bool = True,
    enable_cpu_offload: bool = False,
    clear_cache_frequency: int = 100,
) -> Dict[str, Any]:
    """
    Apply memory optimization techniques to a model.
    
    Args:
        model: Model to optimize
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        enable_cpu_offload: Whether to enable CPU offloading
        clear_cache_frequency: How often to clear GPU cache (in steps)
        
    Returns:
        Dictionary with optimization settings and utilities
    """
    optimizations = {}
    
    # Enable gradient checkpointing
    if enable_gradient_checkpointing:
        GradientCheckpointing.enable_gradient_checkpointing(model)
        optimizations['gradient_checkpointing'] = True
        logger.info("Enabled gradient checkpointing")
    
    # Setup CPU offloader
    if enable_cpu_offload:
        offloader = CPUOffloader(model)
        optimizations['cpu_offloader'] = offloader
        logger.info("Enabled CPU offloading")
    
    # Setup memory manager
    device = next(model.parameters()).device
    memory_manager = MemoryManager(device)
    optimizations['memory_manager'] = memory_manager
    
    optimizations['clear_cache_frequency'] = clear_cache_frequency
    
    return optimizations


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the memory footprint of a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with memory usage breakdown
    """
    param_memory = 0
    buffer_memory = 0
    
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    return {
        "parameters_gb": param_memory / 1024**3,
        "buffers_gb": buffer_memory / 1024**3,
        "total_gb": total_memory / 1024**3,
        "parameters_mb": param_memory / 1024**2,
        "buffers_mb": buffer_memory / 1024**2,
        "total_mb": total_memory / 1024**2,
    }


def estimate_training_memory(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.
    
    Args:
        model: Model to train
        batch_size: Training batch size
        sequence_length: Input sequence length
        gradient_accumulation_steps: Number of gradient accumulation steps
        mixed_precision: Whether using mixed precision training
        gradient_checkpointing: Whether using gradient checkpointing
        
    Returns:
        Dictionary with estimated memory requirements
    """
    # Model memory
    model_memory = get_model_memory_footprint(model)
    
    # Activation memory (rough estimate)
    hidden_size = getattr(model.config, 'hidden_size', 4096)
    num_layers = getattr(model.config, 'num_hidden_layers', 32)
    
    # Estimate activation memory per layer
    activation_per_layer = batch_size * sequence_length * hidden_size * 4  # 4 bytes for float32
    if mixed_precision:
        activation_per_layer *= 0.5  # Roughly half for mixed precision
    
    if gradient_checkpointing:
        # Only store activations for sqrt(num_layers) layers
        activation_memory = activation_per_layer * (num_layers ** 0.5)
    else:
        activation_memory = activation_per_layer * num_layers
    
    # Gradient memory (same as model parameters)
    gradient_memory = model_memory["total_gb"] * 1024**3
    
    # Optimizer memory (AdamW stores momentum and variance)
    optimizer_memory = model_memory["total_gb"] * 1024**3 * 2  # 2x for momentum and variance
    
    # Total memory
    total_memory = (
        model_memory["total_gb"] * 1024**3 +
        activation_memory +
        gradient_memory +
        optimizer_memory
    )
    
    return {
        "model_gb": model_memory["total_gb"],
        "activations_gb": activation_memory / 1024**3,
        "gradients_gb": gradient_memory / 1024**3,
        "optimizer_gb": optimizer_memory / 1024**3,
        "total_gb": total_memory / 1024**3,
        "per_gpu_gb": total_memory / 1024**3,  # Assuming single GPU for now
    }
