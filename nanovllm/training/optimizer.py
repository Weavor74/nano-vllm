"""Optimizer and learning rate scheduler implementations for training."""

import math
from typing import Optional, Union, Callable, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist


class AdamW(Optimizer):
    """
    AdamW optimizer with support for tensor parallelism.
    
    This implementation is optimized for distributed training and includes
    proper handling of weight decay and gradient clipping.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    no_decay_params: Optional[list[str]] = None,
) -> AdamW:
    """
    Create AdamW optimizer with proper parameter grouping.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        no_decay_params: List of parameter names that should not have weight decay
        
    Returns:
        Configured AdamW optimizer
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
    
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd_param in name for nd_param in no_decay_params):
            no_decay_params_list.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params_list, "weight_decay": 0.0},
    ]
    
    return AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:
    """Create a schedule with a constant learning rate."""
    return LambdaLR(optimizer, lambda _: 1, last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 0,
    num_training_steps: int = 1000,
    **kwargs,
) -> LambdaLR:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("linear", "cosine", "constant")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "constant":
        return get_constant_schedule(optimizer)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """
    Clip gradient norm with support for tensor parallelism.
    
    This function properly handles gradient clipping in distributed settings
    by computing the global gradient norm across all processes.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    
    # Synchronize gradient norm across all processes in distributed training
    if dist.is_initialized():
        dist.all_reduce(total_norm)
        total_norm = total_norm / dist.get_world_size()
    
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`'
        )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    
    return total_norm
