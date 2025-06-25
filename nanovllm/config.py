import os
from dataclasses import dataclass
from typing import Optional, Literal
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


@dataclass
class TrainingConfig:
    """Configuration for training language models with nano-vllm."""

    # Model and data configuration
    model_name_or_path: Optional[str] = None  # None for from-scratch training
    dataset_path: str = ""
    output_dir: str = "./checkpoints"

    # From-scratch training options
    train_from_scratch: bool = False
    vocab_size: Optional[int] = None  # Auto-determined from data if None
    model_size: str = "small"  # "tiny", "small", "medium", "large"

    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Batch size and sequence length
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 2048

    # Training schedule
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "cosine"

    # Mixed precision and optimization
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_clipping: float = 1.0
    gradient_checkpointing: bool = False

    # Distributed training
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1

    # Evaluation and logging
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_strategy: Literal["no", "steps", "epoch"] = "steps"

    # Memory optimization
    max_memory_per_gpu: Optional[str] = None
    cpu_offload: bool = False

    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3

    # Data processing
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Seed and reproducibility
    seed: int = 42

    # Advanced training options
    ignore_data_skip: bool = False
    remove_unused_columns: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.per_device_train_batch_size > 0, "Batch size must be positive"
        assert self.gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
        assert self.max_seq_length > 0, "Max sequence length must be positive"
        assert self.tensor_parallel_size >= 1, "Tensor parallel size must be at least 1"
        assert self.data_parallel_size >= 1, "Data parallel size must be at least 1"
        assert self.gradient_clipping > 0, "Gradient clipping must be positive"

        # Validate from-scratch training configuration
        if self.train_from_scratch:
            if self.model_name_or_path is not None:
                print("Warning: model_name_or_path ignored when train_from_scratch=True")
            assert self.dataset_path, "dataset_path required for from-scratch training"
            assert self.model_size in ["tiny", "small", "medium", "large"], f"Invalid model_size: {self.model_size}"
        else:
            assert self.model_name_or_path, "model_name_or_path required when not training from scratch"

        # Ensure warmup configuration is valid
        if self.warmup_ratio > 0 and self.warmup_steps > 0:
            raise ValueError("Cannot specify both warmup_ratio and warmup_steps")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate effective batch size
        self.effective_batch_size = (
            self.per_device_train_batch_size *
            self.gradient_accumulation_steps *
            self.data_parallel_size
        )

    @property
    def world_size(self) -> int:
        """Total number of processes in distributed training."""
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def is_distributed(self) -> bool:
        """Whether this is a distributed training setup."""
        return self.world_size > 1
