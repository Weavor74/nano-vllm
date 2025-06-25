from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.config import Config, TrainingConfig

# Training components (optional import to avoid dependency issues)
try:
    from nanovllm.training import (
        Trainer, CausalLMDataset, CausalLMDataCollator, create_dataloader,
        MultiDocumentDataset, prepare_multi_document_datasets, analyze_document_collection,
        create_optimizer, create_scheduler, set_seed, get_rank, get_world_size,
        create_model_from_scratch, CustomTokenizer
    )
    from nanovllm.training.evaluation import Evaluator, evaluate_model
    from nanovllm.training.checkpoint import CheckpointManager
    from nanovllm.training.memory import optimize_memory_usage, get_model_memory_footprint

    __all__ = [
        "LLM", "SamplingParams", "Config", "TrainingConfig",
        "Trainer", "CausalLMDataset", "CausalLMDataCollator", "create_dataloader",
        "MultiDocumentDataset", "prepare_multi_document_datasets", "analyze_document_collection",
        "create_optimizer", "create_scheduler", "set_seed", "get_rank", "get_world_size",
        "create_model_from_scratch", "CustomTokenizer",
        "Evaluator", "evaluate_model", "CheckpointManager",
        "optimize_memory_usage", "get_model_memory_footprint"
    ]
except ImportError:
    # Training components not available (missing dependencies)
    __all__ = ["LLM", "SamplingParams", "Config", "TrainingConfig"]
