#!/usr/bin/env python3
"""
Training script for nano-vllm language models.

This script demonstrates how to train language models from scratch using the nano-vllm
training framework with support for distributed training, mixed precision, gradient
checkpointing, and other advanced features.

Usage:
    # Single GPU training
    python train.py --config configs/train_config.json

    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 train.py --config configs/train_config.json

    # Resume from checkpoint
    python train.py --config configs/train_config.json --resume_from_checkpoint ./checkpoints/checkpoint-1000
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoConfig

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training import (
    Trainer, CausalLMDataset, prepare_datasets,
    MultiDocumentDataset, prepare_multi_document_datasets, analyze_document_collection,
    set_seed, init_distributed_training, cleanup_distributed,
    setup_logging, is_main_process, get_rank, get_world_size
)
from nanovllm.training.checkpoint import CheckpointManager, auto_resume_training
from nanovllm.training.memory import optimize_memory_usage, get_model_memory_footprint, estimate_training_memory

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train nano-vllm language models")
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file (JSON)"
    )
    
    # Model and data
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for checkpoints and logs"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Maximum number of training steps"
    )
    
    # Optimization
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--gradient_clipping",
        type=float,
        help="Gradient clipping threshold"
    )
    
    # Distributed training
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        help="Tensor parallel size"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluate every N steps"
    )
    
    # Logging
    parser.add_argument(
        "--logging_steps",
        type=int,
        help="Log every N steps"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Multi-document options
    parser.add_argument(
        "--use_multi_document",
        action="store_true",
        help="Use multi-document dataset for training on folders"
    )
    parser.add_argument(
        "--concatenate_documents",
        action="store_true",
        help="Concatenate documents instead of processing individually"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=128,
        help="Number of tokens to overlap between chunks"
    )
    parser.add_argument(
        "--analyze_documents",
        action="store_true",
        help="Analyze document collection before training"
    )

    # From-scratch training options
    parser.add_argument(
        "--train_from_scratch",
        action="store_true",
        help="Train model from scratch instead of fine-tuning"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size for from-scratch training"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Vocabulary size for from-scratch training (auto-determined if not specified)"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def load_config(config_path: str, args) -> TrainingConfig:
    """Load training configuration from file and command line arguments."""
    # Load base config from file
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config_dict[key] = value
    
    return TrainingConfig(**config_dict)


def setup_model_and_tokenizer(config: TrainingConfig):
    """Setup model and tokenizer."""
    if config.train_from_scratch:
        logger.info(f"Creating model from scratch using documents in {config.dataset_path}")

        from nanovllm.training.from_scratch import create_model_from_scratch

        # Create model and tokenizer from documents
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

    else:
        logger.info(f"Loading model and tokenizer from {config.model_name_or_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        # Create model
        model = Qwen3ForCausalLM(model_config)

    # Log model info
    if is_main_process():
        memory_footprint = get_model_memory_footprint(model)
        logger.info(f"Model memory footprint: {memory_footprint['total_gb']:.2f}GB")

        # Estimate training memory requirements
        memory_estimate = estimate_training_memory(
            model=model,
            batch_size=config.per_device_train_batch_size,
            sequence_length=config.max_seq_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision != "no",
            gradient_checkpointing=config.gradient_checkpointing,
        )
        logger.info(f"Estimated training memory: {memory_estimate['total_gb']:.2f}GB")

    return model, tokenizer


def setup_datasets(config: TrainingConfig, tokenizer, use_multi_document=False, **kwargs):
    """Setup training and evaluation datasets."""
    if not config.dataset_path:
        logger.warning("No dataset path provided")
        return None, None

    logger.info(f"Loading datasets from {config.dataset_path}")

    # Analyze documents if requested
    if kwargs.get('analyze_documents', False):
        logger.info("Analyzing document collection...")
        analysis = analyze_document_collection(config.dataset_path)
        logger.info(f"Document analysis:")
        logger.info(f"  Total files: {analysis['total_files']}")
        logger.info(f"  File types: {analysis['file_types']}")
        logger.info(f"  Total documents: {analysis['total_documents']}")
        logger.info(f"  Total characters: {analysis['total_characters']:,}")
        logger.info(f"  Average document length: {analysis['average_doc_length']:.1f} characters")

    if use_multi_document:
        # Use multi-document dataset
        logger.info("Using multi-document dataset")
        train_dataset, eval_dataset = prepare_multi_document_datasets(
            data_path=config.dataset_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            concatenate_documents=kwargs.get('concatenate_documents', False),
            chunk_overlap=kwargs.get('chunk_overlap', 128),
            min_chunk_size=kwargs.get('min_chunk_size', 100),
        )
    else:
        # Use standard dataset
        logger.info("Using standard dataset")
        train_dataset, eval_dataset = prepare_datasets(
            data_path=config.dataset_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

    if is_main_process():
        logger.info(f"Training dataset size: {len(train_dataset) if train_dataset else 0}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")

        if use_multi_document and train_dataset:
            logger.info(f"Training chunks: {len(train_dataset.chunks) if hasattr(train_dataset, 'chunks') else 'N/A'}")
            logger.info(f"Source documents: {len(train_dataset.documents) if hasattr(train_dataset, 'documents') else 'N/A'}")

    return train_dataset, eval_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Initialize distributed training
    init_distributed_training()
    
    try:
        # Load configuration
        config = load_config(args.config, args)
        
        if is_main_process():
            logger.info("Training configuration:")
            logger.info(json.dumps(config.__dict__, indent=2, default=str))
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Apply memory optimizations
        if config.gradient_checkpointing or config.cpu_offload:
            optimize_memory_usage(
                model=model,
                enable_gradient_checkpointing=config.gradient_checkpointing,
                enable_cpu_offload=config.cpu_offload,
            )
        
        # Setup datasets
        train_dataset, eval_dataset = setup_datasets(
            config,
            tokenizer,
            use_multi_document=args.use_multi_document,
            concatenate_documents=args.concatenate_documents,
            chunk_overlap=args.chunk_overlap,
            analyze_documents=args.analyze_documents,
        )
        
        # Create trainer
        trainer = Trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Check for checkpoint to resume from
        resume_checkpoint = auto_resume_training(config.output_dir, config)
        if resume_checkpoint:
            trainer.load_checkpoint(resume_checkpoint)
        
        # Start training
        if is_main_process():
            logger.info("Starting training...")
        
        training_results = trainer.train()
        
        if is_main_process():
            logger.info("Training completed!")
            logger.info(f"Results: {training_results}")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()
