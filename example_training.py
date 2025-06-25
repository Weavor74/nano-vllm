#!/usr/bin/env python3
"""
Example training script showing how to use nano-vllm training features.

This script demonstrates various training scenarios and configurations.
"""

import os
import json
from pathlib import Path

from transformers import AutoTokenizer
from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training import (
    Trainer, CausalLMDataset, CausalLMDataCollator, prepare_datasets,
    set_seed, create_optimizer, create_scheduler
)
from nanovllm.training.evaluation import evaluate_model
from nanovllm.training.memory import optimize_memory_usage, get_model_memory_footprint


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_data = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is a subset of artificial intelligence."},
        {"text": "Python is a popular programming language for data science."},
        {"text": "Deep learning models require large amounts of data for training."},
        {"text": "Natural language processing enables computers to understand human language."},
        {"text": "Transformers have revolutionized the field of NLP."},
        {"text": "Large language models can generate human-like text."},
        {"text": "Training neural networks requires careful hyperparameter tuning."},
        {"text": "Gradient descent is an optimization algorithm used in machine learning."},
        {"text": "Attention mechanisms allow models to focus on relevant parts of the input."},
    ]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save sample data
    with open("data/sample_train.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print("Sample dataset created at data/sample_train.jsonl")


def example_basic_training():
    """Example of basic training setup."""
    print("\n=== Basic Training Example ===")
    
    # Create sample dataset
    create_sample_dataset()
    
    # Training configuration
    config = TrainingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        dataset_path="data/sample_train.jsonl",
        output_dir="./example_checkpoints",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=512,
        num_train_epochs=1,
        warmup_steps=10,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        save_steps=50,
        logging_steps=5,
        seed=42,
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    model = Qwen3ForCausalLM(model_config)
    
    # Log model info
    memory_footprint = get_model_memory_footprint(model)
    print(f"Model memory footprint: {memory_footprint['total_gb']:.2f}GB")
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        config.dataset_path,
        tokenizer,
        config.max_seq_length,
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    print(f"Training completed: {results}")


def example_memory_optimization():
    """Example of memory optimization techniques."""
    print("\n=== Memory Optimization Example ===")
    
    # Load model
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(model_config)
    
    print("Before optimization:")
    memory_footprint = get_model_memory_footprint(model)
    print(f"Model memory: {memory_footprint}")
    
    # Apply memory optimizations
    optimizations = optimize_memory_usage(
        model=model,
        enable_gradient_checkpointing=True,
        enable_cpu_offload=False,
    )
    
    print("Memory optimizations applied:")
    for key, value in optimizations.items():
        print(f"  {key}: {value}")


def example_evaluation():
    """Example of model evaluation."""
    print("\n=== Evaluation Example ===")
    
    # Create sample dataset
    create_sample_dataset()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(model_config)
    
    # Prepare evaluation dataset
    eval_dataset = CausalLMDataset(
        data_path="data/sample_train.jsonl",
        tokenizer=tokenizer,
        max_length=512,
        split="eval"
    )
    
    # Create data collator
    data_collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        max_length=512,
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        device=device,
        use_amp=True,
    )
    
    print(f"Evaluation metrics: {metrics}")


def example_custom_training_loop():
    """Example of a custom training loop."""
    print("\n=== Custom Training Loop Example ===")
    
    # Create sample dataset
    create_sample_dataset()
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(model_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = CausalLMDataset(
        data_path="data/sample_train.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )
    
    data_collator = CausalLMDataCollator(tokenizer=tokenizer, max_length=512)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
    )
    
    # Training loop
    model.train()
    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("Custom training loop completed!")


if __name__ == "__main__":
    import torch
    
    print("Nano-vLLM Training Examples")
    print("=" * 50)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Run examples
    try:
        example_basic_training()
    except Exception as e:
        print(f"Basic training example failed: {e}")
    
    try:
        example_memory_optimization()
    except Exception as e:
        print(f"Memory optimization example failed: {e}")
    
    try:
        example_evaluation()
    except Exception as e:
        print(f"Evaluation example failed: {e}")
    
    try:
        example_custom_training_loop()
    except Exception as e:
        print(f"Custom training loop example failed: {e}")
    
    print("\nAll examples completed!")
