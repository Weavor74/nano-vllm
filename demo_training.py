#!/usr/bin/env python3
"""
Demo script showing nano-vllm training capabilities.

This script creates a minimal training setup to demonstrate the training features.
"""

import os
import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training import (
    Trainer, CausalLMDataset, CausalLMDataCollator, create_dataloader,
    set_seed, get_rank, get_world_size
)


def create_demo_dataset(num_samples: int = 50):
    """Create a demo dataset for training."""
    demo_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a versatile programming language for AI development.",
        "Deep learning models require large datasets for effective training.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of artificial intelligence.",
        "Large language models can generate coherent and contextual text.",
        "Training neural networks involves optimizing millions of parameters.",
        "Gradient descent is a fundamental optimization algorithm in machine learning.",
        "Attention mechanisms allow models to focus on relevant information.",
    ]
    
    # Create dataset with variations
    dataset = []
    for i in range(num_samples):
        base_text = demo_texts[i % len(demo_texts)]
        # Add some variation
        text = f"Sample {i+1}: {base_text}"
        dataset.append({"text": text})
    
    return dataset


def setup_demo_environment():
    """Set up the demo environment."""
    print("Setting up demo environment...")
    
    # Create temporary directory for demo
    demo_dir = Path("./demo_training")
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo dataset
    dataset = create_demo_dataset(50)
    dataset_path = demo_dir / "demo_data.jsonl"
    
    with open(dataset_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    print(f"Created demo dataset with {len(dataset)} samples at {dataset_path}")
    return str(dataset_path), str(demo_dir)


def create_small_model_config():
    """Create a small model configuration for demo purposes."""
    return {
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }


def demo_basic_training():
    """Demonstrate basic training functionality."""
    print("\n" + "="*60)
    print("DEMO: Basic Training with Nano-vLLM")
    print("="*60)
    
    # Setup
    dataset_path, output_dir = setup_demo_environment()
    set_seed(42)
    
    # Create a simple tokenizer for demo
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create small model for demo
    print("Creating model...")
    from transformers import Qwen3Config
    model_config = Qwen3Config(**create_small_model_config())
    model = Qwen3ForCausalLM(model_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training configuration
    config = TrainingConfig(
        model_name_or_path="demo_model",  # Not used for this demo
        dataset_path=dataset_path,
        output_dir=output_dir + "/checkpoints",
        learning_rate=1e-3,  # Higher LR for demo
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=128,
        num_train_epochs=1,
        max_train_steps=10,  # Just a few steps for demo
        warmup_steps=2,
        mixed_precision="no",  # Disable for demo simplicity
        gradient_checkpointing=False,
        save_steps=5,
        eval_steps=5,
        logging_steps=1,
        seed=42,
    )
    
    print(f"Training configuration:")
    print(f"  - Batch size: {config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.effective_batch_size}")
    print(f"  - Max steps: {config.max_train_steps}")
    print(f"  - Learning rate: {config.learning_rate}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = CausalLMDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    
    # Split for evaluation
    eval_size = min(10, len(train_dataset) // 5)
    eval_examples = train_dataset.examples[:eval_size]
    train_examples = train_dataset.examples[eval_size:]
    
    # Create separate datasets
    eval_dataset = CausalLMDataset.__new__(CausalLMDataset)
    eval_dataset.tokenizer = tokenizer
    eval_dataset.max_length = config.max_seq_length
    eval_dataset.split = "eval"
    eval_dataset.examples = eval_examples
    
    train_dataset.examples = train_examples
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    print("\nStarting training...")
    print("-" * 40)
    
    try:
        results = trainer.train()
        
        print("-" * 40)
        print("Training completed successfully!")
        print(f"Results: {results}")
        
        # Test inference with trained model
        print("\nTesting inference with trained model...")
        model.eval()
        
        test_input = "Sample text for testing:"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"])
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
        print(f"Input: {test_input}")
        print(f"Output shape: {logits.shape}")
        print("✓ Inference successful!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def demo_memory_optimization():
    """Demonstrate memory optimization features."""
    print("\n" + "="*60)
    print("DEMO: Memory Optimization")
    print("="*60)
    
    from nanovllm.training.memory import get_model_memory_footprint, estimate_training_memory
    
    # Create model
    from transformers import Qwen3Config
    model_config = Qwen3Config(**create_small_model_config())
    model = Qwen3ForCausalLM(model_config)
    
    # Calculate memory footprint
    memory_footprint = get_model_memory_footprint(model)
    print(f"Model memory footprint:")
    print(f"  - Parameters: {memory_footprint['parameters_mb']:.2f} MB")
    print(f"  - Buffers: {memory_footprint['buffers_mb']:.2f} MB")
    print(f"  - Total: {memory_footprint['total_mb']:.2f} MB")
    
    # Estimate training memory
    memory_estimate = estimate_training_memory(
        model=model,
        batch_size=2,
        sequence_length=128,
        gradient_accumulation_steps=2,
        mixed_precision=False,
        gradient_checkpointing=False,
    )
    
    print(f"\nEstimated training memory:")
    print(f"  - Model: {memory_estimate['model_gb']:.3f} GB")
    print(f"  - Activations: {memory_estimate['activations_gb']:.3f} GB")
    print(f"  - Gradients: {memory_estimate['gradients_gb']:.3f} GB")
    print(f"  - Optimizer: {memory_estimate['optimizer_gb']:.3f} GB")
    print(f"  - Total: {memory_estimate['total_gb']:.3f} GB")


def main():
    """Run the demo."""
    print("Nano-vLLM Training Demo")
    print("This demo shows the training capabilities of nano-vllm")
    
    # Check PyTorch availability
    print(f"\nEnvironment:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    
    # Run demos
    try:
        # Memory optimization demo
        demo_memory_optimization()
        
        # Basic training demo
        success = demo_basic_training()
        
        if success:
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("The nano-vllm training system is working correctly.")
            print("You can now use it to train your own language models!")
            print("\nNext steps:")
            print("1. Prepare your training data in JSONL format")
            print("2. Create a training configuration file")
            print("3. Run: python train.py --config your_config.json")
            print("4. For distributed training: torchrun --nproc_per_node=N train.py --config your_config.json")
        else:
            print("\n❌ Demo failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
