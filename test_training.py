#!/usr/bin/env python3
"""
Test script to verify the training implementation works correctly.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training import (
    CausalLMDataset, CausalLMDataCollator, create_dataloader,
    create_optimizer, create_scheduler, set_seed
)
from nanovllm.training.evaluation import evaluate_model
from nanovllm.training.memory import get_model_memory_footprint


def create_test_dataset(data_dir: str, num_samples: int = 10):
    """Create a small test dataset."""
    test_data = [
        {"text": f"This is test sample number {i}. It contains some text for training."} 
        for i in range(num_samples)
    ]
    
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, "test_data.jsonl")
    
    with open(dataset_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    return dataset_path


def test_model_creation():
    """Test model and tokenizer creation."""
    print("Testing model creation...")
    
    # Use a small model config for testing
    config = {
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
    
    from transformers import Qwen3Config
    model_config = Qwen3Config(**config)
    model = Qwen3ForCausalLM(model_config)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, labels=labels)
    assert "loss" in outputs
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config["vocab_size"])
    
    print("✓ Model creation and forward pass successful")
    return model, model_config


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        dataset_path = create_test_dataset(temp_dir, num_samples=20)
        
        # Create a simple tokenizer (for testing)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        dataset = CausalLMDataset(
            data_path=dataset_path,
            tokenizer=tokenizer,
            max_length=128,
        )
        
        assert len(dataset) == 20
        
        # Test data collator
        collator = CausalLMDataCollator(
            tokenizer=tokenizer,
            max_length=128,
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=2,
            collate_fn=collator,
            shuffle=False,
            distributed=False,
        )
        
        # Test batch
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2  # batch size
        
        print("✓ Data loading successful")
        return dataset, dataloader, tokenizer


def test_optimizer_and_scheduler():
    """Test optimizer and scheduler creation."""
    print("Testing optimizer and scheduler...")
    
    model, _ = test_model_creation()
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        learning_rate=1e-4,
        weight_decay=0.01,
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        num_warmup_steps=10,
        num_training_steps=100,
    )
    
    # Test optimizer step
    initial_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    print("✓ Optimizer and scheduler creation successful")
    return optimizer, scheduler


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    model, _ = test_model_creation()
    dataset, dataloader, tokenizer = test_data_loading()
    optimizer, scheduler = test_optimizer_and_scheduler()
    
    model.train()
    batch = next(iter(dataloader))
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs["loss"]
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    print(f"✓ Training step successful, loss: {loss.item():.4f}")


def test_evaluation():
    """Test evaluation functionality."""
    print("Testing evaluation...")
    
    model, _ = test_model_creation()
    dataset, dataloader, tokenizer = test_data_loading()
    
    device = torch.device("cpu")  # Use CPU for testing
    model.to(device)
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=dataloader,
        device=device,
        use_amp=False,  # Disable AMP for CPU
        max_eval_samples=5,
    )
    
    assert hasattr(metrics, 'loss')
    assert hasattr(metrics, 'perplexity')
    assert metrics.perplexity > 0
    
    print(f"✓ Evaluation successful, perplexity: {metrics.perplexity:.2f}")


def test_memory_utilities():
    """Test memory optimization utilities."""
    print("Testing memory utilities...")
    
    model, _ = test_model_creation()
    
    # Test memory footprint calculation
    memory_footprint = get_model_memory_footprint(model)
    assert "total_gb" in memory_footprint
    assert memory_footprint["total_gb"] > 0
    
    print(f"✓ Memory utilities successful, model size: {memory_footprint['total_mb']:.2f}MB")


def test_training_config():
    """Test training configuration."""
    print("Testing training configuration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TrainingConfig(
            model_name_or_path="gpt2",
            dataset_path="dummy_path",
            output_dir=temp_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=128,
            num_train_epochs=1,
            mixed_precision="no",  # Disable for testing
            seed=42,
        )
        
        # Test configuration validation
        assert config.learning_rate > 0
        assert config.effective_batch_size == 2  # batch_size * grad_accum * data_parallel
        assert os.path.exists(config.output_dir)
        
        print("✓ Training configuration successful")


def run_all_tests():
    """Run all tests."""
    print("Running nano-vllm training tests...")
    print("=" * 50)
    
    try:
        test_training_config()
        test_model_creation()
        test_data_loading()
        test_optimizer_and_scheduler()
        test_training_step()
        test_evaluation()
        test_memory_utilities()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("The training implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 Training implementation is ready to use!")
        print("You can now train language models with nano-vllm.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        exit(1)
