#!/usr/bin/env python3
"""
Verification script for from-scratch training capabilities.

This script verifies that you can train a language model completely from scratch
using only your documents, with no pre-trained model dependencies.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

import torch

from nanovllm.config import TrainingConfig
from nanovllm.training import (
    create_model_from_scratch, CustomTokenizer, analyze_document_collection,
    Trainer, set_seed
)


def create_test_documents():
    """Create test documents for verification."""
    print("📁 Creating test documents...")
    
    docs_dir = Path("./test_from_scratch_docs")
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir()
    
    # Create simple test documents
    test_docs = {
        "doc1.txt": "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing.",
        "doc2.txt": "Machine learning is a powerful tool for data analysis. Neural networks can learn complex patterns.",
        "doc3.txt": "Cooking is an art that requires practice. Good ingredients make delicious meals.",
    }
    
    for filename, content in test_docs.items():
        with open(docs_dir / filename, "w") as f:
            f.write(content)
    
    # Create JSONL file
    jsonl_data = [
        {"text": "Python is a programming language used for artificial intelligence and data science."},
        {"text": "Deep learning models require large amounts of data for training and validation."},
        {"text": "Natural language processing enables computers to understand human language."},
    ]
    
    with open(docs_dir / "data.jsonl", "w") as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Created test documents in {docs_dir}")
    return str(docs_dir)


def test_vocabulary_building(docs_path):
    """Test custom vocabulary building."""
    print("\n🔤 Testing vocabulary building...")
    
    # Test document analysis
    analysis = analyze_document_collection(docs_path)
    assert analysis['total_files'] > 0, "No files found"
    assert analysis['total_documents'] > 0, "No documents found"
    print(f"   Found {analysis['total_documents']} documents")
    
    # Test custom tokenizer
    tokenizer = CustomTokenizer(vocab_size=1000)
    vocab_stats = tokenizer.build_vocab_from_documents(docs_path)
    
    assert vocab_stats['vocab_size'] > 0, "No vocabulary built"
    assert vocab_stats['coverage'] > 0, "Zero vocabulary coverage"
    print(f"   Built vocabulary: {vocab_stats['vocab_size']} tokens")
    print(f"   Coverage: {vocab_stats['coverage']:.2%}")
    
    # Test tokenization
    test_text = "The quick brown fox"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    assert len(tokens) > 0, "No tokens generated"
    assert decoded, "Decoding failed"
    print(f"   Tokenization test: '{test_text}' -> {len(tokens)} tokens")
    
    return tokenizer


def test_model_creation(docs_path):
    """Test model creation from scratch."""
    print("\n🏗️ Testing model creation...")
    
    # Test different model sizes
    for model_size in ["tiny", "small"]:
        print(f"   Testing {model_size} model...")
        
        model, tokenizer, creation_info = create_model_from_scratch(
            documents_path=docs_path,
            model_size=model_size,
            vocab_size=1000,  # Small for testing
        )
        
        assert model is not None, f"Model creation failed for {model_size}"
        assert tokenizer is not None, f"Tokenizer creation failed for {model_size}"
        assert creation_info['parameter_count'] > 0, f"No parameters in {model_size} model"
        
        print(f"     Parameters: {creation_info['parameter_count']:,}")
        
        # Test forward pass
        test_input = tokenizer.encode("test")
        input_tensor = torch.tensor([test_input])
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            assert outputs is not None, f"Forward pass failed for {model_size}"
            
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            assert logits.shape[-1] == len(tokenizer.vocab), f"Wrong output size for {model_size}"
        
        print(f"     ✅ {model_size} model working correctly")


def test_from_scratch_training(docs_path):
    """Test complete from-scratch training."""
    print("\n🚀 Testing from-scratch training...")
    
    set_seed(42)
    
    # Create training configuration
    config = TrainingConfig(
        train_from_scratch=True,
        model_size="tiny",
        vocab_size=500,  # Very small for testing
        dataset_path=docs_path,
        output_dir="./test_from_scratch_output",
        
        # Quick training settings
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=128,
        num_train_epochs=1,
        max_train_steps=5,  # Just a few steps
        warmup_steps=1,
        
        # Disable optimizations for testing
        mixed_precision="no",
        gradient_checkpointing=False,
        
        # Minimal logging
        save_steps=10,
        logging_steps=1,
        
        seed=42,
    )
    
    print("   Creating trainer...")
    trainer = Trainer.from_scratch(config)
    
    assert trainer.model is not None, "Trainer model is None"
    assert trainer.tokenizer is not None, "Trainer tokenizer is None"
    assert trainer.train_dataset is not None, "Training dataset is None"
    
    param_count = sum(p.numel() for p in trainer.model.parameters())
    print(f"   Model parameters: {param_count:,}")
    print(f"   Training samples: {len(trainer.train_dataset)}")
    
    # Test training
    print("   Running training...")
    try:
        results = trainer.train()
        assert results is not None, "Training returned None"
        print(f"   ✅ Training completed: {results}")
        
        # Test inference
        print("   Testing inference...")
        trainer.model.eval()
        
        test_prompt = "The quick"
        inputs = trainer.tokenizer.encode(test_prompt)
        input_tensor = torch.tensor([inputs])
        
        with torch.no_grad():
            outputs = trainer.model(input_ids=input_tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
            # Get next token prediction
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = trainer.tokenizer.decode([next_token_id])
            
            print(f"   Inference test: '{test_prompt}' -> '{next_token}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False


def verify_from_scratch_capabilities():
    """Verify all from-scratch training capabilities."""
    print("🧪 NANO-VLLM FROM-SCRATCH TRAINING VERIFICATION")
    print("=" * 60)
    print("Testing complete from-scratch training capabilities...\n")
    
    success_count = 0
    total_tests = 4
    
    try:
        # Test 1: Document creation
        docs_path = create_test_documents()
        success_count += 1
        
        # Test 2: Vocabulary building
        tokenizer = test_vocabulary_building(docs_path)
        success_count += 1
        
        # Test 3: Model creation
        test_model_creation(docs_path)
        success_count += 1
        
        # Test 4: Training
        if test_from_scratch_training(docs_path):
            success_count += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 60)
        
        tests = [
            "✅ Document creation",
            "✅ Vocabulary building", 
            "✅ Model creation",
            "✅ From-scratch training" if success_count == 4 else "❌ From-scratch training"
        ]
        
        for i, test in enumerate(tests):
            if i < success_count:
                print(test)
            else:
                print(test.replace("✅", "❌"))
        
        if success_count == total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ From-scratch training is fully functional!")
            
            print("\n📖 Usage Summary:")
            print("1. 📁 Put your documents in a folder")
            print("2. ⚙️ Set train_from_scratch=True in config")
            print("3. 🚀 Run: python train.py --config config.json --train_from_scratch")
            print("4. 🔮 Get a model trained ONLY on your data!")
            
            print("\n🎯 Your From-Scratch Model Will:")
            print("  ✅ Use vocabulary built from YOUR documents only")
            print("  ✅ Know ONLY what's in your training data")
            print("  ✅ Have architecture sized for your needs")
            print("  ✅ Be completely independent (no pre-trained dependencies)")
            print("  ✅ Work with nano-vllm for inference")
            
            return True
        else:
            print(f"\n⚠️ {success_count}/{total_tests} tests passed")
            return False
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Nano-vLLM From-Scratch Training Verification")
    print("This script verifies that you can train models completely from scratch\n")
    
    success = verify_from_scratch_capabilities()
    
    if success:
        print("\n✨ FROM-SCRATCH TRAINING VERIFIED!")
        print("You can now train language models completely from scratch using only your documents!")
    else:
        print("\n⚠️ Some issues detected. Please check the implementation.")
    
    print("\n📚 For examples, run: python example_from_scratch_training.py")
    print("📖 For documentation, see: README_TRAINING.md")
