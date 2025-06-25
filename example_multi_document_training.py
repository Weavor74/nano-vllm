#!/usr/bin/env python3
"""
Example script demonstrating training on multiple documents in a folder.

This script shows how to:
1. Prepare a collection of documents for training
2. Analyze document statistics
3. Train with different document processing strategies
4. Handle various file formats (TXT, JSON, JSONL)
"""

import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.config import TrainingConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training import (
    Trainer, MultiDocumentDataset, prepare_multi_document_datasets,
    analyze_document_collection, set_seed
)


def create_sample_document_collection():
    """Create a sample collection of documents for demonstration."""
    print("Creating sample document collection...")
    
    # Create documents directory
    docs_dir = Path("./sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Create various document types
    
    # 1. Plain text files
    text_files = [
        ("machine_learning.txt", """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.

The process of machine learning involves feeding data to algorithms, allowing them to identify patterns and make predictions or decisions. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning uses labeled data to train models, while unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning involves training agents to make decisions through trial and error in an environment.
        """.strip()),
        
        ("deep_learning.txt", """
Deep Learning and Neural Networks

Deep learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain.

Neural networks consist of interconnected nodes called neurons, organized in layers. The input layer receives data, hidden layers process it through weighted connections and activation functions, and the output layer produces the final result.

Deep learning has revolutionized many fields including computer vision, natural language processing, and speech recognition. Popular architectures include convolutional neural networks (CNNs) for image processing and recurrent neural networks (RNNs) for sequential data.
        """.strip()),
        
        ("transformers.txt", """
Transformer Architecture

The Transformer architecture, introduced in the paper "Attention Is All You Need," has become the foundation for many state-of-the-art natural language processing models. Unlike previous architectures that relied on recurrence or convolution, Transformers use self-attention mechanisms.

The key innovation of Transformers is the attention mechanism, which allows the model to focus on different parts of the input sequence when processing each element. This enables parallel processing and better handling of long-range dependencies.

Transformers consist of an encoder-decoder structure, though many modern applications use only the encoder (like BERT) or only the decoder (like GPT). The architecture has enabled breakthrough models in language understanding and generation.
        """.strip()),
    ]
    
    for filename, content in text_files:
        with open(docs_dir / filename, "w") as f:
            f.write(content)
    
    # 2. JSONL file with structured documents
    jsonl_data = [
        {
            "title": "Introduction to Python",
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
            "category": "programming",
            "difficulty": "beginner"
        },
        {
            "title": "Data Structures in Python",
            "content": "Python provides several built-in data structures including lists, tuples, dictionaries, and sets. Lists are ordered and mutable, tuples are ordered and immutable, dictionaries store key-value pairs, and sets contain unique elements.",
            "category": "programming",
            "difficulty": "intermediate"
        },
        {
            "title": "Object-Oriented Programming",
            "content": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which can contain data and code. Python supports OOP through classes and objects, inheritance, encapsulation, and polymorphism.",
            "category": "programming",
            "difficulty": "intermediate"
        }
    ]
    
    with open(docs_dir / "python_tutorials.jsonl", "w") as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + "\n")
    
    # 3. JSON file with multiple documents
    json_data = [
        {
            "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            "topic": "AI Overview"
        },
        {
            "text": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            "topic": "NLP Introduction"
        }
    ]
    
    with open(docs_dir / "ai_concepts.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created sample document collection in {docs_dir}")
    print(f"Files created: {list(docs_dir.glob('*'))}")
    
    return str(docs_dir)


def demo_document_analysis(docs_path: str):
    """Demonstrate document collection analysis."""
    print("\n" + "="*60)
    print("DOCUMENT COLLECTION ANALYSIS")
    print("="*60)
    
    analysis = analyze_document_collection(docs_path)
    
    print(f"Total files: {analysis['total_files']}")
    print(f"File types: {analysis['file_types']}")
    print(f"Total documents: {analysis['total_documents']}")
    print(f"Total characters: {analysis['total_characters']:,}")
    print(f"Average document length: {analysis['average_doc_length']:.1f} characters")
    
    print("\nPer-file breakdown:")
    for file_info in analysis['files']:
        print(f"  {Path(file_info['path']).name}:")
        print(f"    Documents: {file_info['documents']}")
        print(f"    Characters: {file_info['characters']:,}")
        print(f"    Avg length: {file_info['avg_length']:.1f}")


def demo_concatenated_training(docs_path: str):
    """Demonstrate training with concatenated documents."""
    print("\n" + "="*60)
    print("CONCATENATED DOCUMENT TRAINING")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset with concatenated documents
    dataset = MultiDocumentDataset(
        data_path=docs_path,
        tokenizer=tokenizer,
        max_length=512,
        concatenate_documents=True,
        document_separator="\n\n---\n\n",  # Clear separator
        chunk_overlap=50,
        min_chunk_size=100,
    )
    
    print(f"Created dataset with {len(dataset)} chunks")
    
    # Show a sample
    sample = dataset[0]
    print(f"\nSample chunk:")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Text preview: {tokenizer.decode(sample['input_ids'][:100])}...")
    print(f"Metadata: {sample['metadata']}")


def demo_individual_document_training(docs_path: str):
    """Demonstrate training with individual document processing."""
    print("\n" + "="*60)
    print("INDIVIDUAL DOCUMENT TRAINING")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset with individual document processing
    dataset = MultiDocumentDataset(
        data_path=docs_path,
        tokenizer=tokenizer,
        max_length=256,  # Smaller chunks for individual docs
        concatenate_documents=False,
        chunk_overlap=25,
        min_chunk_size=50,
    )
    
    print(f"Created dataset with {len(dataset)} chunks")
    
    # Show samples from different documents
    print(f"\nSample chunks from different documents:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nChunk {i}:")
        print(f"  Source: {sample['metadata']['source_file']}")
        print(f"  Chunk index: {sample['metadata']['chunk_index']}")
        print(f"  Token count: {sample['metadata']['token_count']}")
        text_preview = tokenizer.decode(sample['input_ids'][:50])
        print(f"  Text preview: {text_preview}...")


def demo_full_training_pipeline(docs_path: str):
    """Demonstrate a complete training pipeline with multiple documents."""
    print("\n" + "="*60)
    print("FULL TRAINING PIPELINE")
    print("="*60)
    
    set_seed(42)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create small model for demo
    from transformers import Qwen3Config
    model_config = Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        torch_dtype="float32",
    )
    model = Qwen3ForCausalLM(model_config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_multi_document_datasets(
        data_path=docs_path,
        tokenizer=tokenizer,
        max_length=256,
        train_split_ratio=0.8,
        concatenate_documents=False,  # Process documents individually
        chunk_overlap=25,
        min_chunk_size=50,
    )
    
    print(f"Training chunks: {len(train_dataset)}")
    print(f"Evaluation chunks: {len(eval_dataset) if eval_dataset else 0}")
    
    # Create training configuration
    config = TrainingConfig(
        model_name_or_path="demo_model",
        dataset_path=docs_path,
        output_dir="./multi_doc_checkpoints",
        learning_rate=1e-3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=256,
        num_train_epochs=1,
        max_train_steps=10,  # Just a few steps for demo
        warmup_steps=2,
        mixed_precision="no",
        save_steps=5,
        eval_steps=5,
        logging_steps=1,
        seed=42,
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    print("\nStarting training...")
    try:
        results = trainer.train()
        print(f"Training completed: {results}")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def main():
    """Run the multi-document training demo."""
    print("Multi-Document Training Demo")
    print("This demo shows how to train on multiple documents in a folder")
    
    try:
        # Create sample documents
        docs_path = create_sample_document_collection()
        
        # Analyze the document collection
        demo_document_analysis(docs_path)
        
        # Demo different training approaches
        demo_concatenated_training(docs_path)
        demo_individual_document_training(docs_path)
        
        # Full training pipeline
        success = demo_full_training_pipeline(docs_path)
        
        if success:
            print("\n" + "="*60)
            print("🎉 MULTI-DOCUMENT TRAINING DEMO COMPLETED!")
            print("="*60)
            print("You can now train on your own document collections!")
            print("\nUsage examples:")
            print("1. Point to a folder: MultiDocumentDataset('/path/to/docs')")
            print("2. Mix file types: .txt, .json, .jsonl files")
            print("3. Choose concatenation vs individual processing")
            print("4. Configure chunking and overlap strategies")
        else:
            print("\n❌ Demo failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
