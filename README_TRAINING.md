# Nano-vLLM Training Guide: Complete Usage Manual

This comprehensive guide covers everything you need to know about training language models with nano-vllm, including fine-tuning and from-scratch training capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Approaches](#training-approaches)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Fine-Tuning Training](#fine-tuning-training)
6. [From-Scratch Training](#from-scratch-training)
7. [Multi-Document Training](#multi-document-training)
8. [Configuration](#configuration)
9. [Advanced Features](#advanced-features)
10. [Distributed Training](#distributed-training)
11. [Memory Optimization](#memory-optimization)
12. [Troubleshooting](#troubleshooting)
13. [Examples](#examples)

## Quick Start

### 30-Second Fine-Tuning Setup

```bash
# 1. Install nano-vllm with training dependencies
pip install torch transformers datasets accelerate flash-attn

# 2. Prepare your data (any of these formats work)
mkdir my_documents
echo "Your training text here..." > my_documents/doc1.txt
echo '{"text": "More training data..."}' > my_documents/data.jsonl

# 3. Create fine-tuning config
cat > finetune_config.json << EOF
{
  "model_name_or_path": "Qwen/Qwen3-0.6B",
  "dataset_path": "./my_documents",
  "output_dir": "./checkpoints",
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 2,
  "num_train_epochs": 3
}
EOF

# 4. Start fine-tuning
python train.py --config finetune_config.json --use_multi_document
```

### 30-Second From-Scratch Setup

```bash
# 1. Same installation and data preparation as above

# 2. Create from-scratch config
cat > scratch_config.json << EOF
{
  "train_from_scratch": true,
  "model_size": "small",
  "dataset_path": "./my_documents",
  "output_dir": "./my_custom_model",
  "learning_rate": 1e-3,
  "per_device_train_batch_size": 2,
  "num_train_epochs": 10
}
EOF

# 3. Start from-scratch training
python train.py --config scratch_config.json --train_from_scratch --use_multi_document
```

## Training Approaches

Nano-vLLM supports two distinct training approaches:

### 🔧 **Fine-Tuning** (Recommended for most users)
- **Start with**: Pre-trained model (e.g., Qwen3, GPT-2)
- **Adapt to**: Your specific domain/task
- **Retains**: General knowledge + learns your data
- **Training time**: Faster (1-3 epochs)
- **Data requirements**: Less data needed
- **Use case**: Specializing existing knowledge

### 🏗️ **From-Scratch** (For complete control)
- **Start with**: Random weights (no pre-training)
- **Learn only**: Your documents
- **Knows**: Only what you teach it
- **Training time**: Longer (10+ epochs)
- **Data requirements**: More data needed
- **Use case**: Domain-specific, private models

### 📊 **Comparison Table**

| Aspect | Fine-Tuning | From-Scratch |
|--------|-------------|--------------|
| **Starting Point** | Pre-trained model | Random weights |
| **Knowledge** | General + Your data | Only your data |
| **Training Time** | Fast (hours) | Slow (days) |
| **Data Needed** | Less (MB-GB) | More (GB-TB) |
| **Model Size** | Fixed by base model | Customizable |
| **Privacy** | Contains external knowledge | 100% your data |
| **Use Case** | Task adaptation | Domain specialization |

## Installation

### Basic Installation
```bash
pip install torch transformers datasets accelerate
```

### With Optimizations
```bash
pip install torch transformers datasets accelerate
pip install flash-attn  # For attention optimization
pip install triton      # For custom kernels
```

### Development Installation
```bash
git clone https://github.com/your-repo/nano-vllm
cd nano-vllm
pip install -e .
```

## Data Preparation

### Supported Data Formats

Nano-vLLM supports multiple data formats for maximum flexibility:

#### 1. Plain Text Files (`.txt`)
```
documents/
├── chapter1.txt
├── chapter2.txt
└── appendix.txt
```

Each `.txt` file is treated as a single document.

#### 2. JSON Lines (`.jsonl`)
```jsonl
{"text": "First training example..."}
{"text": "Second training example..."}
{"content": "Alternative field name works too"}
{"input": "Question", "output": "Answer"}
```

#### 3. JSON Files (`.json`)
```json
[
  {"text": "Document 1 content..."},
  {"text": "Document 2 content..."}
]
```

Or single document:
```json
{"text": "Single document content..."}
```

#### 4. Mixed Collections
```
my_training_data/
├── books/
│   ├── book1.txt
│   └── book2.txt
├── articles.jsonl
├── papers.json
└── notes/
    ├── meeting_notes.txt
    └── research_ideas.jsonl
```

### Data Quality Guidelines

1. **Text Quality**: Ensure clean, well-formatted text
2. **Encoding**: Use UTF-8 encoding for all files
3. **Size**: No strict limits, but consider memory constraints
4. **Structure**: Consistent field names across JSON files

## Fine-Tuning Training

Fine-tuning adapts a pre-trained model to your specific domain or task while retaining general knowledge.

### Basic Fine-Tuning Setup

```python
from nanovllm import Trainer, TrainingConfig

# Fine-tuning configuration
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",  # Pre-trained model
    dataset_path="data/train.jsonl",       # Your data
    output_dir="./finetuned_model",

    # Fine-tuning optimized settings
    learning_rate=5e-5,                    # Lower LR for fine-tuning
    per_device_train_batch_size=2,
    num_train_epochs=3,                    # Fewer epochs needed
    warmup_steps=100,
)

# Create trainer from pre-trained model
trainer = Trainer.from_pretrained(config)
trainer.train()
```

### Command Line Fine-Tuning
```bash
# Basic fine-tuning
python train.py --config finetune_config.json

# Multi-document fine-tuning
python train.py --config finetune_config.json --use_multi_document
```

### Fine-Tuning Best Practices

1. **Lower Learning Rates**: Use 1e-5 to 1e-4 to avoid catastrophic forgetting
2. **Fewer Epochs**: 1-3 epochs usually sufficient
3. **Gradual Unfreezing**: Optionally freeze early layers initially
4. **Validation**: Monitor for overfitting on small datasets

## From-Scratch Training

From-scratch training builds a completely custom model using only your documents.

### Basic From-Scratch Setup

```python
from nanovllm import Trainer, TrainingConfig

# From-scratch configuration
config = TrainingConfig(
    train_from_scratch=True,               # Key setting!
    model_size="small",                    # tiny/small/medium/large
    vocab_size=None,                       # Auto-determined from data
    dataset_path="./my_documents",         # Your document collection
    output_dir="./my_custom_model",

    # From-scratch optimized settings
    learning_rate=1e-3,                    # Higher LR for from-scratch
    per_device_train_batch_size=4,
    num_train_epochs=10,                   # More epochs needed
    warmup_steps=1000,                     # Longer warmup
)

# Create trainer for from-scratch training
trainer = Trainer.from_scratch(config)
trainer.train()
```

### Command Line From-Scratch Training
```bash
# Basic from-scratch training
python train.py --config scratch_config.json --train_from_scratch

# With document analysis
python train.py --config scratch_config.json --train_from_scratch --analyze_documents

# Custom model size
python train.py --config scratch_config.json --train_from_scratch --model_size medium
```

### Model Sizes for From-Scratch Training

| Size | Hidden | Layers | Heads | Parameters | Use Case |
|------|--------|--------|-------|------------|----------|
| **tiny** | 256 | 4 | 4 | ~1M | Testing, small domains |
| **small** | 512 | 8 | 8 | ~10M | Specialized domains |
| **medium** | 1024 | 16 | 16 | ~100M | Complex domains |
| **large** | 2048 | 24 | 32 | ~500M | Large-scale applications |

### Custom Vocabulary Building

From-scratch training automatically builds vocabulary from your documents:

```python
from nanovllm.training import create_model_from_scratch, analyze_document_collection

# Analyze your documents first
analysis = analyze_document_collection("./my_documents")
print(f"Found {analysis['total_documents']} documents")

# Create model with custom vocabulary
model, tokenizer, info = create_model_from_scratch(
    documents_path="./my_documents",
    model_size="small",
    vocab_size=16000,  # Or None for auto-determination
)

print(f"Vocabulary size: {info['vocab_stats']['vocab_size']}")
print(f"Coverage: {info['vocab_stats']['coverage']:.2%}")
```

### From-Scratch Best Practices

1. **More Data**: Need substantial data (GB+ recommended)
2. **Higher Learning Rates**: Start with 1e-3, adjust based on convergence
3. **More Epochs**: 10-50 epochs typically needed
4. **Longer Warmup**: Use 1000+ warmup steps
5. **Monitor Carefully**: Watch for proper convergence patterns

## Multi-Document Training

### Overview

Multi-document training allows you to train on entire collections of documents with intelligent chunking and processing strategies.

### Basic Multi-Document Setup

```python
from nanovllm.training import MultiDocumentDataset, analyze_document_collection

# 1. Analyze your documents first
analysis = analyze_document_collection("./documents")
print(f"Found {analysis['total_documents']} documents")
print(f"File types: {analysis['file_types']}")
print(f"Total characters: {analysis['total_characters']:,}")

# 2. Create dataset
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    max_length=2048,
    concatenate_documents=False,  # Process individually
    chunk_overlap=128,
    min_chunk_size=100,
)

print(f"Created {len(dataset)} training chunks")
```

### Processing Strategies

#### Strategy 1: Individual Document Processing
Best for maintaining document boundaries and structure.

```python
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    concatenate_documents=False,  # Keep documents separate
    chunk_overlap=50,             # Small overlap
    min_chunk_size=100,          # Minimum viable chunk
)
```

**Use when:**
- Documents have distinct topics/styles
- You want to preserve document structure
- Training on diverse content types

#### Strategy 2: Concatenated Processing
Best for creating continuous training sequences.

```python
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    concatenate_documents=True,        # Combine all documents
    document_separator="\n\n---\n\n",  # Clear separator
    chunk_overlap=128,                 # Larger overlap
)
```

**Use when:**
- Documents are similar in style/topic
- You want maximum context length
- Training on homogeneous content

### Command Line Multi-Document Training

```bash
# Basic multi-document training
python train.py --config config.json --use_multi_document

# With document analysis
python train.py --config config.json --use_multi_document --analyze_documents

# With concatenation
python train.py --config config.json --use_multi_document --concatenate_documents

# Custom chunk overlap
python train.py --config config.json --use_multi_document --chunk_overlap 256
```

### Advanced Multi-Document Features

#### Document Analysis
```python
from nanovllm.training import analyze_document_collection

analysis = analyze_document_collection("./documents")

# Detailed breakdown
for file_info in analysis['files']:
    print(f"{file_info['path']}:")
    print(f"  Documents: {file_info['documents']}")
    print(f"  Characters: {file_info['characters']:,}")
    print(f"  Avg length: {file_info['avg_length']:.1f}")
```

#### Custom Text Extraction
The system automatically handles various field names:
- `text`, `content`, `body` for main content
- `input` + `output` for Q&A pairs
- Automatic concatenation of string fields

#### Chunk Metadata
Each training chunk includes metadata:
```python
sample = dataset[0]
metadata = sample['metadata']
print(f"Source: {metadata['source_file']}")
print(f"Chunk: {metadata['chunk_index']}")
print(f"Tokens: {metadata['token_count']}")
```

## Configuration

### Complete Configuration Reference

```json
{
  "model_name_or_path": "Qwen/Qwen3-0.6B",
  "dataset_path": "./documents",
  "output_dir": "./checkpoints",
  
  "learning_rate": 5e-5,
  "weight_decay": 0.01,
  "beta1": 0.9,
  "beta2": 0.95,
  
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 2048,
  
  "num_train_epochs": 3,
  "warmup_steps": 100,
  "lr_scheduler_type": "cosine",
  
  "mixed_precision": "bf16",
  "gradient_clipping": 1.0,
  "gradient_checkpointing": true,
  
  "eval_steps": 500,
  "save_steps": 1000,
  "logging_steps": 10,
  
  "seed": 42
}
```

## Advanced Features

### Memory Optimization

#### Gradient Checkpointing
```python
config = TrainingConfig(
    gradient_checkpointing=True,  # Reduces memory usage
    mixed_precision="bf16",       # Further memory savings
)
```

#### Memory Estimation
```python
from nanovllm.training.memory import estimate_training_memory

memory_estimate = estimate_training_memory(
    model=model,
    batch_size=2,
    sequence_length=2048,
    gradient_checkpointing=True,
    mixed_precision=True,
)

print(f"Estimated memory: {memory_estimate['total_gb']:.2f}GB")
```

### Custom Training Loops

```python
from nanovllm.training import create_optimizer, create_scheduler

# Custom optimizer
optimizer = create_optimizer(
    model=model,
    learning_rate=1e-4,
    weight_decay=0.01,
)

# Custom scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    num_warmup_steps=100,
    num_training_steps=1000,
)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Evaluation During Training

```python
from nanovllm.training.evaluation import evaluate_model

# Evaluate model
metrics = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_dataloader=eval_dataloader,
    device=device,
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Loss: {metrics.loss:.4f}")
```

### Checkpoint Management

```python
from nanovllm.training.checkpoint import CheckpointManager

# Advanced checkpoint management
checkpoint_manager = CheckpointManager(
    output_dir="./checkpoints",
    save_total_limit=5,
)

# Save with metadata
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    step=1000,
    epoch=1,
    loss=2.5,
    config=config,
    is_best=True,
)
```

## Distributed Training

### Tensor Parallelism

Split model layers across GPUs:

```json
{
  "tensor_parallel_size": 4,
  "per_device_train_batch_size": 1
}
```

```bash
torchrun --nproc_per_node=4 train.py --config config.json
```

### Data Parallelism

Replicate model across GPUs:

```bash
torchrun --nproc_per_node=4 train.py --config config.json
```

### Mixed Parallelism

Combine tensor and data parallelism:

```json
{
  "tensor_parallel_size": 2,
  "data_parallel_size": 2
}
```

```bash
torchrun --nproc_per_node=4 train.py --config config.json
```

## Memory Optimization

### Strategies by Model Size

#### Small Models (< 1B parameters)
```json
{
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 2,
  "mixed_precision": "bf16"
}
```

#### Medium Models (1B - 7B parameters)
```json
{
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true
}
```

#### Large Models (> 7B parameters)
```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "tensor_parallel_size": 4
}
```

### Memory Monitoring

```python
from nanovllm.training.memory import MemoryManager

memory_manager = MemoryManager(device)

with memory_manager.memory_profiling("training_step"):
    # Your training code here
    pass

memory_manager.log_memory_usage("After training step")
```

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Enable optimizations
--gradient_checkpointing --mixed_precision bf16
```

#### Slow Training
```bash
# Increase batch size if memory allows
--per_device_train_batch_size 4

# Use mixed precision
--mixed_precision bf16

# Multiple workers for data loading
--dataloader_num_workers 4
```

#### Poor Convergence
```bash
# Adjust learning rate
--learning_rate 1e-5

# Add warmup
--warmup_steps 100

# Check gradient clipping
--gradient_clipping 1.0
```

#### Multi-Document Issues

**Problem**: Documents not loading
```bash
# Check file formats and permissions
python -c "from nanovllm.training import analyze_document_collection; print(analyze_document_collection('./documents'))"
```

**Problem**: Chunks too small/large
```python
# Adjust chunking parameters
dataset = MultiDocumentDataset(
    chunk_overlap=64,      # Reduce overlap
    min_chunk_size=50,     # Lower minimum
    max_length=1024,       # Smaller chunks
)
```

## Examples

### Example 1: Fine-Tuning on Research Papers

```python
from nanovllm import Trainer, TrainingConfig
from nanovllm.training import analyze_document_collection

# Analyze papers first
analysis = analyze_document_collection("./research_papers")
print(f"Found {analysis['total_documents']} papers")

# Configure for academic fine-tuning
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-1.8B",  # Pre-trained base
    dataset_path="./research_papers",
    output_dir="./finetuned_paper_model",

    # Fine-tuning settings
    learning_rate=3e-5,                    # Lower LR for fine-tuning
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,                    # Few epochs for fine-tuning
    warmup_steps=100,

    # Academic content settings
    max_seq_length=4096,                   # Long context for papers
    concatenate_documents=False,           # Keep paper boundaries
    chunk_overlap=256,

    # Optimization
    mixed_precision="bf16",
    gradient_checkpointing=True,
)

# Fine-tune
trainer = Trainer.from_pretrained(config)
trainer.train()
```

### Example 2: From-Scratch Medical Domain Model

```python
# Create specialized medical model from scratch
config = TrainingConfig(
    train_from_scratch=True,               # From-scratch training
    model_size="medium",                   # Larger for complex domain
    vocab_size=25000,                      # Medical terminology
    dataset_path="./medical_documents",
    output_dir="./medical_model_scratch",

    # From-scratch optimized settings
    learning_rate=1e-3,                    # Higher LR for from-scratch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,                   # More epochs needed
    warmup_steps=2000,                     # Longer warmup

    # Medical content settings
    max_seq_length=2048,
    concatenate_documents=False,           # Preserve document structure
    chunk_overlap=128,

    # Optimization
    mixed_precision="bf16",
    gradient_checkpointing=True,
)

# Train from scratch
trainer = Trainer.from_scratch(config)
trainer.train()
```

### Example 3: Fine-Tuning vs From-Scratch Comparison

```python
# Fine-tuning approach
finetune_config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",
    dataset_path="./legal_docs",
    output_dir="./legal_finetuned",
    learning_rate=5e-5,                    # Lower LR
    num_train_epochs=3,                    # Fewer epochs
    warmup_steps=100,
)

# From-scratch approach
scratch_config = TrainingConfig(
    train_from_scratch=True,
    model_size="small",
    dataset_path="./legal_docs",
    output_dir="./legal_scratch",
    learning_rate=1e-3,                    # Higher LR
    num_train_epochs=15,                   # More epochs
    warmup_steps=1000,
)

# Compare results
finetune_trainer = Trainer.from_pretrained(finetune_config)
scratch_trainer = Trainer.from_scratch(scratch_config)

# Fine-tuned model: Knows general knowledge + legal docs
# From-scratch model: Knows ONLY legal docs
```

### Example 4: Custom Vocabulary Analysis

```python
from nanovllm.training import create_model_from_scratch, CustomTokenizer

# Build custom vocabulary for specialized domain
tokenizer = CustomTokenizer(vocab_size=15000)
vocab_stats = tokenizer.build_vocab_from_documents("./chemistry_papers")

print(f"Vocabulary coverage: {vocab_stats['coverage']:.2%}")
print(f"Most common tokens: {vocab_stats['most_common'][:10]}")

# Create model with custom vocabulary
model, custom_tokenizer, info = create_model_from_scratch(
    documents_path="./chemistry_papers",
    model_size="small",
    vocab_size=15000,
)

print(f"Model knows only chemistry terminology from your papers!")
```

### Example 5: Distributed Training Comparison

```bash
# Fine-tuning with multiple GPUs
torchrun --nproc_per_node=4 train.py \
  --config finetune_config.json \
  --use_multi_document

# From-scratch with multiple GPUs
torchrun --nproc_per_node=4 train.py \
  --config scratch_config.json \
  --train_from_scratch \
  --use_multi_document \
  --model_size medium
```

### Example 6: Domain-Specific Applications

#### Legal Document Processing
```python
# From-scratch for maximum privacy
legal_config = TrainingConfig(
    train_from_scratch=True,
    model_size="medium",
    dataset_path="./legal_cases",
    learning_rate=8e-4,
    num_train_epochs=25,
    # Result: Model knows ONLY your legal documents
)
```

#### Company Internal Knowledge Base
```python
# Fine-tuning for general knowledge + company info
company_config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-1.8B",
    dataset_path="./company_docs",
    learning_rate=3e-5,
    num_train_epochs=2,
    # Result: General knowledge + company-specific information
)
```

#### Research Domain Specialization
```python
# From-scratch for pure research focus
research_config = TrainingConfig(
    train_from_scratch=True,
    model_size="large",
    dataset_path="./research_papers",
    vocab_size=40000,
    learning_rate=6e-4,
    num_train_epochs=30,
    # Result: Research-only model with domain vocabulary
)
```

## Best Practices

### Choosing Training Approach

#### Use Fine-Tuning When:
- ✅ You want to retain general knowledge
- ✅ You have limited training data (< 1GB)
- ✅ You need faster training (hours vs days)
- ✅ You want to adapt existing capabilities
- ✅ Your domain benefits from general knowledge

#### Use From-Scratch When:
- ✅ You need maximum privacy (no external knowledge)
- ✅ You have substantial domain data (> 1GB)
- ✅ Your domain is highly specialized
- ✅ You want complete control over model knowledge
- ✅ You need custom vocabulary for your domain

### Data Preparation
1. **Clean your data**: Remove corrupted files, fix encoding issues
2. **Analyze first**: Always run document analysis before training
3. **Test chunking**: Verify chunk sizes work with your model
4. **Consistent format**: Use consistent field names across JSON files
5. **Domain focus**: For from-scratch, ensure data is domain-focused

### Training Configuration

#### Fine-Tuning Best Practices:
1. **Lower learning rates**: 1e-5 to 1e-4 to avoid catastrophic forgetting
2. **Fewer epochs**: 1-3 epochs usually sufficient
3. **Shorter warmup**: 100-500 steps typically enough
4. **Monitor overfitting**: Watch validation loss carefully
5. **Gradual unfreezing**: Consider freezing early layers initially

#### From-Scratch Best Practices:
1. **Higher learning rates**: 1e-3 to 1e-4 for proper convergence
2. **More epochs**: 10-50 epochs typically needed
3. **Longer warmup**: 1000+ steps for stable training
4. **More data**: Ensure sufficient data for vocabulary coverage
5. **Monitor convergence**: Watch for proper loss reduction patterns

### Multi-Document Strategy
1. **Individual processing**: For diverse content types
2. **Concatenated processing**: For similar, related content
3. **Appropriate overlap**: Balance context and efficiency
4. **Metadata tracking**: Use chunk metadata for debugging

### Performance Optimization
1. **Mixed precision**: Always use bf16 on modern GPUs
2. **Gradient checkpointing**: For memory-constrained training
3. **Batch size tuning**: Find the sweet spot for your hardware
4. **Data loading**: Use multiple workers when possible
5. **Model sizing**: Choose appropriate model size for your data and compute

## Getting Help

### Debug Information

#### For Fine-Tuning Issues:
```python
# Check pre-trained model loading
from transformers import AutoTokenizer, AutoConfig
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
print(f"Model config: {config}")

# Test fine-tuning dataset
from nanovllm.training import prepare_datasets
train_dataset, eval_dataset = prepare_datasets("./your_data", tokenizer, 512)
print(f"Fine-tuning data: {len(train_dataset)} samples")
```

#### For From-Scratch Issues:
```python
# Check document analysis
from nanovllm.training import analyze_document_collection
analysis = analyze_document_collection("./your_data")
print(f"Documents: {analysis['total_documents']}")
print(f"Characters: {analysis['total_characters']:,}")

# Test vocabulary building
from nanovllm.training import CustomTokenizer
tokenizer = CustomTokenizer(vocab_size=8000)
vocab_stats = tokenizer.build_vocab_from_documents("./your_data")
print(f"Vocabulary coverage: {vocab_stats['coverage']:.2%}")

# Test model creation
from nanovllm.training import create_model_from_scratch
model, tokenizer, info = create_model_from_scratch("./your_data", "small")
print(f"Model parameters: {info['parameter_count']:,}")
```

#### General Debugging:
```python
# Memory estimation
from nanovllm.training.memory import estimate_training_memory
estimate = estimate_training_memory(model, batch_size=2, sequence_length=2048)
print(f"Estimated memory: {estimate['total_gb']:.2f}GB")

# Test dataset creation
from nanovllm.training import MultiDocumentDataset
dataset = MultiDocumentDataset("./your_data", tokenizer, max_length=512)
print(f"Created {len(dataset)} chunks from {len(dataset.documents)} documents")
```

### Community Resources
- GitHub Issues: Report bugs and request features
- Documentation: Comprehensive guides and API reference
- Examples: Working code examples for common scenarios

---

## Summary

Nano-vLLM provides two powerful training approaches:

### 🔧 **Fine-Tuning**: Best for Most Users
- **Quick setup**: Use existing pre-trained models
- **Fast training**: Hours instead of days
- **Retains knowledge**: General knowledge + your specialization
- **Less data needed**: Works with smaller datasets
- **Recommended for**: Task adaptation, domain specialization with general knowledge

### 🏗️ **From-Scratch**: Maximum Control & Privacy
- **Complete control**: Build everything from your data
- **Pure privacy**: No external knowledge contamination
- **Custom vocabulary**: Built specifically from your documents
- **Domain focus**: Knows only what you teach it
- **Recommended for**: Highly specialized domains, maximum privacy needs

### 🎯 **Choose Your Path**

| Your Goal | Approach | Command |
|-----------|----------|---------|
| Adapt existing model to your domain | Fine-Tuning | `python train.py --config finetune_config.json` |
| Create domain-specific private model | From-Scratch | `python train.py --config scratch_config.json --train_from_scratch` |
| Train on document collection | Either + Multi-Doc | `--use_multi_document` |
| Analyze documents first | Either | `--analyze_documents` |

**Both approaches work seamlessly with nano-vllm's inference engine!**

---

**Happy Training with Nano-vLLM! 🚀**

This guide covers everything you need to train language models using fine-tuning or from-scratch approaches on single files or entire document collections. Start with the quick start section and choose the approach that best fits your needs.

### Multi-Document Specific Options

```json
{
  "_comment": "Multi-document training options",
  "use_multi_document": true,
  "concatenate_documents": false,
  "document_separator": "\n\n---\n\n",
  "chunk_overlap": 128,
  "min_chunk_size": 100,
  "analyze_documents": true
}
```

### Configuration Profiles

#### Profile 1: Fine-Tuning Small Model
```json
{
  "model_name_or_path": "Qwen/Qwen3-0.6B",
  "train_from_scratch": false,
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 2,
  "num_train_epochs": 3,
  "warmup_steps": 100,
  "mixed_precision": "bf16"
}
```

#### Profile 2: From-Scratch Small Domain
```json
{
  "train_from_scratch": true,
  "model_size": "small",
  "vocab_size": null,
  "learning_rate": 1e-3,
  "per_device_train_batch_size": 4,
  "num_train_epochs": 15,
  "warmup_steps": 1000,
  "mixed_precision": "bf16"
}
```

#### Profile 3: Fine-Tuning Large Model
```json
{
  "model_name_or_path": "Qwen/Qwen3-7B",
  "train_from_scratch": false,
  "learning_rate": 1e-5,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "num_train_epochs": 2,
  "tensor_parallel_size": 4,
  "gradient_checkpointing": true
}
```

#### Profile 4: From-Scratch Large Domain
```json
{
  "train_from_scratch": true,
  "model_size": "medium",
  "vocab_size": 32000,
  "learning_rate": 8e-4,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "num_train_epochs": 25,
  "warmup_steps": 2000,
  "tensor_parallel_size": 2
}
```

#### Profile 5: Multi-Document Research Papers
```json
{
  "dataset_path": "./research_papers",
  "concatenate_documents": false,
  "chunk_overlap": 256,
  "min_chunk_size": 200,
  "max_seq_length": 4096,
  "document_separator": "\n\n=== PAPER BREAK ===\n\n"
}
```
