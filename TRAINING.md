# Nano-vLLM Training Guide

This guide covers the training capabilities added to nano-vllm, enabling you to train language models from scratch with high performance and advanced optimization features.

## Features

- 🚀 **Distributed Training**: Tensor parallelism and data parallelism support
- 🔥 **Mixed Precision**: FP16/BF16 training for memory efficiency
- 💾 **Memory Optimization**: Gradient checkpointing and CPU offloading
- 📊 **Advanced Optimizers**: AdamW with weight decay and learning rate scheduling
- 🎯 **Gradient Management**: Gradient accumulation and clipping
- 💿 **Checkpoint Management**: Automatic saving, loading, and resuming
- 📈 **Evaluation**: Built-in evaluation with perplexity and accuracy metrics
- 🔧 **Flexible Configuration**: JSON-based configuration system

## Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers datasets accelerate
pip install flash-attn  # For attention optimization
pip install triton      # For custom kernels
```

### 2. Prepare Your Data

Nano-vLLM supports multiple data formats and can train on individual files or entire document collections:

#### Single File (JSONL format):
```jsonl
{"text": "Your training text here..."}
{"text": "Another training example..."}
{"text": "More training data..."}
```

#### Multiple Documents in a Folder:
```
documents/
├── doc1.txt
├── doc2.txt
├── tutorials.jsonl
├── articles.json
└── subfolder/
    ├── more_docs.txt
    └── data.jsonl
```

Supported formats:
- **`.txt`**: Plain text files (each file = one document)
- **`.jsonl`**: JSON Lines with `text`, `content`, or `body` fields
- **`.json`**: JSON files with single documents or arrays of documents

### 3. Create Training Configuration

Create a configuration file `configs/train_config.json`:

```json
{
  "model_name_or_path": "Qwen/Qwen3-0.6B",
  "dataset_path": "data/train.jsonl",
  "output_dir": "./checkpoints",
  "learning_rate": 5e-5,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 2048,
  "num_train_epochs": 3,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "save_steps": 1000,
  "eval_steps": 500,
  "logging_steps": 10
}
```

### 4. Start Training

```bash
# Single GPU training
python train.py --config configs/train_config.json

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --config configs/train_config.json

# Training on multiple documents in a folder
python train.py --config configs/train_config.json --use_multi_document --analyze_documents

# Training with document concatenation
python train.py --config configs/train_config.json --use_multi_document --concatenate_documents
```

## Configuration Options

### Model and Data
- `model_name_or_path`: Path to pretrained model or HuggingFace model ID
- `dataset_path`: Path to training dataset (JSONL format)
- `output_dir`: Directory to save checkpoints and logs
- `max_seq_length`: Maximum sequence length for training

### Training Hyperparameters
- `learning_rate`: Learning rate (default: 5e-5)
- `weight_decay`: Weight decay coefficient (default: 0.01)
- `beta1`, `beta2`: Adam optimizer betas (default: 0.9, 0.95)
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `num_train_epochs`: Number of training epochs
- `max_train_steps`: Maximum training steps (overrides epochs)

### Optimization
- `mixed_precision`: "no", "fp16", or "bf16"
- `gradient_clipping`: Gradient clipping threshold (default: 1.0)
- `gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `lr_scheduler_type`: "linear", "cosine", or "constant"
- `warmup_steps` / `warmup_ratio`: Learning rate warmup

### Distributed Training
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `data_parallel_size`: Number of GPUs for data parallelism

### Evaluation and Logging
- `eval_strategy`: "no", "steps", or "epoch"
- `eval_steps`: Evaluate every N steps
- `save_strategy`: "no", "steps", or "epoch"
- `save_steps`: Save checkpoint every N steps
- `logging_steps`: Log every N steps

## Multi-Document Training

### Training on Document Collections

When you have multiple documents in a folder, use the `MultiDocumentDataset`:

```python
from nanovllm.training import MultiDocumentDataset, analyze_document_collection

# Analyze your document collection first
analysis = analyze_document_collection("./documents")
print(f"Found {analysis['total_documents']} documents in {analysis['total_files']} files")

# Create dataset
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    max_length=2048,
    concatenate_documents=False,  # Process documents individually
    chunk_overlap=128,           # Overlap between chunks
    min_chunk_size=100,         # Minimum chunk size
)
```

### Document Processing Strategies

#### Individual Document Processing
Process each document separately, maintaining document boundaries:

```python
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    concatenate_documents=False,  # Keep documents separate
    chunk_overlap=50,
    min_chunk_size=100,
)
```

#### Concatenated Document Processing
Combine all documents into a continuous stream:

```python
dataset = MultiDocumentDataset(
    data_path="./documents",
    tokenizer=tokenizer,
    concatenate_documents=True,   # Combine all documents
    document_separator="\n\n---\n\n",  # Separator between docs
    chunk_overlap=128,
)
```

### Command Line Usage

```bash
# Analyze documents before training
python train.py --config config.json --use_multi_document --analyze_documents

# Train with individual document processing
python train.py --config config.json --use_multi_document

# Train with document concatenation
python train.py --config config.json --use_multi_document --concatenate_documents

# Custom chunk overlap
python train.py --config config.json --use_multi_document --chunk_overlap 256
```

## Advanced Usage

### Memory Optimization

For training large models with limited GPU memory:

```python
from nanovllm.training.memory import optimize_memory_usage, estimate_training_memory

# Estimate memory requirements
memory_estimate = estimate_training_memory(
    model=model,
    batch_size=batch_size,
    sequence_length=seq_length,
    mixed_precision=True,
    gradient_checkpointing=True,
)
print(f"Estimated memory: {memory_estimate['total_gb']:.2f}GB")

# Apply optimizations
optimize_memory_usage(
    model=model,
    enable_gradient_checkpointing=True,
    enable_cpu_offload=False,
)
```

### Custom Training Loop

```python
from nanovllm import Trainer, TrainingConfig
from nanovllm.training import create_optimizer, create_scheduler

# Create configuration
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",
    dataset_path="data/train.jsonl",
    output_dir="./checkpoints",
    # ... other parameters
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
results = trainer.train()
```

### Evaluation

```python
from nanovllm.training.evaluation import evaluate_model

# Evaluate model
metrics = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_dataloader=eval_dataloader,
    device=device,
    use_amp=True,
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Loss: {metrics.loss:.4f}")
```

### Checkpoint Management

```python
from nanovllm.training.checkpoint import CheckpointManager

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    output_dir="./checkpoints",
    save_total_limit=3,
)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    step=1000,
    epoch=1,
    loss=2.5,
    config=config,
)

# Load checkpoint
training_state = checkpoint_manager.load_checkpoint(
    checkpoint_path="./checkpoints/checkpoint-1000",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

## Distributed Training

### Tensor Parallelism

Tensor parallelism splits model layers across multiple GPUs:

```json
{
  "tensor_parallel_size": 4,
  "per_device_train_batch_size": 1
}
```

### Data Parallelism

Data parallelism replicates the model across GPUs:

```bash
torchrun --nproc_per_node=4 train.py --config configs/train_config.json
```

### Mixed Parallelism

Combine tensor and data parallelism:

```json
{
  "tensor_parallel_size": 2,
  "data_parallel_size": 2
}
```

## Performance Tips

1. **Use Mixed Precision**: Enable `bf16` for better performance on modern GPUs
2. **Gradient Checkpointing**: Reduces memory usage at the cost of computation
3. **Optimal Batch Size**: Use the largest batch size that fits in memory
4. **Gradient Accumulation**: Simulate larger batch sizes with limited memory
5. **Learning Rate Scheduling**: Use cosine scheduling with warmup
6. **Data Loading**: Use multiple workers for data loading

## Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Use `mixed_precision: "bf16"`
- Consider CPU offloading

### Slow Training
- Increase batch size if memory allows
- Use multiple data loading workers
- Enable mixed precision training
- Use tensor parallelism for large models

### Convergence Issues
- Adjust learning rate
- Use learning rate warmup
- Check gradient clipping threshold
- Verify data quality and preprocessing

## Examples

See the `example_training.py` script for comprehensive examples of:
- Basic training setup
- Memory optimization
- Custom training loops
- Evaluation workflows

## Compatibility

The training features are fully compatible with the existing nano-vllm inference engine. You can:
- Train models and use them for inference
- Resume training from inference checkpoints
- Switch between training and inference modes seamlessly

## Contributing

To contribute to the training features:
1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure compatibility with distributed training
