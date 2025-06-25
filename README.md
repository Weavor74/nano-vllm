# Nano-vLLM

A lightweight vLLM implementation built from scratch with comprehensive training capabilities.

## Key Features

### Inference
* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

### Training (NEW!)
* 🔥 **Full Training Support** - Train language models from scratch
* 🌐 **Distributed Training** - Tensor parallelism and data parallelism
* 💾 **Memory Optimization** - Gradient checkpointing, mixed precision (FP16/BF16)
* 📊 **Advanced Optimizers** - AdamW with learning rate scheduling
* 🎯 **Gradient Management** - Accumulation, clipping, and checkpointing
* 💿 **Checkpoint Management** - Automatic saving, loading, and resuming
* 📈 **Evaluation** - Built-in metrics and evaluation during training

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Quick Start

### Inference

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method.
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

### Training

Train language models from scratch with nano-vllm:

```python
from nanovllm import Trainer, TrainingConfig

# Create training configuration
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen3-0.6B",
    dataset_path="data/train.jsonl",
    output_dir="./checkpoints",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    mixed_precision="bf16",
    gradient_checkpointing=True,
)

# Create and run trainer
trainer = Trainer.from_pretrained(config, tokenizer)
trainer.train()
```

Or use the training script:

```bash
# Single GPU training
python train.py --config configs/train_config.json

# Multi-GPU distributed training
torchrun --nproc_per_node=4 train.py --config configs/train_config.json
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Training Documentation

For comprehensive training documentation, see [TRAINING.md](TRAINING.md) which covers:

- 🔧 **Configuration**: Detailed configuration options
- 🚀 **Distributed Training**: Tensor and data parallelism setup
- 💾 **Memory Optimization**: Gradient checkpointing and mixed precision
- 📊 **Advanced Features**: Custom training loops, evaluation, checkpointing
- 🛠️ **Troubleshooting**: Common issues and performance tips
- 📝 **Examples**: Complete training examples and use cases

## Architecture

Nano-vLLM now supports both inference and training with a unified architecture:

```
nanovllm/
├── engine/          # Inference engine (original)
├── layers/          # Neural network layers (shared)
├── models/          # Model implementations (enhanced for training)
├── training/        # Training framework (NEW!)
│   ├── trainer.py   # Main training engine
│   ├── data.py      # Data loading utilities
│   ├── optimizer.py # Optimizers and schedulers
│   ├── evaluation.py # Evaluation utilities
│   ├── checkpoint.py # Checkpoint management
│   └── memory.py    # Memory optimization
├── config.py        # Configuration (enhanced)
└── llm.py          # Main LLM interface
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)