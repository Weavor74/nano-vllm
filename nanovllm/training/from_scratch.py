"""
From-scratch training utilities for nano-vllm.

This module provides functionality to train language models completely from scratch,
including vocabulary building, model architecture creation, and initialization.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.training.data import MultiDocumentDataset, analyze_document_collection


class CustomTokenizer:
    """Simple tokenizer built from training data."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize custom tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for token inclusion
            special_tokens: Special tokens to include
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Default special tokens
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        
        self.special_tokens = special_tokens
        self.vocab = {}
        self.inverse_vocab = {}
        self.token_frequencies = Counter()
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
    
    def build_vocab_from_documents(self, documents_path: str) -> Dict[str, Any]:
        """
        Build vocabulary from document collection.
        
        Args:
            documents_path: Path to documents
            
        Returns:
            Dictionary with vocabulary statistics
        """
        print(f"Building vocabulary from {documents_path}...")
        
        # Analyze documents first
        analysis = analyze_document_collection(documents_path)
        print(f"Found {analysis['total_documents']} documents with {analysis['total_characters']:,} characters")
        
        # Collect all text
        all_text = self._collect_text_from_documents(documents_path)
        
        # Tokenize at character and subword level
        tokens = self._tokenize_text(all_text)
        
        # Count token frequencies
        self.token_frequencies = Counter(tokens)
        print(f"Found {len(self.token_frequencies)} unique tokens")
        
        # Build vocabulary
        self._build_vocabulary()
        
        vocab_stats = {
            "total_tokens": len(tokens),
            "unique_tokens": len(self.token_frequencies),
            "vocab_size": len(self.vocab),
            "coverage": self._calculate_coverage(),
            "most_common": self.token_frequencies.most_common(10)
        }
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        print(f"Coverage: {vocab_stats['coverage']:.2%}")
        
        return vocab_stats
    
    def _collect_text_from_documents(self, documents_path: str) -> str:
        """Collect all text from documents."""
        from nanovllm.training.data import MultiDocumentDataset
        
        # Create a dummy tokenizer for text extraction
        class DummyTokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text.split())))
            def decode(self, tokens, **kwargs):
                return " ".join(str(t) for t in tokens)
        
        dummy_tokenizer = DummyTokenizer()
        
        # Load documents
        dataset = MultiDocumentDataset(
            data_path=documents_path,
            tokenizer=dummy_tokenizer,
            max_length=999999,  # Don't truncate
            concatenate_documents=True,
            document_separator=" ",
        )
        
        # Extract all text
        all_text = ""
        for doc in dataset.documents:
            text = self._extract_text_from_doc(doc)
            all_text += text + " "
        
        return all_text
    
    def _extract_text_from_doc(self, document: Dict[str, Any]) -> str:
        """Extract text from a document."""
        if "text" in document:
            return document["text"]
        elif "content" in document:
            return document["content"]
        elif "body" in document:
            return document["body"]
        else:
            # Concatenate all string values
            text_parts = []
            for key, value in document.items():
                if isinstance(value, str) and key not in ["source_file", "title"]:
                    text_parts.append(value)
            return " ".join(text_parts)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization strategy."""
        tokens = []
        
        # Split on whitespace and punctuation
        import re
        
        # Basic tokenization: split on whitespace and common punctuation
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words:
            if len(word) <= 3:
                # Short words as single tokens
                tokens.append(word)
            else:
                # Longer words: add subword tokens
                tokens.append(word)  # Full word
                
                # Add character-level tokens for rare words
                if len(word) > 6:
                    for i in range(len(word) - 2):
                        tokens.append(word[i:i+3])  # 3-character subwords
        
        return tokens
    
    def _build_vocabulary(self):
        """Build the final vocabulary."""
        # Start with special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add most frequent tokens
        frequent_tokens = [
            token for token, freq in self.token_frequencies.most_common()
            if freq >= self.min_frequency and token not in self.special_tokens
        ]
        
        # Limit to vocab_size
        available_slots = self.vocab_size - len(self.special_tokens)
        frequent_tokens = frequent_tokens[:available_slots]
        
        # Add to vocabulary
        for token in frequent_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Create inverse mapping
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
    
    def _calculate_coverage(self) -> float:
        """Calculate vocabulary coverage of the training data."""
        total_tokens = sum(self.token_frequencies.values())
        covered_tokens = sum(
            freq for token, freq in self.token_frequencies.items()
            if token in self.vocab
        )
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize_text(text)
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        return " ".join(tokens)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(save_path / "vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=2)
        
        # Save tokenizer config
        config = {
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        with open(save_path / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save frequencies for analysis
        with open(save_path / "token_frequencies.pkl", "wb") as f:
            pickle.dump(dict(self.token_frequencies), f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> "CustomTokenizer":
        """Load tokenizer from directory."""
        load_path = Path(load_directory)
        
        # Load config
        with open(load_path / "tokenizer_config.json", "r") as f:
            config = json.load(f)
        
        # Load vocabulary
        with open(load_path / "vocab.json", "r") as f:
            vocab = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(vocab_size=config["vocab_size"])
        tokenizer.vocab = vocab
        tokenizer.inverse_vocab = {int(i): token for token, i in vocab.items()}
        tokenizer.special_tokens = config["special_tokens"]
        
        # Set special tokens
        for attr in ["pad_token", "unk_token", "bos_token", "eos_token"]:
            setattr(tokenizer, attr, config[attr])
            setattr(tokenizer, f"{attr}_id", config[f"{attr}_id"])
        
        return tokenizer


def get_model_config_for_size(model_size: str, vocab_size: int) -> Qwen3Config:
    """
    Get model configuration for different sizes.
    
    Args:
        model_size: "tiny", "small", "medium", or "large"
        vocab_size: Vocabulary size
        
    Returns:
        Model configuration
    """
    configs = {
        "tiny": {
            "vocab_size": vocab_size,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 1024,
        },
        "small": {
            "vocab_size": vocab_size,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "max_position_embeddings": 2048,
        },
        "medium": {
            "vocab_size": vocab_size,
            "hidden_size": 1024,
            "intermediate_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 4096,
        },
        "large": {
            "vocab_size": vocab_size,
            "hidden_size": 2048,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "max_position_embeddings": 4096,
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config_dict = configs[model_size]
    
    # Add default values
    config_dict.update({
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
    })
    
    return Qwen3Config(**config_dict)


def create_model_from_scratch(
    documents_path: str,
    model_size: str = "small",
    vocab_size: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Tuple[Qwen3ForCausalLM, CustomTokenizer, Dict[str, Any]]:
    """
    Create a model from scratch based on document collection.
    
    Args:
        documents_path: Path to training documents
        model_size: Model size ("tiny", "small", "medium", "large")
        vocab_size: Target vocabulary size (auto-determined if None)
        save_path: Path to save tokenizer and config
        
    Returns:
        Tuple of (model, tokenizer, creation_info)
    """
    print(f"Creating {model_size} model from scratch...")
    
    # Analyze documents to determine vocab size
    if vocab_size is None:
        analysis = analyze_document_collection(documents_path)
        # Heuristic: vocab size based on document collection size
        char_count = analysis['total_characters']
        if char_count < 100_000:
            vocab_size = 8_000
        elif char_count < 1_000_000:
            vocab_size = 16_000
        elif char_count < 10_000_000:
            vocab_size = 32_000
        else:
            vocab_size = 50_000
        
        print(f"Auto-determined vocab size: {vocab_size}")
    
    # Build custom tokenizer
    tokenizer = CustomTokenizer(vocab_size=vocab_size)
    vocab_stats = tokenizer.build_vocab_from_documents(documents_path)
    
    # Create model configuration
    config = get_model_config_for_size(model_size, len(tokenizer.vocab))
    
    # Create model
    model = Qwen3ForCausalLM(config)
    
    # Initialize weights properly for from-scratch training
    model.apply(_init_weights)
    
    creation_info = {
        "model_size": model_size,
        "vocab_stats": vocab_stats,
        "model_config": config.to_dict(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    print(f"Created model with {creation_info['parameter_count']:,} parameters")
    
    # Save tokenizer and config if requested
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save_pretrained(save_path)
        config.save_pretrained(save_path)
        
        # Save creation info
        with open(save_path / "creation_info.json", "w") as f:
            json.dump(creation_info, f, indent=2, default=str)
        
        print(f"Saved tokenizer and config to {save_path}")
    
    return model, tokenizer, creation_info


def _init_weights(module):
    """Initialize model weights for from-scratch training."""
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for layer norm
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
