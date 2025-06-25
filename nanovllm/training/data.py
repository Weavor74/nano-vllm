"""Data loading utilities for causal language modeling training."""

import os
import json
import random
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer
import torch.distributed as dist


class CausalLMDataset(Dataset):
    """Dataset for causal language modeling training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file (JSON lines format)
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
            split: Dataset split ("train" or "eval")
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data
        self.examples = self._load_data(data_path)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file or directory."""
        examples = []
        data_path = Path(data_path)

        if data_path.is_file():
            # Single file
            examples = self._load_single_file(data_path)
        elif data_path.is_dir():
            # Directory with multiple files
            examples = self._load_directory(data_path)
        else:
            raise ValueError(f"Data path does not exist: {data_path}")

        return examples

    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a single file."""
        examples = []

        if file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        examples.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}")
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
        elif file_path.suffix == '.txt':
            # Plain text file - treat each line as a separate document
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        examples.append({"text": line})
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return examples

    def _load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Load data from all supported files in a directory."""
        examples = []
        supported_extensions = {'.jsonl', '.json', '.txt'}

        # Get all supported files
        files = []
        for ext in supported_extensions:
            files.extend(dir_path.glob(f"*{ext}"))
            files.extend(dir_path.glob(f"**/*{ext}"))  # Recursive search

        # Sort files for consistent ordering
        files = sorted(set(files))

        if not files:
            raise ValueError(f"No supported files found in directory: {dir_path}")

        print(f"Loading data from {len(files)} files in {dir_path}")

        for file_path in files:
            try:
                file_examples = self._load_single_file(file_path)
                # Add source file information
                for example in file_examples:
                    example['source_file'] = str(file_path.relative_to(dir_path))
                examples.extend(file_examples)
                print(f"Loaded {len(file_examples)} examples from {file_path.name}")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Extract text - support different formats
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "input" in example and "output" in example:
            text = example["input"] + example["output"]
        else:
            raise ValueError(f"Cannot find text field in example: {example.keys()}")
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CausalLMDataCollator:
    """Data collator for causal language modeling."""
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch."""
        batch_size = len(examples)
        
        # Get maximum length in batch
        max_len = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_len = min(max_len, self.max_length)
        
        # Initialize batch tensors
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        # Fill batch tensors
        for i, example in enumerate(examples):
            seq_len = min(len(example["input_ids"]), max_len)
            input_ids[i, :seq_len] = example["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = example["attention_mask"][:seq_len]
            labels[i, :seq_len] = example["labels"][:seq_len]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Any] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional distributed sampling."""
    
    sampler = None
    if distributed and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False  # Sampler handles shuffling
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def prepare_datasets(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_split_ratio: float = 0.9,
) -> tuple[CausalLMDataset, Optional[CausalLMDataset]]:
    """
    Prepare train and eval datasets.
    
    Args:
        data_path: Path to dataset file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        train_split_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load full dataset
    full_dataset = CausalLMDataset(data_path, tokenizer, max_length)
    
    if train_split_ratio >= 1.0:
        return full_dataset, None
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split_ratio)
    
    # Create indices for splitting
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    
    # Create subset datasets
    train_examples = [full_dataset.examples[i] for i in train_indices]
    eval_examples = [full_dataset.examples[i] for i in eval_indices]
    
    train_dataset = CausalLMDataset.__new__(CausalLMDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = max_length
    train_dataset.split = "train"
    train_dataset.examples = train_examples
    
    eval_dataset = CausalLMDataset.__new__(CausalLMDataset)
    eval_dataset.tokenizer = tokenizer
    eval_dataset.max_length = max_length
    eval_dataset.split = "eval"
    eval_dataset.examples = eval_examples
    
    return train_dataset, eval_dataset


class MultiDocumentDataset(Dataset):
    """Dataset for training on multiple documents with document-aware processing."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
        concatenate_documents: bool = True,
        document_separator: str = "\n\n",
        chunk_overlap: int = 0,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the multi-document dataset.

        Args:
            data_path: Path to dataset file or directory
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
            split: Dataset split ("train" or "eval")
            concatenate_documents: Whether to concatenate multiple documents
            document_separator: Separator between documents when concatenating
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be included
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.concatenate_documents = concatenate_documents
        self.document_separator = document_separator
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and process documents
        self.documents = self._load_documents(data_path)
        self.chunks = self._create_chunks()

        print(f"Loaded {len(self.documents)} documents, created {len(self.chunks)} training chunks")

    def _load_documents(self, data_path: str) -> List[Dict[str, Any]]:
        """Load documents from file or directory."""
        data_path = Path(data_path)
        documents = []

        if data_path.is_file():
            documents = self._load_single_file(data_path)
        elif data_path.is_dir():
            documents = self._load_directory(data_path)
        else:
            raise ValueError(f"Data path does not exist: {data_path}")

        return documents

    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load documents from a single file."""
        documents = []

        if file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        doc = json.loads(line.strip())
                        doc['source_file'] = str(file_path)
                        doc['line_number'] = line_num
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}")
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, doc in enumerate(data):
                        doc['source_file'] = str(file_path)
                        doc['document_index'] = i
                        documents.append(doc)
                else:
                    data['source_file'] = str(file_path)
                    documents = [data]
        elif file_path.suffix == '.txt':
            # Treat entire file as one document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "text": content,
                        "source_file": str(file_path),
                        "title": file_path.stem
                    })
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return documents

    def _load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Load documents from all supported files in a directory."""
        documents = []
        supported_extensions = {'.jsonl', '.json', '.txt'}

        # Get all supported files
        files = []
        for ext in supported_extensions:
            files.extend(dir_path.glob(f"*{ext}"))
            files.extend(dir_path.glob(f"**/*{ext}"))  # Recursive search

        files = sorted(set(files))

        if not files:
            raise ValueError(f"No supported files found in directory: {dir_path}")

        print(f"Loading documents from {len(files)} files in {dir_path}")

        for file_path in files:
            try:
                file_documents = self._load_single_file(file_path)
                documents.extend(file_documents)
                print(f"Loaded {len(file_documents)} documents from {file_path.name}")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return documents

    def _create_chunks(self) -> List[Dict[str, Any]]:
        """Create training chunks from documents."""
        chunks = []

        if self.concatenate_documents:
            # Concatenate all documents and create chunks
            all_text = self.document_separator.join(
                self._extract_text(doc) for doc in self.documents
            )
            chunks = self._chunk_text(all_text, metadata={"type": "concatenated"})
        else:
            # Process each document separately
            for doc_idx, document in enumerate(self.documents):
                text = self._extract_text(document)
                doc_chunks = self._chunk_text(
                    text,
                    metadata={
                        "document_index": doc_idx,
                        "source_file": document.get("source_file", "unknown"),
                        "title": document.get("title", f"Document {doc_idx}")
                    }
                )
                chunks.extend(doc_chunks)

        return chunks

    def _extract_text(self, document: Dict[str, Any]) -> str:
        """Extract text from a document."""
        if "text" in document:
            return document["text"]
        elif "content" in document:
            return document["content"]
        elif "body" in document:
            return document["body"]
        elif "input" in document and "output" in document:
            return document["input"] + " " + document["output"]
        else:
            # Try to concatenate all string values
            text_parts = []
            for key, value in document.items():
                if isinstance(value, str) and key not in ["source_file", "title"]:
                    text_parts.append(value)
            return " ".join(text_parts) if text_parts else ""

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks of appropriate size."""
        if not text.strip():
            return []

        # Tokenize the full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_length:
            # Text fits in one chunk
            return [{
                "text": text,
                "token_count": len(tokens),
                **metadata
            }]

        # Split into overlapping chunks
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.max_length, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Skip chunks that are too small
            if len(chunk_tokens) < self.min_chunk_size:
                break

            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "chunk_index": len(chunks),
                "start_token": start_idx,
                "end_token": end_idx,
                **metadata
            })

            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        chunk = self.chunks[idx]
        text = chunk["text"]

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {
                "source_file": chunk.get("source_file", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "token_count": chunk.get("token_count", len(input_ids)),
            }
        }


def prepare_multi_document_datasets(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_split_ratio: float = 0.9,
    concatenate_documents: bool = True,
    document_separator: str = "\n\n",
    chunk_overlap: int = 128,
    min_chunk_size: int = 100,
) -> tuple[MultiDocumentDataset, Optional[MultiDocumentDataset]]:
    """
    Prepare train and eval datasets from multiple documents.

    Args:
        data_path: Path to dataset file or directory
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        train_split_ratio: Ratio of data to use for training
        concatenate_documents: Whether to concatenate documents
        document_separator: Separator between documents
        chunk_overlap: Number of tokens to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be included

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Create full dataset
    full_dataset = MultiDocumentDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        concatenate_documents=concatenate_documents,
        document_separator=document_separator,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    if train_split_ratio >= 1.0:
        return full_dataset, None

    # Split chunks for train/eval
    total_chunks = len(full_dataset.chunks)
    train_size = int(total_chunks * train_split_ratio)

    # Create indices for splitting
    indices = list(range(total_chunks))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    # Create train dataset
    train_chunks = [full_dataset.chunks[i] for i in train_indices]
    train_dataset = MultiDocumentDataset.__new__(MultiDocumentDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = max_length
    train_dataset.split = "train"
    train_dataset.concatenate_documents = concatenate_documents
    train_dataset.document_separator = document_separator
    train_dataset.chunk_overlap = chunk_overlap
    train_dataset.min_chunk_size = min_chunk_size
    train_dataset.documents = full_dataset.documents
    train_dataset.chunks = train_chunks

    # Create eval dataset
    eval_chunks = [full_dataset.chunks[i] for i in eval_indices]
    eval_dataset = MultiDocumentDataset.__new__(MultiDocumentDataset)
    eval_dataset.tokenizer = tokenizer
    eval_dataset.max_length = max_length
    eval_dataset.split = "eval"
    eval_dataset.concatenate_documents = concatenate_documents
    eval_dataset.document_separator = document_separator
    eval_dataset.chunk_overlap = chunk_overlap
    eval_dataset.min_chunk_size = min_chunk_size
    eval_dataset.documents = full_dataset.documents
    eval_dataset.chunks = eval_chunks

    return train_dataset, eval_dataset


def analyze_document_collection(data_path: str) -> Dict[str, Any]:
    """
    Analyze a collection of documents to provide statistics.

    Args:
        data_path: Path to dataset file or directory

    Returns:
        Dictionary with analysis results
    """
    data_path = Path(data_path)
    analysis = {
        "total_files": 0,
        "file_types": {},
        "total_documents": 0,
        "total_characters": 0,
        "average_doc_length": 0,
        "files": []
    }

    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        supported_extensions = {'.jsonl', '.json', '.txt'}
        files = []
        for ext in supported_extensions:
            files.extend(data_path.glob(f"*{ext}"))
            files.extend(data_path.glob(f"**/*{ext}"))
        files = sorted(set(files))
    else:
        raise ValueError(f"Data path does not exist: {data_path}")

    analysis["total_files"] = len(files)

    for file_path in files:
        file_ext = file_path.suffix
        analysis["file_types"][file_ext] = analysis["file_types"].get(file_ext, 0) + 1

        try:
            file_docs = 0
            file_chars = 0

            if file_ext == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            doc = json.loads(line.strip())
                            text = _extract_text_for_analysis(doc)
                            file_docs += 1
                            file_chars += len(text)
                        except json.JSONDecodeError:
                            continue
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for doc in data:
                            text = _extract_text_for_analysis(doc)
                            file_docs += 1
                            file_chars += len(text)
                    else:
                        text = _extract_text_for_analysis(data)
                        file_docs = 1
                        file_chars = len(text)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_docs = 1
                    file_chars = len(content)

            analysis["files"].append({
                "path": str(file_path),
                "documents": file_docs,
                "characters": file_chars,
                "avg_length": file_chars / file_docs if file_docs > 0 else 0
            })

            analysis["total_documents"] += file_docs
            analysis["total_characters"] += file_chars

        except Exception as e:
            print(f"Warning: Failed to analyze {file_path}: {e}")

    if analysis["total_documents"] > 0:
        analysis["average_doc_length"] = analysis["total_characters"] / analysis["total_documents"]

    return analysis


def _extract_text_for_analysis(document: Dict[str, Any]) -> str:
    """Extract text from a document for analysis purposes."""
    if "text" in document:
        return document["text"]
    elif "content" in document:
        return document["content"]
    elif "body" in document:
        return document["body"]
    elif "input" in document and "output" in document:
        return document["input"] + " " + document["output"]
    else:
        # Try to concatenate all string values
        text_parts = []
        for key, value in document.items():
            if isinstance(value, str) and key not in ["source_file", "title", "id"]:
                text_parts.append(value)
        return " ".join(text_parts) if text_parts else ""
