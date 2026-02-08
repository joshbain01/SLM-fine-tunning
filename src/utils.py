#!/usr/bin/env python3
"""
Utility functions for SLM fine-tuning project.

This module contains helper functions for data processing,
model management, and evaluation.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch


def count_parameters(model) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_parameter_info(model):
    """Print model parameter information."""
    params = count_parameters(model)
    
    print("\n" + "="*60)
    print("Model Parameters:")
    print("="*60)
    print(f"Total parameters:      {params['total']:,}")
    print(f"Trainable parameters:  {params['trainable']:,}")
    print(f"Trainable percentage:  {params['trainable_percent']:.2f}%")
    print("="*60 + "\n")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def check_gpu_memory():
    """Check and print GPU memory usage."""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU Memory Status:")
        print("="*60)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = props.total_memory / 1024**3
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory:     {memory_total:.2f} GB")
            print(f"  Allocated memory: {memory_allocated:.2f} GB")
            print(f"  Reserved memory:  {memory_reserved:.2f} GB")
            print(f"  Free memory:      {memory_total - memory_reserved:.2f} GB")
        print("="*60 + "\n")
    else:
        print("\nNo GPU available. Using CPU.\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_data_sample(data: List[Dict], n: int = 5) -> List[Dict]:
    """
    Create a small sample from dataset for testing.
    
    Args:
        data: Full dataset
        n: Number of samples
        
    Returns:
        Sample dataset
    """
    import random
    if len(data) <= n:
        return data
    return random.sample(data, n)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["model", "lora", "training", "data"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate model config
    if "name" not in config["model"]:
        raise ValueError("Model name not specified in configuration")
    
    # Validate training config
    if "output_dir" not in config["training"]:
        raise ValueError("Output directory not specified in training configuration")
    
    # Validate data config
    if "train_file" not in config["data"]:
        raise ValueError("Training file not specified in data configuration")
    
    return True


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    seconds_per_step: float = 1.0
) -> str:
    """
    Estimate training time.
    
    Args:
        num_examples: Number of training examples
        batch_size: Batch size
        num_epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        seconds_per_step: Estimated seconds per training step
        
    Returns:
        Formatted time estimate
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = num_examples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_step
    
    return format_time(total_seconds)


def get_model_size(model_path: str) -> float:
    """
    Get the total size of model files in MB.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Size in MB
    """
    total_size = 0
    if os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    elif os.path.isfile(model_path):
        total_size = os.path.getsize(model_path)
    
    return total_size / (1024 * 1024)  # Convert to MB


def prepare_output_directory(output_dir: str, overwrite: bool = False):
    """
    Prepare output directory for training.
    
    Args:
        output_dir: Output directory path
        overwrite: Whether to overwrite existing directory
    """
    if os.path.exists(output_dir) and not overwrite:
        response = input(f"\nOutput directory '{output_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Quick tests
    print("Testing utility functions...")
    
    # Test time formatting
    print(f"Format 3725 seconds: {format_time(3725)}")
    
    # Test training time estimation
    estimate = estimate_training_time(
        num_examples=1000,
        batch_size=2,
        num_epochs=3,
        gradient_accumulation_steps=4,
        seconds_per_step=1.5
    )
    print(f"Estimated training time: {estimate}")
    
    print("\n✓ Utility functions working correctly")
