#!/usr/bin/env python3
"""
Fine-tuning script for SLM using Unsloth and QLoRA.

This script fine-tunes models like Llama 3.2 3B on ATP 6-0.5 data
using Unsloth for fast training and QLoRA for efficient memory usage.
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SLM with Unsloth and QLoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_prompt(example: Dict[str, str], prompt_template: str) -> Dict[str, str]:
    """Format training examples using the prompt template."""
    text = prompt_template.format(
        input=example.get("input", ""),
        output=example.get("output", "")
    )
    return {"text": text}


def load_and_prepare_model(config: Dict[str, Any]):
    """Load and prepare model with Unsloth optimization."""
    model_config = config["model"]
    lora_config = config["lora"]
    
    print(f"Loading model: {model_config['name']}")
    print(f"Max sequence length: {model_config['max_seq_length']}")
    print(f"4-bit quantization: {model_config['load_in_4bit']}")
    
    # Load model with Unsloth FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config["max_seq_length"],
        dtype=None if model_config["dtype"] == "auto" else model_config["dtype"],
        load_in_4bit=model_config["load_in_4bit"],
    )
    
    # Apply LoRA with Unsloth
    print("Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
        random_state=config["training"]["seed"],
        use_rslora=lora_config["use_rslora"],
        loftq_config=None,
    )
    
    return model, tokenizer


def load_dataset_files(config: Dict[str, Any]):
    """Load training and evaluation datasets."""
    data_config = config["data"]
    
    train_file = data_config["train_file"]
    eval_file = data_config.get("eval_file")
    
    print(f"\nLoading datasets:")
    print(f"  Train: {train_file}")
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            f"Please run: python src/prepare_data.py --input data/raw/atp_6_0_5.txt"
        )
    
    # Load datasets
    dataset = load_dataset("json", data_files={
        "train": train_file,
        "test": eval_file if eval_file and os.path.exists(eval_file) else train_file
    })
    
    print(f"  Train examples: {len(dataset['train'])}")
    if eval_file and os.path.exists(eval_file):
        print(f"  Eval examples: {len(dataset['test'])}")
    
    return dataset


def train_model(config: Dict[str, Any], model, tokenizer, dataset, resume_from: str = None):
    """Train the model using SFTTrainer."""
    training_config = config["training"]
    data_config = config["data"]
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        warmup_steps=training_config["warmup_steps"],
        learning_rate=training_config["learning_rate"],
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", not training_config.get("fp16", False)),
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config["save_total_limit"],
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        max_grad_norm=training_config["max_grad_norm"],
        group_by_length=training_config.get("group_by_length", True),
        report_to=training_config.get("report_to", "tensorboard"),
        logging_dir=training_config.get("logging_dir", "./logs"),
        evaluation_strategy=training_config.get("evaluation_strategy", "no"),
        load_best_model_at_end=False,
    )
    
    # Format datasets with prompt template
    prompt_template = data_config["prompt_template"]
    
    def formatting_func(examples):
        """Format examples for training."""
        texts = []
        for i in range(len(examples["input"])):
            text = prompt_template.format(
                input=examples["input"][i],
                output=examples["output"][i]
            )
            texts.append(text)
        return {"text": texts}
    
    # Apply formatting
    train_dataset = dataset["train"].map(
        formatting_func,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    eval_dataset = None
    if "test" in dataset and len(dataset["test"]) > 0:
        eval_dataset = dataset["test"].map(
            formatting_func,
            batched=True,
            remove_columns=dataset["test"].column_names
        )
    
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Output directory: {training_config['output_dir']}")
    print(f"Epochs: {training_config['num_train_epochs']}")
    print(f"Batch size: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"LoRA rank: {config['lora']['r']}")
    print(f"Training examples: {len(train_dataset)}")
    print("="*60 + "\n")
    
    # Initialize trainer with Unsloth's fast trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        args=training_args,
        packing=False,  # Set to True for better efficiency if sequences are short
    )
    
    # Start training
    print("Starting training...")
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    # Save the final model
    print("\nSaving final model...")
    output_dir = training_config["output_dir"]
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save merged model (LoRA weights merged into base model)
    merged_output_dir = f"{output_dir}_merged"
    print(f"Saving merged model to {merged_output_dir}...")
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    
    # Optionally save GGUF for edge deployment
    gguf_output = f"{output_dir}_gguf"
    print(f"Saving GGUF quantized model for edge deployment to {gguf_output}...")
    model.save_pretrained_gguf(gguf_output, tokenizer, quantization_method="q4_k_m")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {output_dir}")
    print(f"Merged model saved to: {merged_output_dir}")
    print(f"GGUF model saved to: {gguf_output}")
    print("\nNext steps:")
    print(f"  1. Test the model: python src/inference.py --model {output_dir}")
    print(f"  2. Deploy to edge: Use the GGUF model in {gguf_output}")
    print("="*60)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load and prepare model
    model, tokenizer = load_and_prepare_model(config)
    
    # Load datasets
    dataset = load_dataset_files(config)
    
    # Train model
    train_model(config, model, tokenizer, dataset, args.resume)


if __name__ == "__main__":
    main()
