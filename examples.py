#!/usr/bin/env python3
"""
Example usage script demonstrating the complete workflow.

This script shows how to use the fine-tuning pipeline programmatically.
"""

import os
import sys


def check_environment():
    """Check if the environment is set up correctly."""
    print("Checking environment...")
    
    # Check if running in repository root
    if not os.path.exists("src/train.py"):
        print("❌ Error: Please run this script from the repository root")
        return False
    
    # Check if data exists
    if not os.path.exists("data/processed/train.jsonl"):
        print("⚠️  Warning: Training data not found")
        print("   Run: python src/prepare_data.py --input data/raw/atp_6_0_5.txt")
        return False
    
    print("✓ Environment OK")
    return True


def example_data_preparation():
    """Example: Prepare training data."""
    print("\n" + "="*60)
    print("Example 1: Data Preparation")
    print("="*60)
    
    print("""
# Prepare data from ATP 6-0.5 manual
python src/prepare_data.py \\
    --input data/raw/atp_6_0_5.txt \\
    --output data/processed \\
    --train-split 0.8 \\
    --eval-split 0.1 \\
    --seed 42

# This will create:
# - data/processed/train.jsonl (80% of data)
# - data/processed/eval.jsonl (10% of data)
# - data/processed/test.jsonl (10% of data)
    """)


def example_training():
    """Example: Train the model."""
    print("\n" + "="*60)
    print("Example 2: Training")
    print("="*60)
    
    print("""
# Train Llama 3.2 3B (recommended)
python src/train.py --config configs/llama_3_2_3b.yaml

# Or train the smaller 1B model
python src/train.py --config configs/llama_3_2_1b.yaml

# Resume training from checkpoint
python src/train.py \\
    --config configs/llama_3_2_3b.yaml \\
    --resume models/llama-3.2-3b-atp/checkpoint-100

# Monitor training with TensorBoard
tensorboard --logdir logs --port 6006
# Then open: http://localhost:6006
    """)


def example_inference():
    """Example: Run inference."""
    print("\n" + "="*60)
    print("Example 3: Inference")
    print("="*60)
    
    print("""
# Interactive mode (chat-like)
python src/inference.py \\
    --model models/llama-3.2-3b-atp_merged \\
    --interactive

# Single query
python src/inference.py \\
    --model models/llama-3.2-3b-atp_merged \\
    --prompt "What is mission command according to ATP 6-0.5?"

# With custom generation parameters
python src/inference.py \\
    --model models/llama-3.2-3b-atp_merged \\
    --prompt "Explain CP delegation procedures" \\
    --max-tokens 512 \\
    --temperature 0.7 \\
    --top-p 0.9
    """)


def example_edge_deployment():
    """Example: Deploy to edge devices."""
    print("\n" + "="*60)
    print("Example 4: Edge Deployment")
    print("="*60)
    
    print("""
# The training script automatically creates GGUF models
# Location: models/llama-3.2-3b-atp_gguf/

# Method 1: Using llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make  # Or 'make LLAMA_CUBLAS=1' for GPU support

# Run inference
./main -m ../SLM-fine-tunning/models/llama-3.2-3b-atp.gguf \\
    -p "What is mission command?" \\
    -n 256 \\
    -t 4

# Method 2: Using Ollama
ollama create atp-assistant -f Modelfile
ollama run atp-assistant

# Method 3: Using Python with Unsloth
# See notebooks/finetune_example.ipynb for code examples
    """)


def example_docker():
    """Example: Using Docker."""
    print("\n" + "="*60)
    print("Example 5: Docker Deployment")
    print("="*60)
    
    print("""
# Build Docker image
docker build -t slm-finetuning .

# Run training in Docker
docker run --gpus all \\
    -v $(pwd)/models:/workspace/models \\
    -v $(pwd)/data:/workspace/data \\
    slm-finetuning \\
    python src/train.py --config configs/llama_3_2_3b.yaml

# Run inference in Docker
docker run --gpus all \\
    -v $(pwd)/models:/workspace/models \\
    -p 8080:8080 \\
    slm-finetuning \\
    python src/inference.py --model models/llama-3.2-3b-atp_merged --interactive
    """)


def example_programmatic_usage():
    """Example: Using the modules programmatically."""
    print("\n" + "="*60)
    print("Example 6: Programmatic Usage")
    print("="*60)
    
    print("""
# Python script example:

from unsloth import FastLanguageModel
import yaml

# Load configuration
with open('configs/llama_3_2_3b.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config['model']['name'],
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Generate
prompt = "What is mission command?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0])
print(response)
    """)


def main():
    """Main function."""
    print("="*60)
    print("SLM Fine-tuning Examples")
    print("="*60)
    
    if not check_environment():
        print("\nPlease set up the environment first:")
        print("  bash setup.sh")
        sys.exit(1)
    
    # Show all examples
    example_data_preparation()
    example_training()
    example_inference()
    example_edge_deployment()
    example_docker()
    example_programmatic_usage()
    
    print("\n" + "="*60)
    print("Additional Resources")
    print("="*60)
    print("""
Documentation:
  - README.md - Project overview
  - QUICKSTART.md - 5-minute setup
  - docs/training.md - Detailed training guide
  - docs/deployment.md - Edge deployment guide
  - notebooks/finetune_example.ipynb - Interactive tutorial

Configuration:
  - configs/llama_3_2_3b.yaml - 3B model config
  - configs/llama_3_2_1b.yaml - 1B model config

Scripts:
  - src/prepare_data.py - Data preprocessing
  - src/train.py - Model training
  - src/inference.py - Inference
  - src/utils.py - Utility functions

Makefile Commands:
  - make help - Show all available commands
  - make install - Install dependencies
  - make setup - Full setup
  - make train - Start training
  - make test - Run tests
    """)
    
    print("\n✓ Examples displayed successfully!")
    print("\nFor more help, visit: https://github.com/joshbain01/SLM-fine-tunning")


if __name__ == "__main__":
    main()
