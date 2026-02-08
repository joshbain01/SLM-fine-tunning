#!/bin/bash
# Setup script for SLM fine-tuning project

set -e

echo "=================================="
echo "SLM Fine-tuning Setup"
echo "=================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if running in virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "\n   ⚠️  Warning: Not running in a virtual environment"
    echo "   It's recommended to use a virtual environment:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA availability
echo -e "\n2. Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "   ⚠️  No NVIDIA GPU detected. Training will be slower on CPU."
fi

# Install dependencies
echo -e "\n3. Installing dependencies..."
echo "   This may take several minutes..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n4. Creating directories..."
mkdir -p data/raw data/processed models logs

# Generate sample data
echo -e "\n5. Generating sample training data..."
python src/prepare_data.py --input data/raw/nonexistent.txt --output data/processed

echo -e "\n=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo -e "\nNext steps:"
echo "  1. Place ATP 6-0.5 manual in data/raw/atp_6_0_5.txt"
echo "  2. Prepare data: python src/prepare_data.py --input data/raw/atp_6_0_5.txt"
echo "  3. Start training: python src/train.py --config configs/llama_3_2_3b.yaml"
echo -e "\nFor more information, see README.md and docs/training.md"
