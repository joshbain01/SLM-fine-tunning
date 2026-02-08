# Quick Start Guide

Get started with SLM fine-tuning in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 20GB free disk space

## Installation

```bash
# Clone the repository
git clone https://github.com/joshbain01/SLM-fine-tunning.git
cd SLM-fine-tunning

# Run setup script
bash setup.sh
```

The setup script will:
- Install all dependencies
- Create necessary directories
- Generate sample training data

## Training (with sample data)

Test the pipeline with sample data:

```bash
# Train with sample data (quick test)
python src/train.py --config configs/llama_3_2_3b.yaml
```

This will take 1-2 hours depending on your GPU.

## Using Real Data

1. **Get ATP 6-0.5 manual:**
   - Download from: https://armypubs.army.mil/
   - Save as: `data/raw/atp_6_0_5.txt`

2. **Prepare data:**
```bash
python src/prepare_data.py \
    --input data/raw/atp_6_0_5.txt \
    --output data/processed
```

3. **Train:**
```bash
python src/train.py --config configs/llama_3_2_3b.yaml
```

## Running Inference

After training completes:

```bash
# Interactive mode
python src/inference.py \
    --model models/llama-3.2-3b-atp_merged \
    --interactive

# Single query
python src/inference.py \
    --model models/llama-3.2-3b-atp_merged \
    --prompt "What is mission command?"
```

## Edge Deployment

Deploy the quantized model:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run inference
./main -m ../SLM-fine-tunning/models/llama-3.2-3b-atp.gguf \
    -p "What is mission command?" \
    -n 256
```

## Jupyter Notebook

For interactive development:

```bash
jupyter notebook notebooks/finetune_example.ipynb
```

## Common Issues

### Out of Memory

Reduce batch size in `configs/llama_3_2_3b.yaml`:
```yaml
training:
  per_device_train_batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 8   # Increase from 4
```

### Slow Training

- Use a GPU with more VRAM (RTX 4090, A100)
- Reduce `max_seq_length` to 1024
- Use Llama 3.2 1B instead of 3B

### Model Quality Issues

- Add more training data
- Increase training epochs to 5-10
- Increase LoRA rank to 32

## Next Steps

- [Full Training Guide](docs/training.md)
- [Edge Deployment](docs/deployment.md)
- [Configuration Options](configs/llama_3_2_3b.yaml)

## Getting Help

- Open an issue on GitHub
- Check [Troubleshooting Guide](docs/training.md#troubleshooting)
- Review [Unsloth Documentation](https://github.com/unslothai/unsloth)

## Key Files

| File | Purpose |
|------|---------|
| `src/prepare_data.py` | Prepare training data |
| `src/train.py` | Fine-tune the model |
| `src/inference.py` | Run inference |
| `configs/llama_3_2_3b.yaml` | Training configuration |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Automated setup |

## Architecture

```
Input Text (ATP 6-0.5)
    ↓
Data Preparation (Q&A pairs)
    ↓
Fine-tuning (Unsloth + QLoRA)
    ↓
Trained Model
    ↓
Quantization (GGUF)
    ↓
Edge Deployment
```

## Performance Targets

| Hardware | Speed | Memory |
|----------|-------|--------|
| RTX 4090 | ~2 hrs | 8GB |
| RTX 3090 | ~3 hrs | 8GB |
| A100 | ~1 hr | 8GB |
| CPU only | ~20 hrs | 16GB |

Training time for 1000 examples, 3 epochs.
