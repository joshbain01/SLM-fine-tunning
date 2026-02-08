# SLM Fine-Tuning for Military Manual (ATP 6-0.5)

This repository provides a complete pipeline for fine-tuning Small Language Models (SLMs) like Llama 3.2 3B on military manuals, specifically ATP 6-0.5, using Unsloth and QLoRA for efficient training on edge hardware.

## 🎯 Project Goal

Fine-tune SLMs to handle Command Post (CP) delegation tasks on edge hardware with optimized memory usage and inference speed through:
- **Unsloth**: Fast and memory-efficient training (2-5x faster than standard methods)
- **QLoRA**: 4-bit quantization for reduced memory footprint
- **Edge Deployment**: Optimized for resource-constrained environments

## 📋 Features

- ✅ Data preprocessing for military manuals (ATP 6-0.5 format)
- ✅ Fine-tuning pipeline with Unsloth + QLoRA
- ✅ Configurable training parameters
- ✅ Inference and evaluation scripts
- ✅ Edge deployment optimization
- ✅ Example notebooks and documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/joshbain01/SLM-fine-tunning.git
cd SLM-fine-tunning

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Prepare your data** (place ATP 6-0.5 manual in `data/raw/`):
```bash
python src/prepare_data.py --input data/raw/atp_6_0_5.txt --output data/processed/
```

2. **Fine-tune the model**:
```bash
python src/train.py --config configs/llama_3_2_3b.yaml
```

3. **Run inference**:
```bash
python src/inference.py --model models/finetuned_model --prompt "Explain CP delegation procedures"
```

## 📁 Project Structure

```
SLM-fine-tunning/
├── src/
│   ├── prepare_data.py      # Data preprocessing
│   ├── train.py              # Fine-tuning script
│   ├── inference.py          # Inference script
│   └── utils.py              # Utility functions
├── configs/
│   └── llama_3_2_3b.yaml    # Model configuration
├── data/
│   ├── raw/                  # Raw training data
│   └── processed/            # Processed datasets
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks
│   └── finetune_example.ipynb
├── tests/                    # Unit tests
└── requirements.txt          # Python dependencies
```

## 🔧 Configuration

Edit `configs/llama_3_2_3b.yaml` to customize:
- Model selection (Llama 3.2 1B/3B, Phi-3, etc.)
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (batch size, learning rate, epochs)
- Quantization settings (4-bit, 8-bit)

## 📊 Model Performance

| Model | Size | Training Time | Memory Usage | Edge Compatible |
|-------|------|---------------|--------------|-----------------|
| Llama 3.2 3B (QLoRA) | 3B | ~2-3 hrs | ~8GB VRAM | ✅ |
| Llama 3.2 1B (QLoRA) | 1B | ~1-2 hrs | ~4GB VRAM | ✅ |

## 🎓 Documentation

- [Training Guide](docs/training.md) - Detailed training instructions
- [Edge Deployment](docs/deployment.md) - Deploy on edge devices
- [Data Preparation](docs/data_prep.md) - Format training data
- [Troubleshooting](docs/troubleshooting.md) - Common issues

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning
- [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient quantization
- ATP 6-0.5 Mission Command reference material

## 📞 Contact

For questions or issues, please open a GitHub issue.
