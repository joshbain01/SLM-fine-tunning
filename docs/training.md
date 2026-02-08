# Training Guide

This guide provides detailed instructions for fine-tuning Small Language Models on ATP 6-0.5 data.

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (e.g., RTX 3070, RTX 4060 Ti)
- RAM: 16GB
- Storage: 50GB free space

**Recommended:**
- GPU: 16GB+ VRAM (e.g., RTX 4090, A100)
- RAM: 32GB+
- Storage: 100GB+ SSD

### Software Requirements

- Python 3.8 or later
- CUDA 11.8 or later
- PyTorch 2.1.0 or later

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshbain01/SLM-fine-tunning.git
cd SLM-fine-tunning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Obtain ATP 6-0.5 Manual

Download the ATP 6-0.5 manual from Army Publishing Directorate:
- Visit: https://armypubs.army.mil/
- Search for "ATP 6-0.5"
- Download in PDF or text format

### 2. Convert to Text (if PDF)

If you have a PDF, convert it to text:
```bash
# Using pdftotext (Linux/Mac)
pdftotext atp_6_0_5.pdf data/raw/atp_6_0_5.txt

# Or using Python
pip install pypdf2
python -c "import PyPDF2; pdf = open('atp_6_0_5.pdf', 'rb'); reader = PyPDF2.PdfReader(pdf); text = ''.join(page.extract_text() for page in reader.pages); open('data/raw/atp_6_0_5.txt', 'w').write(text)"
```

### 3. Prepare Training Data

```bash
python src/prepare_data.py \
    --input data/raw/atp_6_0_5.txt \
    --output data/processed \
    --train-split 0.8 \
    --eval-split 0.1 \
    --seed 42
```

This creates:
- `data/processed/train.jsonl` - Training data (80%)
- `data/processed/eval.jsonl` - Evaluation data (10%)
- `data/processed/test.jsonl` - Test data (10%)

## Model Selection

### Available Models

| Model | Parameters | VRAM (4-bit) | Best For |
|-------|-----------|--------------|----------|
| Llama 3.2 1B | 1B | ~4GB | Edge devices, fast inference |
| Llama 3.2 3B | 3B | ~8GB | Balance of quality and speed |
| Llama 3.1 8B | 8B | ~12GB | Higher quality outputs |
| Phi-3 Mini | 3.8B | ~8GB | Alternative to Llama |

### Choosing a Model

For edge deployment with CP delegation tasks, we recommend:
- **Llama 3.2 3B** - Best balance for most use cases
- **Llama 3.2 1B** - If VRAM is limited (<6GB)

## Configuration

Edit `configs/llama_3_2_3b.yaml`:

```yaml
# Key parameters to adjust:

lora:
  r: 16                    # Higher = more parameters (8, 16, 32, 64)
  lora_alpha: 16          # Usually same as r

training:
  num_train_epochs: 3      # More epochs = better fit (but risk overfitting)
  per_device_train_batch_size: 2  # Increase if you have more VRAM
  learning_rate: 2.0e-4    # Lower = more stable, higher = faster
```

### Memory Optimization

If you run out of memory:

1. **Reduce batch size:**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

2. **Reduce sequence length:**
```yaml
model:
  max_seq_length: 1024  # Down from 2048
```

3. **Reduce LoRA rank:**
```yaml
lora:
  r: 8  # Down from 16
```

## Training

### Basic Training

```bash
python src/train.py --config configs/llama_3_2_3b.yaml
```

### Monitor Training

Training logs are saved to TensorBoard:
```bash
tensorboard --logdir logs
```

Access at: http://localhost:6006

### Resume Training

If training is interrupted:
```bash
python src/train.py \
    --config configs/llama_3_2_3b.yaml \
    --resume models/llama-3.2-3b-atp/checkpoint-100
```

## Training Output

The training script produces three model versions:

1. **LoRA Adapters** (`models/llama-3.2-3b-atp/`)
   - Smallest size (~50-100MB)
   - Requires base model to run
   - Best for version control

2. **Merged Model** (`models/llama-3.2-3b-atp_merged/`)
   - Medium size (~3-7GB)
   - Standalone model
   - Best for inference

3. **GGUF Quantized** (`models/llama-3.2-3b-atp_gguf/`)
   - Smallest size (~1-2GB)
   - Optimized for edge devices
   - Best for deployment

## Training Tips

### 1. Start Small
- Test with 100-200 examples first
- Verify the pipeline works
- Scale up gradually

### 2. Monitor Metrics
- Loss should decrease steadily
- Watch for overfitting (eval loss increases)
- Check GPU memory usage

### 3. Hyperparameter Tuning

**Learning Rate:**
- Too high: Training unstable, loss spikes
- Too low: Training too slow, underfitting
- Good range: 1e-4 to 5e-4

**LoRA Rank:**
- Higher rank = more capacity, slower training
- Lower rank = faster, less capacity
- Sweet spot: 16-32 for most tasks

**Epochs:**
- Start with 3 epochs
- Add more if still improving
- Stop if overfitting

### 4. Data Quality
- More data is not always better
- Focus on high-quality, relevant examples
- Remove duplicates and noise

## Troubleshooting

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size to 1
2. Reduce max_seq_length to 1024
3. Enable gradient checkpointing (already on)
4. Use smaller model (Llama 3.2 1B)

### Slow Training

**Speed up training:**
1. Increase batch size (if memory allows)
2. Use bf16 instead of fp16
3. Enable packing in SFTTrainer
4. Use faster GPU (A100 > RTX 4090 > RTX 3090)

### Loss Not Decreasing

**Possible causes:**
1. Learning rate too low - increase it
2. Data quality issues - review training data
3. Model frozen - check LoRA application
4. Prompt format mismatch - verify formatting

### Model Not Following Instructions

**Fixes:**
1. Add more diverse training examples
2. Increase training epochs
3. Use a larger LoRA rank
4. Improve prompt template

## Advanced Topics

### Multi-GPU Training

```bash
torchrun --nproc_per_node=2 src/train.py --config configs/llama_3_2_3b.yaml
```

### Custom Data Augmentation

Edit `src/prepare_data.py` to add:
- Paraphrasing
- Back-translation
- Synthetic examples

### Evaluation Metrics

Add custom metrics in `src/train.py`:
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Add your metrics here
    return {"custom_metric": score}
```

## Next Steps

After training:
1. [Test inference](../src/inference.py)
2. [Evaluate performance](deployment.md)
3. [Deploy to edge](deployment.md)

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)
