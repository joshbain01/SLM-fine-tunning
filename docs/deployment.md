# Edge Deployment Guide

This guide explains how to deploy your fine-tuned model on edge hardware for CP delegation tasks.

## Overview

Edge deployment requires:
1. **Model quantization** - Reduce size and memory usage
2. **Efficient runtime** - Fast inference engine
3. **Hardware optimization** - CPU/GPU acceleration

## Deployment Options

### Option 1: GGUF with llama.cpp (Recommended)

**Best for:** CPU inference, small devices, embedded systems

#### Step 1: Model Conversion

The training script automatically generates GGUF files:
```bash
models/llama-3.2-3b-atp_gguf/
```

Or convert manually:
```bash
python -m unsloth.convert_to_gguf \
    --model models/llama-3.2-3b-atp_merged \
    --output models/llama-3.2-3b-atp.gguf \
    --quantization q4_k_m
```

#### Step 2: Install llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CPU support
make

# Or with CUDA support (if GPU available)
make LLAMA_CUBLAS=1

# Or with Metal support (macOS)
make LLAMA_METAL=1
```

#### Step 3: Run Inference

```bash
# Interactive mode
./main -m ../SLM-fine-tunning/models/llama-3.2-3b-atp.gguf \
    -p "What is mission command?" \
    -n 256 \
    -t 4 \
    --temp 0.7

# Server mode
./server -m ../SLM-fine-tunning/models/llama-3.2-3b-atp.gguf \
    --host 0.0.0.0 \
    --port 8080
```

**Parameters:**
- `-n`: Max tokens to generate
- `-t`: Number of CPU threads
- `--temp`: Temperature (0.0-2.0)
- `--ctx-size`: Context window size

### Option 2: Ollama (Easy Setup)

**Best for:** Quick deployment, API server, development

#### Step 1: Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

#### Step 2: Create Modelfile

```bash
cat > Modelfile << EOF
FROM ./models/llama-3.2-3b-atp.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50

TEMPLATE """### Instruction:
You are an expert on ATP 6-0.5 (Mission Command). Answer the following question based on the manual.

### Question:
{{ .Prompt }}

### Answer:
"""
EOF
```

#### Step 3: Create and Run

```bash
# Import model
ollama create atp-assistant -f Modelfile

# Run interactively
ollama run atp-assistant

# Or as API server
ollama serve
```

#### Step 4: Use API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "atp-assistant",
  "prompt": "What is mission command?",
  "stream": false
}'
```

### Option 3: Python with Unsloth

**Best for:** Python applications, custom integrations

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/llama-3.2-3b-atp_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Generate
prompt = "What is mission command?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0])
print(response)
```

## Hardware-Specific Optimization

### Raspberry Pi / ARM Devices

**Recommended configuration:**
```bash
# Use smaller model (Llama 3.2 1B)
# Quantize to Q4 or Q5
# Limit context size

./main -m model-1b-q4.gguf \
    -t 4 \
    --ctx-size 512 \
    -n 128
```

**Performance tips:**
- Use swap file for memory
- Limit concurrent requests
- Consider Q3 or Q4 quantization

### Edge Servers (NVIDIA Jetson)

**With GPU acceleration:**
```bash
# Build with CUDA
make LLAMA_CUBLAS=1

# Run with GPU layers
./main -m model.gguf \
    -ngl 32 \
    -t 4
```

**Optimization:**
- Offload as many layers as VRAM allows (`-ngl`)
- Use TensorRT for extra speed
- Monitor thermal throttling

### x86 Servers / Desktop

**Best performance:**
```bash
# Use AVX2/AVX512 instructions
make LLAMA_AVX2=1

# Or with GPU
make LLAMA_CUBLAS=1

# Run optimized
./main -m model.gguf \
    -ngl 99 \
    -t 8 \
    -b 512
```

## Quantization Levels

Choose quantization based on accuracy vs. size trade-off:

| Quantization | Size | Quality | Use Case |
|--------------|------|---------|----------|
| Q2_K | Smallest | Lower | Extreme constraints |
| Q3_K_M | Small | Good | Resource-limited |
| Q4_K_M | Medium | Very Good | **Recommended** |
| Q5_K_M | Large | Excellent | High accuracy needed |
| Q6_K | Larger | Near-original | Minimal degradation |
| Q8_0 | Largest | Original | No compromise |

**Recommendation:** Q4_K_M offers the best balance.

## Production Deployment

### Docker Container

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl

# Install llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /llama
WORKDIR /llama
RUN make LLAMA_CUBLAS=1

# Copy model
COPY models/llama-3.2-3b-atp.gguf /models/

# Run server
CMD ["./server", "-m", "/models/llama-3.2-3b-atp.gguf", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:
```bash
docker build -t atp-assistant .
docker run -p 8080:8080 atp-assistant
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atp-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: atp-assistant
  template:
    metadata:
      labels:
        app: atp-assistant
    spec:
      containers:
      - name: llama
        image: atp-assistant:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: atp-assistant
spec:
  selector:
    app: atp-assistant
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Performance Benchmarks

### Llama 3.2 3B Q4_K_M

**Raspberry Pi 4 (8GB):**
- Tokens/sec: ~2-3
- Context: 512 tokens
- Latency: 5-10s for 50 tokens

**Intel i7-12700K (CPU only):**
- Tokens/sec: ~15-20
- Context: 2048 tokens
- Latency: 2-3s for 50 tokens

**NVIDIA RTX 4090:**
- Tokens/sec: ~100-150
- Context: 2048 tokens
- Latency: <1s for 50 tokens

**NVIDIA Jetson Orin:**
- Tokens/sec: ~30-40
- Context: 2048 tokens
- Latency: 1-2s for 50 tokens

## Optimization Tips

### 1. Batch Processing
Process multiple requests together:
```bash
./server -m model.gguf --batch-size 512
```

### 2. Context Caching
Cache common prefixes:
```bash
./server -m model.gguf --cache-type-k f16
```

### 3. Memory Mapping
Use mmap for faster loading:
```bash
./main -m model.gguf --mmap
```

### 4. Thread Tuning
Match CPU cores:
```bash
# For 8-core CPU
./main -m model.gguf -t 6  # Leave 2 cores for system
```

## Monitoring & Logging

### Resource Usage

```bash
# CPU/Memory
htop

# GPU
nvidia-smi -l 1

# Temperature
watch -n 1 sensors
```

### Performance Metrics

```python
import time

start = time.time()
response = generate_response(prompt)
latency = time.time() - start

tokens = len(tokenizer.encode(response))
tokens_per_second = tokens / latency

print(f"Latency: {latency:.2f}s")
print(f"Throughput: {tokens_per_second:.2f} tokens/s")
```

## Troubleshooting

### Slow Inference
1. Check CPU/GPU usage
2. Reduce context size
3. Use lower quantization
4. Enable hardware acceleration

### High Memory Usage
1. Use lower quantization (Q4 → Q3)
2. Reduce batch size
3. Limit context window
4. Enable memory mapping

### Quality Degradation
1. Use higher quantization (Q4 → Q5/Q6)
2. Retrain with more data
3. Adjust inference parameters
4. Check prompt formatting

## Next Steps

- Evaluation guide
- API integration
- Monitoring setup

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Ollama Documentation](https://ollama.com/docs)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
