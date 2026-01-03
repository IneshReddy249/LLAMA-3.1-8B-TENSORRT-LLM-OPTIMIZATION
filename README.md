# TensorRT-LLM Inference Optimization

[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-0.15.0-76B900?logo=nvidia)](https://github.com/NVIDIA/TensorRT-LLM)
[![Triton](https://img.shields.io/badge/Triton_Server-24.10-76B900?logo=nvidia)](https://github.com/triton-inference-server/server)
[![H100](https://img.shields.io/badge/GPU-H100_PCIe_80GB-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/h100/)

Production-ready LLM inference with **Llama 3.1 8B** on **NVIDIA H100** achieving **170 tok/s** with real-time streaming chat UI.

---

## 🚀 Performance

| Metric | Value |
|--------|-------|
| **TTFT** (Time to First Token) | 11-13ms |
| **Throughput** (single request) | 160-170 tok/s |
| **Throughput** (16 concurrent) | 1,700+ tok/s |
| **Model Size** | 8.6 GB (FP8) |

---

## ⚡ Optimizations

- **FP8 Quantization** - Native H100 tensor core support, 2-3x memory reduction
- **Paged KV Cache** - Eliminates memory fragmentation
- **FlashAttention** - Fused attention kernels via paged context FMHA
- **In-flight Batching** - Continuous batching without padding
- **TensorRT-LLM** - Optimized CUDA kernels and graph optimization

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Inference Engine | TensorRT-LLM 0.15.0 |
| Model Serving | Triton Inference Server 24.10 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Reflex |
| Container | nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 |
| GPU | NVIDIA H100 PCIe 80GB |

---

## 📋 Prerequisites

- NVIDIA H100 80GB (or A100 80GB)
- Docker with NVIDIA runtime
- HuggingFace account with Llama 3.1 access
- 50GB+ disk space

### Cloud GPU Options
- [Shadeform](https://shadeform.ai) - H100 PCIe ~$2.5/hr
- [Brev.dev](https://brev.dev)
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

---

## 🏃 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/ineshtickoo/llm-inference-optimization.git
cd llm-inference-optimization

export HF_TOKEN="your_hf_token_here"
```

### 2. Start Container

```bash
docker compose up -d
docker exec -it trtllm-dev bash
```

### 3. Download Model

```bash
pip install huggingface_hub
huggingface-cli login --token $HF_TOKEN

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /workspace/models/llama-3.1-8b-instruct \
  --local-dir-use-symlinks False
```

### 4. Save Tokenizer

```bash
mkdir -p /workspace/tokenizer
python3 << 'EOF'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.save_pretrained("/workspace/tokenizer")
print("Tokenizer saved!")
EOF
```

### 5. FP8 Quantization

```bash
# Quantize to FP8
python3 /opt/TensorRT-LLM-examples/quantization/quantize.py \
  --model_dir /workspace/models/llama-3.1-8b-instruct \
  --output_dir /workspace/models/llama-3.1-8b-fp8 \
  --dtype float16 \
  --qformat fp8 \
  --calib_dataset /workspace/calib_data.json \
  --calib_size 512 \
  --kv_cache_dtype fp8
```

### 6. Build TensorRT Engine

```bash
mkdir -p /workspace/engines/llama-8b-fp8

trtllm-build \
  --checkpoint_dir /workspace/models/llama-3.1-8b-fp8 \
  --output_dir /workspace/engines/llama-8b-fp8 \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --max_batch_size 16 \
  --max_input_len 2048 \
  --max_seq_len 4096 \
  --paged_kv_cache enable \
  --use_paged_context_fmha enable \
  --multiple_profiles enable \
  --use_fp8_context_fmha enable
```

### 7. Setup Triton Server

```bash
mkdir -p /workspace/triton_model_repo/tensorrt_llm/1
cp -r /workspace/engines/llama-8b-fp8/* /workspace/triton_model_repo/tensorrt_llm/1/
cp -r /workspace/tokenizer/* /workspace/triton_model_repo/tensorrt_llm/1/
```

### 8. Start All Services

```bash
./start_all.sh
```

### 9. Access

- **Chat UI**: http://localhost:8080
- **API**: http://localhost:8082/v1/chat/completions
- **Triton**: http://localhost:8000/v2/health/ready

---

## 🐳 Docker Configuration

### docker-compose.yaml

```yaml
version: '3.8'

services:
  trtllm:
    image: nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
    container_name: trtllm-dev
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8000:8000"  # Triton HTTP
      - "8001:8001"  # Triton gRPC
      - "8002:8002"  # Triton Metrics
      - "8080:8080"  # Chat UI
      - "8081:8081"  # Reflex WebSocket
      - "8082:8082"  # FastAPI Backend
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: sleep infinity
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 📦 Dependencies

```txt
# Backend
fastapi
uvicorn
tritonclient[grpc]
transformers
numpy

# Frontend
reflex
httpx

# Quantization
datasets
```

Install all:
```bash
pip install fastapi uvicorn tritonclient[grpc] transformers numpy reflex httpx datasets
```

---

## 📁 Project Structure

```
llm-inference-optimization/
├── docker-compose.yaml
├── start_all.sh
├── models/
│   ├── llama-3.1-8b-instruct/
│   └── llama-3.1-8b-fp8/
├── engines/
│   └── llama-8b-fp8/
│       └── rank0.engine
├── tokenizer/
├── triton_model_repo/
│   └── tensorrt_llm/
│       ├── config.pbtxt
│       └── 1/
├── backend/
│   └── src/
│       └── server.py
└── frontend/
    ├── rxconfig.py
    └── trtllm_chat/
        └── trtllm_chat.py
```

---

## 🔌 API Usage

### Streaming Request

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is TensorRT-LLM?"}],
    "max_tokens": 512,
    "stream": true
  }'
```

### Response Format

```json
{
  "choices": [{"delta": {"content": "token"}}],
  "metrics": {
    "ttft_ms": 11.2,
    "tps": 168.5,
    "tokens": 42,
    "latency_s": 0.25
  }
}
```

---

## ⚠️ Troubleshooting

### Port 8081 Not Exposed
```bash
docker stop trtllm-dev && docker rm trtllm-dev
docker compose up -d
```

### Triton Not Starting
```bash
tail -100 /workspace/triton.log
```

### Stop Tokens Not Working
Ensure `STOP_TOKENS = {128001, 128008, 128009}` in server.py

---

## 📊 Triton Config

```protobuf
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 16

model_transaction_policy {
  decoupled: true
}

input [
  { name: "input_ids", data_type: TYPE_INT32, dims: [-1] },
  { name: "input_lengths", data_type: TYPE_INT32, dims: [1] },
  { name: "request_output_len", data_type: TYPE_INT32, dims: [1] },
  { name: "streaming", data_type: TYPE_BOOL, dims: [1] },
  { name: "end_id", data_type: TYPE_INT32, dims: [1], optional: true }
]

output [
  { name: "output_ids", data_type: TYPE_INT32, dims: [-1, -1] },
  { name: "sequence_length", data_type: TYPE_INT32, dims: [-1] }
]

parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}
parameters {
  key: "batching_type"
  value: { string_value: "inflight_fused_batching" }
}
parameters {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.85" }
}
```

---

## 🔗 Resources

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Reflex](https://reflex.dev)
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)

---

## 👤 Author

**Inesh Tickoo**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/ineshtickoo)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/ineshtickoo)

---

**⭐ Star this repo if it helped you!**
