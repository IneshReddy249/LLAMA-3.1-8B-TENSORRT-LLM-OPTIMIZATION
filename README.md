# LLM Inference Optimization: Llama 3.1 8B with TensorRT-LLM

End-to-end optimization of Llama 3.1 8B achieving **3.86x speedup** using TensorRT-LLM, INT8 quantization, and production deployment with Triton Inference Server.

**Hardware:** NVIDIA A100-SXM4-80GB | **Framework:** TensorRT-LLM 0.15.0 | **Model:** Meta-Llama-3.1-8B-Instruct

---

## Performance Results

### Baseline vs Optimized

| Metric | HuggingFace | TRT-LLM INT8 | Speedup |
|--------|-------------|--------------|---------|
| **Total Latency** (512 tokens) | 15,449 ms | 4,001 ms | **3.86x** |
| **TTFT (p50)** | 27.6 ms | 9.6 ms | **2.88x** |
| **Throughput (TPS)** | 33 tok/s | 128 tok/s | **3.86x** |
| **GPU Utilization** | 58% | 94% | **+36%** |

### Triton Server Scalability

| Concurrent Requests | Throughput | Efficiency |
|---------------------|------------|------------|
| 1 | 131 tok/s | 100% |
| 2 | 270 tok/s | 103% |
| 4 | 481 tok/s | 92% |
| 8 | 897 tok/s | 85% |

**Key Bottleneck Fixed:** CPU kernel launch overhead reduced from 364K → 52K (7x reduction)

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/ineshtickoo/llama31-trtllm-optimization.git
cd llama31-trtllm-optimization

# Install dependencies
pip install tensorrt-llm==0.15.0 transformers torch numpy
```

### 2. Convert Model to TRT-LLM

```bash
# Convert checkpoint with INT8 quantization
python3 convert_checkpoint.py \
    --model_dir /path/to/llama-3.1-8b-instruct \
    --output_dir checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8

# Build TRT-LLM engine
trtllm-build \
    --checkpoint_dir checkpoint \
    --output_dir engine \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --use_paged_context_fmha enable \
    --use_fused_mlp enable \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --max_batch_size 16
```

### 3. Run Inference

```python
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

# Load engine and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
runner = ModelRunner.from_dir("engine")

# Run inference
prompt = "Explain transformers in one sentence:"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
outputs = runner.generate(
    input_ids,
    max_new_tokens=128,
    end_id=tokenizer.eos_token_id
)
```

### 4. Deploy with Triton Server

```bash
# Launch Triton Inference Server
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3 \
    tritonserver --model-repository=/models
```

---

## Optimization Techniques

### 1. INT8 Weight-Only Quantization
- 50% reduction in weight memory
- Full precision activations for accuracy
- Minimal accuracy loss

### 2. Paged Context FMHA
- Fused multi-head attention kernels
- Eliminates KV cache fragmentation
- Better memory efficiency

### 3. Fused MLP & CUDA Graphs
- Kernel fusion for feed-forward layers
- 7x reduction in kernel launches (364K → 52K)
- Reduced CPU-GPU synchronization overhead

### 4. In-Flight Batching
- Continuous batching without waiting for completion
- 85-103% scaling efficiency
- Better GPU utilization under load

---

## Project Structure

```
llama31-trtllm-optimization/
├── scripts/
│   ├── baseline_benchmark.py         # HuggingFace baseline
│   ├── trtllm_benchmark.py           # TRT-LLM optimized
│   ├── triton_benchmark.py           # Triton server tests
│   ├── convert_checkpoint.py         # Model conversion
│   └── profile_nsight.py             # Nsight profiling
├── model_repository/
│   └── tensorrt_llm/
│       ├── config.pbtxt              # Triton config
│       └── 1/                        # Engine files
├── results/
│   ├── baseline_metrics.json
│   ├── optimized_metrics.json
│   └── visualizations/
└── README.md
```

---

## Benchmarking Methodology

**Test Configuration:**
- Model: Meta-Llama-3.1-8B-Instruct
- Prompt: "Write a Python function to reverse a linked list:"
- Output: 512 tokens
- Runs: 10 iterations (after 3 warmup runs)

**Baseline:** HuggingFace Transformers with SDPA (Scaled Dot Product Attention)

**Optimized:** TensorRT-LLM with INT8 quantization, paged FMHA, fused MLP, in-flight batching

**Profiling:** NVIDIA Nsight Systems for kernel-level analysis and bottleneck identification

---

## Key Findings

### Bottleneck Analysis (Nsight Systems)

**Baseline Issues:**
- 364K kernel launches → excessive CPU overhead
- 58% GPU utilization → GPU starved by CPU
- 51W power draw → only 12.8% of 400W capacity

**After Optimization:**
- 52K kernel launches → 7x reduction
- 94% GPU utilization → near-optimal usage
- ~300W power draw → proper GPU saturation

### Why 3.86x Speedup?

1. **INT8 Quantization** → 50% memory bandwidth reduction
2. **Fused Kernels** → Eliminated 312K unnecessary launches
3. **Paged FMHA** → Optimized attention computation
4. **In-Flight Batching** → Continuous GPU utilization

---

## Tech Stack

- **TensorRT-LLM** 0.15.0 - Inference engine
- **Triton Server** 24.11 - Model serving
- **PyTorch** 2.5.0 - Model loading
- **CUDA** 12.6 - GPU acceleration
- **Transformers** 4.46.0 - Tokenization

---

## Usage Examples

### Baseline Benchmark

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
input_ids = tokenizer("Your prompt", return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_new_tokens=512)
```

### Triton Client

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare inputs
input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int32)
inputs = [
    httpclient.InferInput("input_ids", [1, len(input_ids[0])], "INT32"),
    httpclient.InferInput("input_lengths", [1, 1], "INT32"),
    httpclient.InferInput("request_output_len", [1, 1], "INT32"),
]

inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(np.array([[len(input_ids[0])]], dtype=np.int32))
inputs[2].set_data_from_numpy(np.array([[128]], dtype=np.int32))

# Inference
result = client.infer("tensorrt_llm", inputs)
```

---

## Future Improvements

- [ ] Multi-GPU tensor parallelism
- [ ] FP8 quantization support
- [ ] Speculative decoding
- [ ] Automated hyperparameter tuning
- [ ] Production monitoring with Prometheus/Grafana

---

## Citation

```bibtex
@misc{llama31_trtllm_optimization,
  author = {Inesh Tickoo},
  title = {LLM Inference Optimization: Llama 3.1 8B with TensorRT-LLM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ineshtickoo/llama31-trtllm-optimization}
}
```

---

## Contact

**Inesh Tickoo**  
MS Computer Science, Florida Atlantic University (Dec 2025)  
Email: itickoo2023@fau.edu  
LinkedIn: [linkedin.com/in/inesh-tickoo](https://linkedin.com/in/inesh-tickoo)  
GitHub: [@ineshtickoo](https://github.com/ineshtickoo)

---

**License:** MIT | **Status:** Completed December 2024
