# LLM Inference Optimization: Llama 3.1 8B with TensorRT-LLM

**End-to-end optimization achieving 3.86x speedup** through TensorRT-LLM INT8 quantization, kernel fusion, and production deployment with Triton Inference Server.

---

## TL;DR

- **3.86x faster inference** (15.4s → 4.0s for 512 tokens)
- **2.88x faster TTFT** (27.6ms → 9.6ms)
- **GPU utilization improved** from 58% → 94%
- **Key optimization:** Reduced kernel launches 364K → 52K (7x)
- **Production-ready:** Deployed with Triton Server, 897 tok/s at 8 concurrent requests

**Tech:** TensorRT-LLM 0.15.0 | INT8 quantization | Paged FMHA | In-flight batching | NVIDIA A100

---

## Performance Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Latency** (512 tokens) | 15.4s | 4.0s | **3.86x faster** |
| **TTFT (p50)** | 27.6ms | 9.6ms | **2.88x faster** |
| **Throughput** | 33 tok/s | 128 tok/s | **3.86x faster** |
| **GPU Util** | 58% | 94% | **+36%** |

### Triton Server Scalability

| Concurrent Requests | Throughput | Scaling |
|---------------------|------------|---------|
| 1 | 131 tok/s | 100% |
| 2 | 270 tok/s | 103% ✨ |
| 4 | 481 tok/s | 92% |
| 8 | 897 tok/s | 85% |

---

## Visual Results

The project includes comprehensive performance visualizations in `results/`:

### 📊 Key Visualizations

1. **`executive_summary.png`** - Overall 3.86x speedup dashboard
2. **`performance_comparison.png`** - 6-panel detailed comparison (HuggingFace vs TRT-LLM)
3. **`triton_scaling.png`** - Concurrent request scalability (1-8 requests)
4. **`ttft_percentiles.png`** - TTFT distribution across percentiles (p50-p99)

See `results/PERFORMANCE_REPORT.md` for detailed analysis.

---

## Quick Start

### Installation

```bash
git clone https://github.com/ineshtickoo/llama31-trtllm-optimization.git
cd llama31-trtllm-optimization
pip install tensorrt-llm==0.15.0 transformers torch numpy
```

### Build Optimized Engine

```bash
# Convert to INT8 (see scripts/convert_checkpoint.sh)
python3 scripts/convert_checkpoint.py \
    --model_dir /path/to/llama-3.1-8b \
    --output_dir checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8

# Build TRT-LLM engine (see scripts/build_engine.sh)
trtllm-build \
    --checkpoint_dir checkpoint \
    --output_dir engine \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --use_paged_context_fmha enable \
    --use_fused_mlp enable \
    --max_batch_size 16
```

### Run Benchmarks

```bash
# Baseline benchmark
python benchmarks/baseline.py

# Optimized benchmark
python benchmarks/benchmark_optimized.py

# Triton server benchmark
python benchmarks/benchmark_triton.py

# Profile with Nsight Systems
python benchmarks/baseline_nvtx.py  # or trt_nvtx.py
```

### Run Inference

```python
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
runner = ModelRunner.from_dir("engine")

input_ids = tokenizer("Your prompt", return_tensors="pt")["input_ids"].cuda()
outputs = runner.generate(input_ids, max_new_tokens=128)
```

### Deploy with Triton

```bash
# Triton config is in triton_model_repo/tensorrt_llm/config.pbtxt
docker run --gpus=all --rm -p 8000:8000 \
    -v $(pwd)/triton_model_repo:/models \
    nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3 \
    tritonserver --model-repository=/models

# Test with client
python benchmarks/triton_client.py
```

---

## Optimization Techniques

### 1. INT8 Weight-Only Quantization
50% memory reduction, full-precision activations, minimal accuracy loss

### 2. Paged Context FMHA
Fused attention kernels, eliminates KV cache fragmentation

### 3. Kernel Fusion & CUDA Graphs
**7x reduction in kernel launches** (364K → 52K), eliminates CPU bottleneck

### 4. In-Flight Batching
Continuous batching, 85-103% scaling efficiency across concurrent requests

---

## Why This Matters for Production

### Bottleneck Analysis (NVIDIA Nsight Systems)

**Before Optimization:**
- 364K kernel launches = excessive CPU overhead
- 58% GPU utilization = GPU starved waiting for CPU
- 51W power (12.8% of capacity) = massive underutilization

**After Optimization:**
- 52K kernel launches = 7x reduction
- 94% GPU utilization = near-optimal
- ~300W power = proper GPU saturation

### Cost Impact

For production inference serving (e.g., Baseten, BentoML):
- **3.86x throughput** = 3.86x more requests per GPU
- **Same SLA with 74% fewer GPUs** = massive cost savings
- **Or handle 3.86x more traffic** with same infrastructure

---

## Benchmarking Details

**Setup:**
- Hardware: NVIDIA A100-SXM4-80GB
- Model: Meta-Llama-3.1-8B-Instruct
- Test: 512 token generation, 10 runs + 3 warmup

**Baseline:** HuggingFace Transformers + SDPA  
**Optimized:** TensorRT-LLM + INT8 + Paged FMHA + In-flight batching  
**Profiling:** NVIDIA Nsight Systems

---

## Project Structure

```
llama31-trtllm-optimization/
├── benchmarks/
│   ├── baseline.py                   # HuggingFace baseline benchmark
│   ├── baseline_nvtx.py              # Baseline with Nsight profiling
│   ├── benchmark_optimized.py        # TRT-LLM optimized benchmark
│   ├── benchmark_triton.py           # Triton server load tests
│   ├── triton_client.py              # Triton inference client
│   └── trt_nvtx.py                   # TRT-LLM Nsight profiling
├── llm_demo/                         # Demo applications
├── results/
│   ├── PERFORMANCE_REPORT.md         # Detailed analysis
│   ├── executive_summary.png         # 3.86x speedup visualization
│   ├── performance_comparison.png    # 6-panel comparison
│   ├── triton_scaling.png            # Concurrent request scaling
│   └── ttft_percentiles.png          # Latency distribution
├── scripts/
│   ├── convert_checkpoint.py         # Model conversion script
│   ├── convert_checkpoint.sh         # Conversion wrapper
│   └── build_engine.sh               # TRT-LLM engine builder
├── triton_model_repo/
│   └── tensorrt_llm/
│       └── config.pbtxt              # Triton server config
├── .gitignore
└── README.md
```

---

## Code Examples

### Baseline (HuggingFace)

```python
# See benchmarks/baseline.py for full implementation
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa"
)
outputs = model.generate(input_ids, max_new_tokens=512)
```

### Optimized (TensorRT-LLM)

```python
# See benchmarks/benchmark_optimized.py for full implementation
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("engine")
outputs = runner.generate(input_ids, max_new_tokens=512)
```

### Triton Client

```python
# See benchmarks/triton_client.py for full implementation
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [
    httpclient.InferInput("input_ids", [1, seq_len], "INT32"),
    httpclient.InferInput("input_lengths", [1, 1], "INT32"),
    httpclient.InferInput("request_output_len", [1, 1], "INT32"),
]
inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(np.array([[seq_len]], dtype=np.int32))
inputs[2].set_data_from_numpy(np.array([[128]], dtype=np.int32))

result = client.infer("tensorrt_llm", inputs)
```

### Profiling with Nsight

```bash
# Profile baseline
nsys profile --trace=cuda,nvtx --output=baseline_profile \
    python benchmarks/baseline_nvtx.py

# Profile optimized
nsys profile --trace=cuda,nvtx --output=optimized_profile \
    python benchmarks/trt_nvtx.py
```

---

## Tech Stack

- **TensorRT-LLM** 0.15.0 - Inference optimization
- **Triton Server** 24.11 - Production serving
- **PyTorch** 2.5.0 - Model loading
- **CUDA** 12.6 - GPU compute
- **Transformers** 4.46.0 - Tokenization

---

## Key Takeaways

1. **Kernel launch overhead is critical** - Reduced 364K → 52K for 7x speedup
2. **INT8 quantization is production-viable** - 50% memory savings, minimal accuracy loss
3. **In-flight batching scales well** - 85-103% efficiency up to 8 concurrent requests
4. **GPU utilization matters** - 58% → 94% drove most of the 3.86x speedup
5. **Profiling is essential** - Nsight Systems revealed the CPU bottleneck


---

**Looking for:** LLM Inference Optimization Engineer roles

