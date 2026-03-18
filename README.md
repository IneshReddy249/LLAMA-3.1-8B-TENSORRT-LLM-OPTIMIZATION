# TensorRT-LLM Inference Optimization

[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-0.15.0-76B900?logo=nvidia)](https://github.com/NVIDIA/TensorRT-LLM)
[![Triton](https://img.shields.io/badge/Triton_Server-24.10-76B900?logo=nvidia)](https://github.com/triton-inference-server/server)
[![H100](https://img.shields.io/badge/GPU-H100_PCIe_80GB-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/h100/)

Production-ready LLM inference with **Llama 3.1 8B** on **NVIDIA H100** achieving **1,700+ tok/s at 16 concurrent requests**, **11ms TTFT**, and **94% GPU utilization** — a 3.9× throughput improvement over baseline vLLM BF16.

---

## Results

| Metric              | Baseline (vLLM BF16) | Optimized (TRT-LLM FP8) | Delta    |
|---------------------|----------------------|--------------------------|----------|
| Throughput (16 req) | ~440 tok/s           | 1,700+ tok/s             | +3.9×    |
| TTFT (P99)          | ~48ms                | 11–13ms                  | −77%     |
| GPU Utilization     | ~60%                 | 94%                      | +34pp    |
| Model VRAM          | ~16GB (BF16)         | 8.6GB (FP8)              | −46%     |

**Hardware:** NVIDIA H100 PCIe 80GB | CUDA 12.x | TRT-LLM 0.15.0 | Triton 24.10

---

## Optimization Decisions

**FP8 Quantization (PTQ)**  
Reduced model VRAM from ~16GB (BF16) to 8.6GB, freeing HBM for larger batch sizes. Used calibration dataset of 512 samples with `fp8` quant format and FP8 KV cache. Accuracy delta vs BF16: <0.5% on standard benchmarks. Native H100 tensor core support means no emulation overhead — FP8 GEMM runs at full hardware throughput.

**Paged KV Cache + In-flight Batching**  
Eliminated memory fragmentation from variable-length sequences. Enabled continuous batching — the primary driver of the 1,700+ tok/s result at batch=16 vs ~440 tok/s with static batching. `kv_cache_free_gpu_mem_fraction: 0.85` tuned to maximize cache size without OOM risk.

**Paged Context FMHA (FlashAttention)**  
`use_paged_context_fmha enable` fuses attention kernels — eliminates separate softmax and matmul dispatches. Reduced attention compute time ~30%. Visible in Nsight traces below as a single wide kernel block replacing multiple sequential dispatches.

**Multiple Profiles**  
`multiple_profiles enable` allows TensorRT to compile separate execution plans for different batch sizes and sequence lengths. Prevents latency spikes when batch size varies across requests.

**What I tried that didn't work:**  
INT4 AWQ introduced >2% accuracy degradation on reasoning tasks at this model size. FP8 PTQ hit a better accuracy/performance tradeoff — VRAM savings were sufficient to unlock the batch size increases that drive throughput, without the quality penalty.

---

## Profiling (Nsight Systems)

**Trace 1 — Kernel timeline showing fused attention execution and sustained SM utilization across a 16-request batch. Note the near-continuous kernel activity with minimal idle gaps.**

<img width="1470" height="956" alt="NSIGHT-SYSTEMS1" src="https://github.com/user-attachments/assets/57bde3a1-1e8d-42e5-b9b9-bc5cd9020fab" />

**Trace 2 — Memory bandwidth and compute overlap. FP8 tensor core utilization visible in the GEMM kernel blocks. GPU remains compute-bound throughout the decode phase.**

<img width="1470" height="956" alt="NSIGHT-SYSTEMS-2" src="https://github.com/user-attachments/assets/91bc6b2f-41ff-4fb0-baa6-725ba2036d9b" />

---

## Tech Stack

| Component         | Technology                                          |
|-------------------|-----------------------------------------------------|
| Inference Engine  | TensorRT-LLM 0.15.0                                 |
| Model Serving     | Triton Inference Server 24.10                       |
| Backend API       | FastAPI + Uvicorn                                   |
| Frontend          | Reflex                                              |
| Container         | nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 |
| GPU               | NVIDIA H100 PCIe 80GB                               |

---

## Prerequisites

- NVIDIA H100 80GB (or A100 80GB)
- Docker with NVIDIA runtime
- HuggingFace account with Llama 3.1 access
- 50GB+ disk space

### Cloud GPU Options
- [Shadeform](https://shadeform.ai) — H100 PCIe ~$2.5/hr
- [Brev.dev](https://brev.dev)
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

---

## Quick Start

### 1. Clone & Setup
```bash
