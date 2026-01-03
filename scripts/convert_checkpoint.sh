#!/bin/bash
# Convert HuggingFace Llama 3.1 8B to TensorRT-LLM checkpoint

MODEL_DIR="/workspace/models/llama-3.1-8b-instruct"
OUTPUT_DIR="/workspace/models/llama-3.1-8b-checkpoint"

python3 /opt/TensorRT-LLM-examples/llama/convert_checkpoint.py \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dtype float16
