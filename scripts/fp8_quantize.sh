python3 /opt/TensorRT-LLM-examples/quantization/quantize.py \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dtype float16 \
    --qformat fp8 \
    --calib_dataset $CALIB_DATA \
    --calib_size 512 \
    --kv_cache_dtype fp8
