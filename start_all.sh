#!/bin/bash

echo "Starting Triton..."
tritonserver --model-repository=/workspace/triton_model_repo > triton.log 2>&1 &
sleep 15
curl -s localhost:8000/v2/health/ready && echo "✓ Triton OK" || echo "✗ Triton FAILED"

echo "Starting FastAPI..."
cd /workspace/backend/src && python3 server.py > /workspace/fastapi.log 2>&1 &
sleep 3
curl -s localhost:8082/health && echo "✓ FastAPI OK" || echo "✗ FastAPI FAILED"

echo "Starting Reflex..."
cd /workspace/frontend && reflex run --env prod
```

---

### File 4: `requirements.txt`

**Path:** `requirements.txt`
```
fastapi
uvicorn
tritonclient[grpc]
transformers
numpy
reflex
httpx
datasets
