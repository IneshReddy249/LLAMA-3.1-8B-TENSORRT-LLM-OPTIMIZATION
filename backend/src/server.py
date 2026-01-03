from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import time
import uuid
import numpy as np
import queue
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

tokenizer = AutoTokenizer.from_pretrained("/workspace/tokenizer")
STOP_TOKENS = {128001, 128008, 128009}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "llama"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = True

def format_prompt(messages):
    prompt = "<|begin_of_text|>"
    for m in messages:
        prompt += f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    prompt = format_prompt(request.messages)
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int32)
    
    inputs = [
        grpcclient.InferInput("input_ids", input_ids.shape, "INT32"),
        grpcclient.InferInput("input_lengths", [1, 1], "INT32"),
        grpcclient.InferInput("request_output_len", [1, 1], "INT32"),
        grpcclient.InferInput("streaming", [1, 1], "BOOL"),
        grpcclient.InferInput("end_id", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(np.array([[input_ids.shape[1]]], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[request.max_tokens]], dtype=np.int32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype=bool))
    inputs[4].set_data_from_numpy(np.array([[128009]], dtype=np.int32))
    
    req_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    token_queue = queue.Queue()
    
    def callback(result, error):
        if error:
            token_queue.put(None)
        elif result:
            out = result.as_numpy("output_ids")
            if out is not None and out.size > 0:
                token_queue.put(int(out.flatten()[-1]))
            else:
                token_queue.put(None)
    
    def generate():
        first_token_time = None
        token_count = 0
        client.start_stream(callback=callback)
        client.async_stream_infer("tensorrt_llm", inputs, request_id=req_id)
        
        while True:
            try:
                token_id = token_queue.get(timeout=30)
                if token_id is None or token_id in STOP_TOKENS:
                    break
                token = tokenizer.decode([token_id], skip_special_tokens=True)
                if token:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                    
                    ttft_ms = (first_token_time - start_time) * 1000
                    latency_s = now - start_time
                    tps = token_count / (now - first_token_time) if now > first_token_time else 0
                    
                    chunk = {
                        "choices": [{"delta": {"content": token}}],
                        "metrics": {
                            "ttft_ms": round(ttft_ms, 1),
                            "tps": round(tps, 1),
                            "tokens": token_count,
                            "latency_s": round(latency_s, 2)
                        }
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            except queue.Empty:
                break
        client.stop_stream()
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
