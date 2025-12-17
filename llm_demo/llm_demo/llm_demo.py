import reflex as rx
import requests
import time
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TRITON_URL = "localhost:8000"
MODEL_NAME = "tensorrt_llm"
MODEL_PATH = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
hf_model.eval()
print("✓ Models ready!")


class State(rx.State):
    prompt: str = "What is machine learning? Explain in simple terms."
    max_tokens: int = 128
    
    hf_response: str = ""
    hf_time: str = "-"
    hf_tps: str = "-"
    hf_status: str = "Ready"
    
    trt_response: str = ""
    trt_time: str = "-"
    trt_tps: str = "-"
    trt_status: str = "Ready"
    
    speedup: str = "-"
    is_running: bool = False
    show_results: bool = False
    show_speedup: bool = False  # Only show after streaming complete
    
    _hf_latency: float = 0
    _trt_latency: float = 0
    _hf_full: str = ""
    _trt_full: str = ""
    
    def set_prompt(self, value: str):
        self.prompt = value
    
    async def run_comparison(self):
        self.is_running = True
        self.show_results = True
        self.show_speedup = False
        self.hf_response = ""
        self.trt_response = ""
        self.hf_status = "Running..."
        self.trt_status = "Waiting..."
        self.hf_time = "-"
        self.hf_tps = "-"
        self.trt_time = "-"
        self.trt_tps = "-"
        self.speedup = "-"
        yield
        
        messages = [{"role": "user", "content": self.prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda:0")
        
        torch.cuda.synchronize()
        hf_start = time.perf_counter()
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        self._hf_latency = time.perf_counter() - hf_start
        
        gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
        self._hf_full = tokenizer.decode(gen_ids, skip_special_tokens=True)
        hf_tokens = len(gen_ids)
        self.hf_time = f"{self._hf_latency:.2f}"
        self.hf_tps = f"{hf_tokens / self._hf_latency:.1f}"
        self.hf_status = "Complete"
        
        self.trt_status = "Running..."
        yield
        
        input_ids = tokenizer.encode(formatted)
        payload = {
            "inputs": [
                {"name": "input_ids", "shape": [1, len(input_ids)], "datatype": "INT32", "data": input_ids},
                {"name": "input_lengths", "shape": [1, 1], "datatype": "INT32", "data": [len(input_ids)]},
                {"name": "request_output_len", "shape": [1, 1], "datatype": "INT32", "data": [self.max_tokens]},
                {"name": "end_id", "shape": [1, 1], "datatype": "INT32", "data": [tokenizer.eos_token_id]},
                {"name": "pad_id", "shape": [1, 1], "datatype": "INT32", "data": [tokenizer.eos_token_id]},
            ],
            "outputs": [{"name": "output_ids"}, {"name": "sequence_length"}]
        }
        
        trt_start = time.perf_counter()
        resp = requests.post(f"http://{TRITON_URL}/v2/models/{MODEL_NAME}/infer", json=payload, timeout=300)
        self._trt_latency = time.perf_counter() - trt_start
        
        result = resp.json()
        output_ids = result["outputs"][0]["data"]
        seq_len = result["outputs"][1]["data"][0]
        gen_ids = output_ids[len(input_ids):seq_len]
        
        self._trt_full = tokenizer.decode(gen_ids, skip_special_tokens=True)
        trt_tokens = len(gen_ids)
        self.trt_time = f"{self._trt_latency:.2f}"
        self.trt_tps = f"{trt_tokens / self._trt_latency:.1f}"
        self.trt_status = "Complete"
        self.speedup = f"{self._hf_latency / self._trt_latency:.1f}"
        yield
        
        # Stream outputs
        hf_words = self._hf_full.split()
        trt_words = self._trt_full.split()
        hf_time_per_word = self._hf_latency / len(hf_words) if hf_words else 0.1
        trt_time_per_word = self._trt_latency / len(trt_words) if trt_words else 0.1
        
        hf_idx = trt_idx = 0
        hf_timer = trt_timer = 0.0
        start_stream = time.perf_counter()
        
        while hf_idx < len(hf_words) or trt_idx < len(trt_words):
            elapsed = time.perf_counter() - start_stream
            while trt_idx < len(trt_words) and trt_timer <= elapsed:
                self.trt_response = " ".join(trt_words[:trt_idx + 1])
                trt_idx += 1
                trt_timer += trt_time_per_word
            while hf_idx < len(hf_words) and hf_timer <= elapsed:
                self.hf_response = " ".join(hf_words[:hf_idx + 1])
                hf_idx += 1
                hf_timer += hf_time_per_word
            yield
            await asyncio.sleep(0.02)
        
        # Show speedup only after streaming is done
        self.show_speedup = True
        self.is_running = False
        yield


def index() -> rx.Component:
    return rx.box(
        # Header
        rx.box(
            rx.center(
                rx.hstack(
                    rx.icon("zap", size=28, color="white"),
                    rx.heading("LLM Inference Optimization", size="7", color="white"),
                    rx.text("•", color="rgba(255,255,255,0.5)", margin_x="10px"),
                    rx.text("Llama 3.1 8B", color="rgba(255,255,255,0.8)", font_size="14px"),
                    rx.text("•", color="rgba(255,255,255,0.5)", margin_x="10px"),
                    rx.text("A100 80GB", color="rgba(255,255,255,0.8)", font_size="14px"),
                    align="center",
                    spacing="2",
                ),
            ),
            width="100%",
            padding_y="20px",
            background="linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%)",
        ),
        
        # Baseline vs Optimized - Bigger boxes
        rx.center(
            rx.hstack(
                rx.box(
                    rx.hstack(
                        rx.box(width="14px", height="14px", background="#ef4444", border_radius="4px"),
                        rx.vstack(
                            rx.text("Baseline", font_weight="700", font_size="15px"),
                            rx.text("HuggingFace • FP16", font_size="13px", color="gray"),
                            spacing="0",
                            align="start",
                        ),
                        align="center",
                        spacing="3",
                    ),
                    padding="14px 24px",
                    background="white",
                    border="2px solid #fecaca",
                    border_radius="10px",
                ),
                rx.text("vs", color="gray", font_size="16px", font_weight="500"),
                rx.box(
                    rx.hstack(
                        rx.box(width="14px", height="14px", background="#22c55e", border_radius="4px"),
                        rx.vstack(
                            rx.text("Optimized", font_weight="700", font_size="15px"),
                            rx.text("TensorRT-LLM • INT8", font_size="13px", color="gray"),
                            spacing="0",
                            align="start",
                        ),
                        align="center",
                        spacing="3",
                    ),
                    padding="14px 24px",
                    background="white",
                    border="2px solid #86efac",
                    border_radius="10px",
                ),
                spacing="5",
                align="center",
            ),
            padding_y="20px",
        ),
        
        # Main Content
        rx.center(
            rx.vstack(
                # Input Row
                rx.hstack(
                    rx.text_area(
                        value=State.prompt,
                        on_change=State.set_prompt,
                        placeholder="Enter a prompt...",
                        width="100%",
                        min_height="60px",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("play", size=16), rx.text("Run"), spacing="2"),
                        on_click=State.run_comparison,
                        loading=State.is_running,
                        size="3",
                        height="60px",
                    ),
                    spacing="3",
                    width="100%",
                    align="end",
                ),
                
                # Output Cards
                rx.cond(
                    State.show_results,
                    rx.hstack(
                        # HuggingFace
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.hstack(
                                        rx.box(width="10px", height="10px", background="#ef4444", border_radius="2px"),
                                        rx.text("HuggingFace", font_weight="600"),
                                        spacing="2",
                                    ),
                                    rx.spacer(),
                                    rx.badge(State.hf_status, variant="soft", size="1"),
                                    width="100%",
                                ),
                                rx.box(
                                    rx.text(State.hf_response, font_size="14px", line_height="1.7"),
                                    height="350px",
                                    overflow_y="auto",
                                    padding="16px",
                                    background="#f9fafb",
                                    border_radius="8px",
                                    width="100%",
                                ),
                                rx.hstack(
                                    rx.text(f"⏱ {State.hf_time}s", font_size="14px", font_weight="500"),
                                    rx.text("•", color="gray"),
                                    rx.text(f"⚡ {State.hf_tps} tok/s", font_size="14px"),
                                    spacing="3",
                                    justify="center",
                                    width="100%",
                                ),
                                width="100%",
                                spacing="3",
                            ),
                            padding="16px",
                            background="white",
                            border="1px solid #e5e5e5",
                            border_radius="12px",
                            width="50%",
                        ),
                        # TensorRT-LLM
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.hstack(
                                        rx.box(width="10px", height="10px", background="#22c55e", border_radius="2px"),
                                        rx.text("TensorRT-LLM", font_weight="600"),
                                        spacing="2",
                                    ),
                                    rx.spacer(),
                                    rx.badge(State.trt_status, color_scheme="green", variant="soft", size="1"),
                                    width="100%",
                                ),
                                rx.box(
                                    rx.text(State.trt_response, font_size="14px", line_height="1.7"),
                                    height="350px",
                                    overflow_y="auto",
                                    padding="16px",
                                    background="#f0fdf4",
                                    border_radius="8px",
                                    width="100%",
                                ),
                                rx.hstack(
                                    rx.text(f"⏱ {State.trt_time}s", font_size="14px", font_weight="500"),
                                    rx.text("•", color="gray"),
                                    rx.text(f"⚡ {State.trt_tps} tok/s", font_size="14px", color="#22c55e", font_weight="600"),
                                    spacing="3",
                                    justify="center",
                                    width="100%",
                                ),
                                width="100%",
                                spacing="3",
                            ),
                            padding="16px",
                            background="white",
                            border="2px solid #22c55e",
                            border_radius="12px",
                            width="50%",
                        ),
                        spacing="4",
                        width="100%",
                    ),
                ),
                
                # Speedup Badge - Only after streaming complete
                rx.cond(
                    State.show_speedup,
                    rx.center(
                        rx.hstack(
                            rx.icon("zap", size=24, color="#22c55e"),
                            rx.text(f"{State.speedup}x Faster", font_size="20px", font_weight="700", color="#22c55e"),
                            spacing="2",
                        ),
                        padding="12px 30px",
                        background="#f0fdf4",
                        border="2px solid #22c55e",
                        border_radius="30px",
                        margin_top="10px",
                    ),
                ),
                
                # Footer
                rx.center(
                    rx.hstack(
                        rx.badge("INT8 Quantization", variant="outline", size="1"),
                        rx.badge("FlashAttention", variant="outline", size="1"),
                        rx.badge("Paged KV Cache", variant="outline", size="1"),
                        rx.badge("Triton Server", variant="outline", size="1"),
                        spacing="2",
                    ),
                    padding_y="15px",
                ),
                
                max_width="1100px",
                width="100%",
                spacing="4",
                padding_x="20px",
                padding_y="20px",
            ),
            width="100%",
        ),
        
        min_height="100vh",
        background="#f1f5f9",
    )


app = rx.App()
app.add_page(index)
