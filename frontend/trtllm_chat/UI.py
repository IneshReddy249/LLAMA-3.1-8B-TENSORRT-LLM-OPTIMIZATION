import reflex as rx
import httpx
import json
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class State(rx.State):
    messages: list[Message] = []
    current_input: str = ""
    is_generating: bool = False
    ttft_ms: float = 0.0
    tps: float = 0.0
    tokens: int = 0
    latency_s: float = 0.0

    def set_input(self, value: str):
        self.current_input = value

    def clear_chat(self):
        self.messages = []
        self.ttft_ms = 0.0
        self.tps = 0.0
        self.tokens = 0
        self.latency_s = 0.0

    def handle_key(self, key: str):
        if key == "Enter":
            return State.send_message()

    async def send_message(self):
        if not self.current_input.strip() or self.is_generating:
            return
        user_msg = self.current_input.strip()
        self.messages = self.messages + [Message(role="user", content=user_msg)]
        self.current_input = ""
        self.is_generating = True
        self.ttft_ms = 0.0
        self.tps = 0.0
        self.tokens = 0
        self.latency_s = 0.0
        yield
        assistant_msg = ""
        try:
            msgs = [{"role": m.role, "content": m.content} for m in self.messages]
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", "http://localhost:8082/v1/chat/completions",
                    json={"messages": msgs, "max_tokens": 1024, "stream": True}) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and "[DONE]" not in line:
                            try:
                                data = json.loads(line[6:])
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                metrics = data.get("metrics", {})
                                if content:
                                    assistant_msg += content
                                    if "ttft_ms" in metrics:
                                        self.ttft_ms = metrics["ttft_ms"]
                                    if "tps" in metrics:
                                        self.tps = metrics["tps"]
                                    if "tokens" in metrics:
                                        self.tokens = metrics["tokens"]
                                    if "latency_s" in metrics:
                                        self.latency_s = metrics["latency_s"]
                                   
                                    if self.messages and self.messages[-1].role == "assistant":
                                        self.messages = self.messages[:-1] + [Message(role="assistant", content=assistant_msg)]
                                    else:
                                        self.messages = self.messages + [Message(role="assistant", content=assistant_msg)]
                                    yield
                            except:
                                pass
        except Exception as e:
            self.messages = self.messages + [Message(role="assistant", content=f"Error: {e}")]
        self.is_generating = False
        yield

def metric_card(label: str, value, unit: str):
    return rx.box(
        rx.text(label, color="#888", font_size="11px", font_weight="500"),
        rx.hstack(
            rx.text(value, font_size="28px", font_weight="700", color="#a855f7"),
            rx.text(unit, color="#666", font_size="12px", margin_top="6px"),
            justify="center",
            align_items="baseline",
            spacing="1",
        ),
        padding="12px 20px",
        border_radius="10px",
        background="rgba(25, 25, 35, 0.9)",
        border="1px solid rgba(255,255,255,0.08)",
        min_width="100px",
        text_align="center",
    )

def chat_bubble(msg: Message):
    is_user = msg.role == "user"
    return rx.box(
        rx.text(msg.content, color="white", white_space="pre-wrap", line_height="1.5"),
        background=rx.cond(
            is_user,
            "linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%)",
            "rgba(35, 35, 45, 0.95)"
        ),
        padding="12px 16px",
        border_radius="12px",
        max_width="75%",
        width="fit-content",
        margin_left=rx.cond(is_user, "auto", "0"),
        margin_right=rx.cond(is_user, "0", "auto"),
        margin_bottom="10px",
    )

def index():
    return rx.box(
        rx.box(
            rx.hstack(
                rx.box(width="70px"),
                rx.box(
                    rx.text("🚀 TensorRT-LLM Chat", font_size="28px", font_weight="800", color="white"),
                    rx.text("Llama 3.1 8B Instruct | H100 PCIe", font_size="14px", color="#10b981", font_weight="600", margin_top="4px"),
                    rx.text("FP8 • KV Cache • FlashAttention • In-flight Batching", font_size="12px", color="#666", margin_top="6px"),
                    rx.text(
                        "Built with Jaseci Labs Agent Architecture",
                        font_size="12px",
                        color="#666",
                        margin_top="6px",
                        as_="a",
                        href="https://www.jaseci.org/",
                        target="_blank",
                    ),
                    text_align="center",
                    flex="1",
                ),
                rx.button(
                    "Clear",
                    on_click=State.clear_chat,
                    color="#f87171",
                    background="transparent",
                    border="1px solid #f87171",
                    border_radius="8px",
                    padding="8px 16px",
                    font_size="14px",
                    cursor="pointer",
                    width="70px",
                ),
                width="100%",
                align_items="start",
                justify="between",
            ),
            rx.hstack(
                metric_card("TTFT", State.ttft_ms.to(int), "ms"),
                metric_card("Speed", State.tps.to(int), "tok/s"),
                metric_card("Tokens", State.tokens, ""),
                metric_card("Latency", State.latency_s, "s"),
                justify="center",
                spacing="3",
                margin_top="20px",
            ),
            padding="24px",
            background="linear-gradient(180deg, #08080c 0%, #0f0f15 100%)",
            position="fixed",
            top="0",
            left="0",
            right="0",
            z_index="100",
        ),
        rx.box(
            rx.foreach(State.messages, chat_bubble),
            padding="220px 24px 100px 24px",
            min_height="100vh",
            width="100%",
        ),
        rx.box(
            rx.hstack(
                rx.input(
                    value=State.current_input,
                    on_change=State.set_input,
                    on_key_down=State.handle_key,
                    placeholder="Message...",
                    background="rgba(25, 25, 35, 0.95)",
                    border="1px solid rgba(255,255,255,0.15)",
                    border_radius="12px",
                    padding="14px 18px",
                    color="white",
                    font_size="15px",
                    width="100%",
                ),
                rx.button(
                    rx.cond(State.is_generating, "...", "➤"),
                    on_click=State.send_message,
                    disabled=State.is_generating,
                    background="linear-gradient(135deg, #2563eb, #3b82f6)",
                    color="white",
                    border="none",
                    border_radius="12px",
                    width="50px",
                    height="50px",
                    font_size="18px",
                    cursor="pointer",
                ),
                spacing="3",
                width="100%",
                max_width="900px",
                margin="0 auto",
            ),
            padding="16px 24px",
            background="linear-gradient(180deg, transparent 0%, #08080c 20%)",
            position="fixed",
            bottom="0",
            left="0",
            right="0",
            z_index="100",
        ),
        min_height="100vh",
        width="100%",
        background="#0a0a0f",
    )

app = rx.App(theme=rx.theme(appearance="dark"))
app.add_page(index, title="TensorRT-LLM Chat")
