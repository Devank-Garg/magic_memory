"""
ollama_client.py  —  Async Ollama API wrapper

Handles:
  - Regular (non-streaming) completions
  - Streaming completions (token by token)
  - Summarization calls (separate, non-streaming)
"""

import json
import httpx

from agent_memory.config import MemoryConfig


async def chat(messages: list[dict], stream: bool = False, config: MemoryConfig = None) -> str:
    """
    Send messages to Ollama.
    stream=False: returns full response string
    stream=True:  prints tokens as they arrive, returns full string
    """
    config = config or MemoryConfig()
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": stream,
        "options": {
            "num_ctx": 4096,       # model context window
            "temperature": 0.7,
        }
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        if not stream:
            resp = await client.post(f"{config.ollama_base}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            full_response = []
            async with client.stream("POST", f"{config.ollama_base}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response.append(token)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            print()  # newline after streaming
            return "".join(full_response)


async def summarize(messages_to_summarize: list[dict], config: MemoryConfig = None) -> str:
    """Call LLM specifically for summarization (non-streaming, lower temp)."""
    config = config or MemoryConfig()
    from agent_memory.layers.summary import build_summarize_request
    prompt = build_summarize_request(messages_to_summarize)

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0.3,   # lower temp for factual summaries
        }
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        resp = await client.post(f"{config.ollama_base}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def check_ollama(config: MemoryConfig = None) -> bool:
    """Verify Ollama is running and model is available."""
    config = config or MemoryConfig()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.ollama_base}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(config.model in m for m in models)
    except Exception:
        return False
