"""
providers/ollama.py  —  Ollama LLM provider

Replaces root ollama_client.py.

Issue 8 fix: a single httpx.AsyncClient is created at construction time and
reused for all calls, rather than opening a new connection on every request.
"""

from __future__ import annotations

import json
import logging

import httpx

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    LLM provider backed by a local Ollama server.

    Parameters
    ----------
    config : MemoryConfig
        Reads ollama_base, model, and timeout from here.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        # Issue 8 fix: one shared client for the lifetime of this provider
        self._client = httpx.AsyncClient(timeout=self._config.timeout)

    # ── public interface ────────────────────────────────────────────────────

    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        opts = options or LLMOptions()
        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": opts.stream,
            "options": {
                "num_ctx": opts.num_ctx,
                "temperature": opts.temperature,
            },
        }

        if not opts.stream:
            resp = await self._client.post(
                f"{self._config.ollama_base}/api/chat", json=payload
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        # streaming path
        full_response: list[str] = []
        async with self._client.stream(
            "POST", f"{self._config.ollama_base}/api/chat", json=payload
        ) as resp:
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
        print()  # newline after stream
        return "".join(full_response)

    async def summarize(self, messages: list[dict]) -> str:
        from agent_memory.layers.summary import build_summarize_request
        prompt = build_summarize_request(messages)
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.3,
            },
        }
        resp = await self._client.post(
            f"{self._config.ollama_base}/api/chat", json=payload
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                f"{self._config.ollama_base}/api/tags", timeout=5.0
            )
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self._config.model in m for m in models)
        except Exception:
            return False

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call when the provider is no longer needed."""
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
