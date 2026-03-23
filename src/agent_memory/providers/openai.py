"""
providers/openai.py  —  OpenAI LLM provider

Requires: pip install openai
"""

from __future__ import annotations

import logging

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    LLM provider backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key : OpenAI API key (or set OPENAI_API_KEY env var and leave blank)
    model   : model name, default "gpt-4o"
    config  : MemoryConfig (uses timeout from here)
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        config: MemoryConfig | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )
        self._api_key = api_key
        self._model = model
        self._config = config or MemoryConfig()

        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=api_key or None,   # None → reads OPENAI_API_KEY from env
            timeout=self._config.timeout,
        )

    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        opts = options or LLMOptions()

        if not opts.stream:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=opts.temperature,
            )
            return resp.choices[0].message.content

        # streaming path
        full_response: list[str] = []
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=opts.temperature,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                print(token, end="", flush=True)
                full_response.append(token)
        print()
        return "".join(full_response)

    async def summarize(self, messages: list[dict]) -> str:
        from agent_memory.layers.summary import build_summarize_request
        prompt = build_summarize_request(messages)
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return any(self._model in m.id for m in models.data)
        except Exception:
            return False
