"""
providers/anthropic.py  —  Anthropic LLM provider

Requires: pip install anthropic
"""

from __future__ import annotations

import logging

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    LLM provider backed by the Anthropic Messages API (Claude).

    Parameters
    ----------
    api_key : Anthropic API key (or set ANTHROPIC_API_KEY env var and leave blank)
    model   : model name, default "claude-sonnet-4-6"
    config  : MemoryConfig (uses token_budget and timeout from here)
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        config: MemoryConfig | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )
        self._api_key = api_key
        self._model = model
        self._config = config or MemoryConfig()

        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or None,   # None → reads ANTHROPIC_API_KEY from env
        )

    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        opts = options or LLMOptions()

        # Anthropic separates system prompt from the messages list
        system_prompt = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)

        max_tokens = self._config.token_budget

        if not opts.stream:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=chat_messages,
                temperature=opts.temperature,
            )
            return resp.content[0].text

        # streaming path
        full_response: list[str] = []
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=chat_messages,
            temperature=opts.temperature,
        ) as stream:
            async for token in stream.text_stream:
                print(token, end="", flush=True)
                full_response.append(token)
        print()
        return "".join(full_response)

    async def summarize(self, messages: list[dict]) -> str:
        from agent_memory.layers.summary import build_summarize_request
        prompt = build_summarize_request(messages)
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.content[0].text

    async def health_check(self) -> bool:
        try:
            # Anthropic has no list-models endpoint; do a minimal API call
            await self._client.messages.create(
                model=self._model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
