"""
providers/anthropic.py  —  Anthropic LLM provider (stub)

Full implementation planned for Phase 7.
Install the optional dependency: pip install agent-memory[anthropic]
"""

from __future__ import annotations

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions


class AnthropicProvider(BaseLLMProvider):
    """
    LLM provider backed by the Anthropic Messages API (Claude).

    Requires: pip install anthropic
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", config: MemoryConfig | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._config = config or MemoryConfig()

    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        raise NotImplementedError("AnthropicProvider is not yet implemented. Coming in Phase 7.")

    async def summarize(self, messages: list[dict]) -> str:
        raise NotImplementedError("AnthropicProvider is not yet implemented. Coming in Phase 7.")

    async def health_check(self) -> bool:
        raise NotImplementedError("AnthropicProvider is not yet implemented. Coming in Phase 7.")
