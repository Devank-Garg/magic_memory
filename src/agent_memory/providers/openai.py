"""
providers/openai.py  —  OpenAI LLM provider (stub)

Full implementation planned for Phase 7.
Install the optional dependency: pip install agent-memory[openai]
"""

from __future__ import annotations

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions


class OpenAIProvider(BaseLLMProvider):
    """
    LLM provider backed by the OpenAI Chat Completions API.

    Requires: pip install openai
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", config: MemoryConfig | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._config = config or MemoryConfig()

    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        raise NotImplementedError("OpenAIProvider is not yet implemented. Coming in Phase 7.")

    async def summarize(self, messages: list[dict]) -> str:
        raise NotImplementedError("OpenAIProvider is not yet implemented. Coming in Phase 7.")

    async def health_check(self) -> bool:
        raise NotImplementedError("OpenAIProvider is not yet implemented. Coming in Phase 7.")
