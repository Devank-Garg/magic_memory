"""
providers/registry.py  —  Provider factory

create_provider(name, config) returns a ready-to-use BaseLLMProvider instance.
"""

from __future__ import annotations

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider


_KNOWN_PROVIDERS = ("ollama", "openai", "anthropic")


def create_provider(name: str, config: MemoryConfig | None = None, **kwargs) -> BaseLLMProvider:
    """
    Instantiate an LLM provider by name.

    Parameters
    ----------
    name   : one of "ollama", "openai", "anthropic"
    config : MemoryConfig to pass to the provider (defaults to MemoryConfig())
    kwargs : extra keyword arguments forwarded to the provider constructor
             (e.g. api_key="..." for openai/anthropic)

    Returns
    -------
    BaseLLMProvider subclass instance

    Raises
    ------
    ValueError  if name is not a known provider
    """
    config = config or MemoryConfig()
    name = name.lower()

    if name == "ollama":
        from agent_memory.providers.ollama import OllamaProvider
        return OllamaProvider(config=config)

    if name == "openai":
        from agent_memory.providers.openai import OpenAIProvider
        api_key = kwargs.get("api_key", "")
        model = kwargs.get("model", "gpt-4o")
        return OpenAIProvider(api_key=api_key, model=model, config=config)

    if name == "anthropic":
        from agent_memory.providers.anthropic import AnthropicProvider
        api_key = kwargs.get("api_key", "")
        model = kwargs.get("model", "claude-sonnet-4-6")
        return AnthropicProvider(api_key=api_key, model=model, config=config)

    raise ValueError(
        f"Unknown provider {name!r}. Known providers: {', '.join(_KNOWN_PROVIDERS)}"
    )
