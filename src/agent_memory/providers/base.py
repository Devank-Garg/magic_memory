"""
providers/base.py  —  Abstract base class for all LLM providers

Defines the interface every provider must implement:
  - chat()         regular + streaming completions
  - summarize()    summarization call (lower temperature)
  - health_check() verify the backend is reachable and the model is available
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class LLMOptions:
    """Per-call options forwarded to the model."""
    temperature: float = 0.7
    num_ctx: int = 4096
    stream: bool = False


class BaseLLMProvider(abc.ABC):
    """
    Abstract base for LLM backends.

    Subclasses implement chat(), summarize(), and health_check().
    All methods are async.
    """

    @abc.abstractmethod
    async def chat(self, messages: list[dict], options: LLMOptions | None = None) -> str:
        """
        Send a list of chat messages and return the assistant reply as a string.

        If options.stream is True, tokens should be printed to stdout as they
        arrive, and the full response returned at the end.
        """

    @abc.abstractmethod
    async def summarize(self, messages: list[dict]) -> str:
        """
        Request a factual summary of the provided messages.

        Implementations should use a lower temperature (e.g. 0.3) and
        disable streaming.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Return True if the backend is reachable and the configured model is available.
        Should not raise — return False on any error.
        """
