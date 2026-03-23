"""
types.py  —  Public result types for the agent_memory API

These are the return types exposed to library consumers.
Internal layers use plain dicts; these dataclasses form the clean boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryAction:
    """
    A single memory write that was triggered during a response.

    Attributes
    ----------
    type  : one of "remember", "note", "name"
    value : the text that was stored
    """
    type: str
    value: str

    def __str__(self) -> str:
        return f"[{self.type.upper()}: {self.value}]"


@dataclass
class MemoryResponse:
    """
    The result of a single MemoryEngine.process_message() call.

    Attributes
    ----------
    response       : the LLM's reply with memory commands stripped
    memory_actions : list of MemoryAction that were applied this turn
    """
    response: str
    memory_actions: list[MemoryAction] = field(default_factory=list)


@dataclass
class MemoryState:
    """
    A snapshot of the current memory state for a user.

    Returned by MemoryEngine.get_memory_state().

    Attributes
    ----------
    user_name     : stored name, or empty string
    facts         : list of core memory facts
    scratch       : scratch pad text
    summary       : rolling summary text (empty if none yet)
    summary_turn  : turn count the summary covers
    message_count : total messages in the raw log
    """
    user_name: str
    facts: list[str]
    scratch: str
    summary: str
    summary_turn: int
    message_count: int
