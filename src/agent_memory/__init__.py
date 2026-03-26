"""
agent_memory — modular memory layer for AI agents.

Quick start::

    from agent_memory import MemoryEngine, MemoryConfig
    from agent_memory.providers import OllamaProvider

    engine = MemoryEngine(config=MemoryConfig(), provider=OllamaProvider())
    result = await engine.process_message("alice", "hello!")
    print(result.response)
    print(result.memory_actions)
"""

from agent_memory.config import MemoryConfig
from agent_memory.engine import MemoryEngine
from agent_memory.types import MemoryAction, MemoryResponse, MemoryState
from agent_memory.providers.registry import create_provider
from agent_memory.context_assembler import DEFAULT_BEHAVIOUR_PROMPT, build_system_prompt

__all__ = [
    "MemoryConfig",
    "MemoryEngine",
    "MemoryAction",
    "MemoryResponse",
    "MemoryState",
    "create_provider",
    "DEFAULT_BEHAVIOUR_PROMPT",
    "build_system_prompt",
]
