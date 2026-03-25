"""
config.py  —  Single source of truth for all configuration.

All constants previously scattered across context_assembler.py,
ollama_client.py, and layer DB_PATH definitions live here.

Usage:
    config = MemoryConfig()                          # all defaults
    config = MemoryConfig(token_budget=4000)         # override one field
    config = MemoryConfig.from_env()                 # read AGENT_MEMORY_* env vars
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root: src/agent_memory/config.py → parent × 3 = project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class MemoryConfig:
    # ── Storage paths ──────────────────────────────────────────────────────────
    db_path:     Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "conversations.db")
    chroma_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "chroma")

    # ── Token budget ───────────────────────────────────────────────────────────
    token_budget:       int = 3000   # total tokens sent to model (input side)
    response_reserve:   int = 1000   # tokens reserved for model response
    recent_turns_window: int = 10    # number of recent turns to include verbatim
    summarize_after_turns: int = 15  # trigger summarization after N total turns

    # ── Core memory limits ─────────────────────────────────────────────────────
    core_memory_max_facts:        int = 10
    core_memory_max_scratch_chars: int = 500

    # ── Archival memory ────────────────────────────────────────────────────────
    archival_similarity_threshold: float = 0.7
    archival_top_k:                int   = 3
    embedder_model:                str   = "all-MiniLM-L6-v2"

    # ── LLM backend ────────────────────────────────────────────────────────────
    ollama_base: str   = "http://localhost:11434"
    model:       str   = "ministral-3:3b"
    timeout:     float = 120.0

    # ── System prompt override ──────────────────────────────────────────────────
    # Replaces the default BEHAVIOUR + MEMORY COMMANDS block injected at the
    # end of the system prompt.  The memory context (core memory, rolling
    # summary, archival results) is always prepended regardless of this setting,
    # so the memory system remains functional even with a fully custom prompt.
    # Set via AGENT_MEMORY_SYSTEM_PROMPT env var or pass directly:
    #
    #   from agent_memory.context_assembler import DEFAULT_BEHAVIOUR_PROMPT
    #   config = MemoryConfig(
    #       system_prompt=DEFAULT_BEHAVIOUR_PROMPT + "\n\nYou are a support agent."
    #   )
    system_prompt: str | None = None

    def __post_init__(self):
        # Normalise paths so callers always get Path objects
        self.db_path     = Path(self.db_path)
        self.chroma_path = Path(self.chroma_path)

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Build a MemoryConfig from AGENT_MEMORY_* environment variables.
        Any unset variable falls back to the dataclass default.
        """
        kwargs: dict = {}
        _str  = lambda k: os.getenv(k)
        _int  = lambda k: int(os.environ[k]) if k in os.environ else None
        _flt  = lambda k: float(os.environ[k]) if k in os.environ else None

        if (v := _str("AGENT_MEMORY_DB_PATH"))        is not None: kwargs["db_path"]     = Path(v)
        if (v := _str("AGENT_MEMORY_CHROMA_PATH"))    is not None: kwargs["chroma_path"] = Path(v)
        if (v := _int("AGENT_MEMORY_TOKEN_BUDGET"))   is not None: kwargs["token_budget"] = v
        if (v := _int("AGENT_MEMORY_RESPONSE_RESERVE")) is not None: kwargs["response_reserve"] = v
        if (v := _int("AGENT_MEMORY_RECENT_TURNS"))   is not None: kwargs["recent_turns_window"] = v
        if (v := _int("AGENT_MEMORY_SUMMARIZE_AFTER")) is not None: kwargs["summarize_after_turns"] = v
        if (v := _int("AGENT_MEMORY_MAX_FACTS"))      is not None: kwargs["core_memory_max_facts"] = v
        if (v := _int("AGENT_MEMORY_MAX_SCRATCH"))    is not None: kwargs["core_memory_max_scratch_chars"] = v
        if (v := _flt("AGENT_MEMORY_ARCHIVAL_THRESHOLD")) is not None: kwargs["archival_similarity_threshold"] = v
        if (v := _int("AGENT_MEMORY_ARCHIVAL_TOP_K")) is not None: kwargs["archival_top_k"] = v
        if (v := _str("AGENT_MEMORY_EMBEDDER_MODEL")) is not None: kwargs["embedder_model"] = v
        if (v := _str("AGENT_MEMORY_OLLAMA_BASE"))    is not None: kwargs["ollama_base"] = v
        if (v := _str("AGENT_MEMORY_MODEL"))          is not None: kwargs["model"] = v
        if (v := _flt("AGENT_MEMORY_TIMEOUT"))        is not None: kwargs["timeout"] = v
        if (v := _str("AGENT_MEMORY_SYSTEM_PROMPT"))  is not None: kwargs["system_prompt"] = v

        return cls(**kwargs)
