"""
core.py  —  Layer 1: Core Memory (MemGPT-style)

A small, always-present block of key facts about the user.
Always injected into the system prompt — never evicted.
The LLM can update this by returning special commands.

Structure:
  - persona:    who the assistant is for this user
  - user_facts: name, preferences, important context
  - scratch:    LLM's working notes (rewritten each session)

Max size: ~300 tokens  (kept tiny — it's always in context)
"""

import json

from agent_memory.config import MemoryConfig
from agent_memory.storage.sqlite_store import SQLiteStore

_store = SQLiteStore(MemoryConfig().db_path)


_CORE_TABLE_KEY = "core_memory"


def _ensure_table(conn) -> None:
    if _CORE_TABLE_KEY in _store._tables_ensured:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core_memory (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL
        )
    """)
    _store._tables_ensured.add(_CORE_TABLE_KEY)


def _default(user_id: str) -> dict:
    return {
        "user_name": "User",
        "user_facts": [],
        "assistant_persona": "You are a helpful, concise assistant with an excellent memory.",
        "scratch": "",
    }


def load(user_id: str) -> dict:
    with _store.connection() as conn:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT data FROM core_memory WHERE user_id = ?", (user_id,)
        ).fetchone()
    return json.loads(row[0]) if row else _default(user_id)


def save(user_id: str, data: dict) -> None:
    with _store.connection() as conn:
        _ensure_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO core_memory (user_id, data) VALUES (?, ?)",
            (user_id, json.dumps(data)),
        )


def update_fact(user_id: str, fact: str, config: MemoryConfig | None = None) -> None:
    """Add a user fact (deduplicates; caps at core_memory_max_facts)."""
    config = config or MemoryConfig()
    data = load(user_id)
    if fact not in data["user_facts"]:
        data["user_facts"].append(fact)
        data["user_facts"] = data["user_facts"][-config.core_memory_max_facts:]
    save(user_id, data)


def update_scratch(user_id: str, note: str, config: MemoryConfig | None = None) -> None:
    config = config or MemoryConfig()
    data = load(user_id)
    data["scratch"] = note[:config.core_memory_max_scratch_chars]
    save(user_id, data)


def set_user_name(user_id: str, name: str) -> None:
    data = load(user_id)
    data["user_name"] = name
    save(user_id, data)


def render_for_prompt(user_id: str) -> str:
    """Render core memory as a formatted string for the system prompt."""
    data = load(user_id)
    facts_str = "\n".join(f"  - {f}" for f in data["user_facts"]) or "  (none yet)"
    scratch_str = data["scratch"] or "(empty)"
    return (
        f"## CORE MEMORY (always present)\n"
        f"Assistant Persona: {data['assistant_persona']}\n"
        f"User Name: {data['user_name']}\n"
        f"Known User Facts:\n{facts_str}\n"
        f"Working Notes: {scratch_str}"
    )
