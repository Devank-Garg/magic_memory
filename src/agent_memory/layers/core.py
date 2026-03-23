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
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[3] / "data" / "conversations.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core_memory (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _default(user_id: str) -> dict:
    return {
        "user_name": "User",
        "user_facts": [],       # e.g. ["Works in ML", "Prefers Python", "Lives in Delhi"]
        "assistant_persona": "You are a helpful, concise assistant with an excellent memory.",
        "scratch": ""           # LLM can write short working notes here
    }


def load(user_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute("SELECT data FROM core_memory WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return _default(user_id)


def save(user_id: str, data: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO core_memory (user_id, data) VALUES (?, ?)",
        (user_id, json.dumps(data))
    )
    conn.commit()
    conn.close()


def update_fact(user_id: str, fact: str):
    """Add or update a user fact (deduplicates similar facts)."""
    data = load(user_id)
    # Simple dedup: avoid exact duplicates
    if fact not in data["user_facts"]:
        data["user_facts"].append(fact)
        # Keep only last 10 facts to stay within token budget
        data["user_facts"] = data["user_facts"][-10:]
    save(user_id, data)


def update_scratch(user_id: str, note: str):
    data = load(user_id)
    data["scratch"] = note[:500]  # hard cap scratch to 500 chars
    save(user_id, data)


def set_user_name(user_id: str, name: str):
    data = load(user_id)
    data["user_name"] = name
    save(user_id, data)


def render_for_prompt(user_id: str) -> str:
    """Render core memory as a formatted string for the system prompt."""
    data = load(user_id)
    facts_str = "\n".join(f"  - {f}" for f in data["user_facts"]) or "  (none yet)"
    scratch_str = data["scratch"] or "(empty)"
    return f"""## CORE MEMORY (always present)
Assistant Persona: {data['assistant_persona']}
User Name: {data['user_name']}
Known User Facts:
{facts_str}
Working Notes: {scratch_str}"""
