"""
conversation.py  —  Layer 0: Raw Conversation Log

Stores every single message ever sent/received to SQLite.
This is the "disk" in our memory hierarchy — full fidelity, never trimmed.
The other layers read from here to build their compressed views.
"""

import time

from agent_memory.config import MemoryConfig
from agent_memory.storage.sqlite_store import SQLiteStore

_store = SQLiteStore(MemoryConfig().db_path)


def _ensure_table(conn, user_id: str) -> None:
    safe = SQLiteStore._safe(user_id)
    key  = f"conv_{safe}"
    if key in _store._tables_ensured:
        return
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {key} (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    _store._tables_ensured.add(key)


def save_message(user_id: str, role: str, content: str) -> int:
    """Persist a message. Returns the row id."""
    with _store.connection() as conn:
        _ensure_table(conn, user_id)
        cur = conn.execute(
            f"INSERT INTO conv_{SQLiteStore._safe(user_id)} (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, time.time()),
        )
        return cur.lastrowid


def get_all_messages(user_id: str) -> list[dict]:
    """Retrieve full conversation history ordered by time."""
    with _store.connection() as conn:
        _ensure_table(conn, user_id)
        rows = conn.execute(
            f"SELECT role, content, timestamp FROM conv_{SQLiteStore._safe(user_id)} ORDER BY id ASC"
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def get_recent_messages(user_id: str, n: int) -> list[dict]:
    """Get last N messages in chronological order.

    Uses a subquery so ordering is guaranteed by the database,
    not by Python-side reversed() (Issue 3 fix).
    """
    safe = SQLiteStore._safe(user_id)
    with _store.connection() as conn:
        _ensure_table(conn, user_id)
        rows = conn.execute(
            f"SELECT role, content FROM "
            f"(SELECT id, role, content FROM conv_{safe} ORDER BY id DESC LIMIT ?) "
            f"ORDER BY id ASC",
            (n,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_message_count(user_id: str) -> int:
    with _store.connection() as conn:
        _ensure_table(conn, user_id)
        return conn.execute(
            f"SELECT COUNT(*) FROM conv_{SQLiteStore._safe(user_id)}"
        ).fetchone()[0]
