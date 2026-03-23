"""
conversation.py  —  Layer 0: Raw Conversation Log

Stores every single message ever sent/received to SQLite.
This is the "disk" in our memory hierarchy — full fidelity, never trimmed.
The other layers read from here to build their compressed views.
"""

import sqlite3
import json
import time
from pathlib import Path


DB_PATH = Path(__file__).parents[3] / "data" / "conversations.db"


def _get_conn(user_id: str) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Create per-user table if not exists
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS conv_{_safe(user_id)} (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _safe(user_id: str) -> str:
    """Sanitize user_id for use as table name suffix."""
    return "".join(c if c.isalnum() else "_" for c in user_id)


def save_message(user_id: str, role: str, content: str) -> int:
    """Persist a message. Returns the row id."""
    conn = _get_conn(user_id)
    cur = conn.execute(
        f"INSERT INTO conv_{_safe(user_id)} (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, time.time())
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def get_all_messages(user_id: str) -> list[dict]:
    """Retrieve full conversation history ordered by time."""
    conn = _get_conn(user_id)
    rows = conn.execute(
        f"SELECT role, content, timestamp FROM conv_{_safe(user_id)} ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def get_recent_messages(user_id: str, n: int) -> list[dict]:
    """Get last N messages."""
    conn = _get_conn(user_id)
    rows = conn.execute(
        f"SELECT role, content FROM conv_{_safe(user_id)} ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_message_count(user_id: str) -> int:
    conn = _get_conn(user_id)
    count = conn.execute(f"SELECT COUNT(*) FROM conv_{_safe(user_id)}").fetchone()[0]
    conn.close()
    return count
