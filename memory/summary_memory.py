"""
summary_memory.py  —  Layer 2: Rolling Summary

When conversation history grows beyond a threshold, the oldest turns are
summarized by the LLM and the raw messages are archived.

Flow:
  full history → [old turns → summary] + [recent turns verbatim]
  
The summary is stored in SQLite and injected into context as:
  "## CONVERSATION SUMMARY (older context)\n<summary text>"

This compresses 2000 tokens of old conversation into ~200 tokens.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "conversations.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            user_id    TEXT PRIMARY KEY,
            summary    TEXT NOT NULL,
            turn_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def load(user_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT summary, turn_count FROM summaries WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        return {"summary": row[0], "turn_count": row[1]}
    return {"summary": "", "turn_count": 0}


def save(user_id: str, summary: str, turn_count: int):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO summaries (user_id, summary, turn_count) VALUES (?, ?, ?)",
        (user_id, summary, turn_count)
    )
    conn.commit()
    conn.close()


def render_for_prompt(user_id: str) -> str:
    """Render summary block for system prompt. Empty string if no summary yet."""
    data = load(user_id)
    if not data["summary"]:
        return ""
    return f"""## CONVERSATION SUMMARY (older context, {data['turn_count']} turns ago)
{data['summary']}"""


SUMMARIZE_PROMPT = """You are a memory manager. Compress the following conversation turns into a dense, factual summary of at most 150 words. 
Preserve: key facts learned, decisions made, topics discussed, user preferences revealed.
Discard: pleasantries, filler, redundant information.
Return ONLY the summary paragraph, nothing else.

CONVERSATION TO SUMMARIZE:
{conversation}"""


def build_summarize_request(messages_to_summarize: list[dict]) -> str:
    """Build the prompt to send to the LLM for summarization."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_summarize
    )
    return SUMMARIZE_PROMPT.format(conversation=conv_text)
