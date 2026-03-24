"""
sqlite_store.py  —  Shared SQLite connection manager

Replaces the per-function _get_conn() pattern that existed in all three
SQLite-backed layer modules. Every open/commit/close is handled here via
a context manager so callers never hold raw connections.
"""

import sqlite3
import contextlib
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # Tracks which table CREATE IF NOT EXISTS calls have already run so
        # layers can skip them on subsequent calls within the same process.
        self._tables_ensured: set[str] = set()

    @staticmethod
    def _safe(user_id: str) -> str:
        """Sanitize user_id for use as a table-name suffix.

        Raises ValueError for empty/blank user_ids, which would produce a bare
        ``conv_`` table that all callers with empty IDs would share.
        """
        if not user_id or not user_id.strip():
            raise ValueError(f"user_id must not be empty, got {user_id!r}")
        return "".join(c if c.isalnum() else "_" for c in user_id)

    @contextlib.contextmanager
    def connection(self):
        """
        Yield an open sqlite3.Connection.
        Commits on clean exit, rolls back on exception, always closes.
        """
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_user(self, user_id: str) -> None:
        """
        Remove all stored data for a user across all tables.
        Replaces the duplicated _safe() + raw SQL in main.py reset_user().
        """
        safe = self._safe(user_id)
        conv_key = f"conv_{safe}"
        with self.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {conv_key}")
            # core_memory and summaries may not exist yet (e.g. brand-new db
            # that only ever wrote conversation messages). Guard each DELETE.
            for shared_table in ("core_memory", "summaries"):
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (shared_table,),
                ).fetchone()
                if exists:
                    conn.execute(
                        f"DELETE FROM {shared_table} WHERE user_id = ?", (user_id,)
                    )
        # The conversation table was dropped; remove it from the ensured-set so
        # _ensure_table will recreate it if this user_id is used again.
        self._tables_ensured.discard(conv_key)
