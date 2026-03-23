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

    @staticmethod
    def _safe(user_id: str) -> str:
        """Sanitize user_id for use as a table-name suffix."""
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
        with self.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS conv_{safe}")
            conn.execute(
                "DELETE FROM core_memory WHERE user_id = ?", (user_id,)
            )
            conn.execute(
                "DELETE FROM summaries WHERE user_id = ?", (user_id,)
            )
