"""Unit tests for storage/sqlite_store.py"""

import pytest
from agent_memory.storage.sqlite_store import SQLiteStore


def test_safe_empty_string_raises():
    with pytest.raises(ValueError, match="user_id"):
        SQLiteStore._safe("")


def test_safe_blank_string_raises():
    with pytest.raises(ValueError, match="user_id"):
        SQLiteStore._safe("   ")


def test_safe_normal_user_id():
    assert SQLiteStore._safe("alice") == "alice"


def test_safe_replaces_non_alnum():
    # user-1 and user_1 both become user_1 — document the known collision
    assert SQLiteStore._safe("user-1") == "user_1"
    assert SQLiteStore._safe("user_1") == "user_1"


def test_connection_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "test.db"
    store = SQLiteStore(nested)
    with store.connection() as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    assert nested.exists()


def test_delete_user_removes_tables(tmp_path):
    from agent_memory.layers import conversation, core
    store = SQLiteStore(tmp_path / "del.db")
    # Monkeypatch layers to use this store
    conversation._store = store
    core._store = store

    conversation.save_message("bob", "user", "hello")
    core.update_fact("bob", "likes tea")

    store.delete_user("bob")

    assert conversation.get_message_count("bob") == 0
    data = core.load("bob")
    assert data["user_facts"] == []
