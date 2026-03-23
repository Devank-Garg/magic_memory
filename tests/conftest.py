"""
Shared pytest fixtures for agent_memory tests.
"""

import pytest
from agent_memory.storage.sqlite_store import SQLiteStore


@pytest.fixture
def db_store(tmp_path):
    """A SQLiteStore backed by a temporary file — isolated per test."""
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture
def patch_conversation(db_store, monkeypatch):
    """Redirect the conversation layer to use the test SQLiteStore."""
    from agent_memory.layers import conversation
    monkeypatch.setattr(conversation, "_store", db_store)
    return db_store


@pytest.fixture
def patch_core(db_store, monkeypatch):
    """Redirect the core layer to use the test SQLiteStore."""
    from agent_memory.layers import core
    monkeypatch.setattr(core, "_store", db_store)
    return db_store


@pytest.fixture
def patch_summary(db_store, monkeypatch):
    """Redirect the summary layer to use the test SQLiteStore."""
    from agent_memory.layers import summary
    monkeypatch.setattr(summary, "_store", db_store)
    return db_store
