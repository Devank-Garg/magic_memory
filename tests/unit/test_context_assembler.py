"""Unit tests for context_assembler.py — focus on Issue 2 (budget overflow)"""

import pytest
from unittest.mock import patch
from agent_memory.config import MemoryConfig
from agent_memory import context_assembler


@pytest.fixture
def db_store(tmp_path):
    from agent_memory.storage.sqlite_store import SQLiteStore
    return SQLiteStore(tmp_path / "assembler_test.db")


@pytest.fixture
def patched_layers(db_store, monkeypatch):
    from agent_memory.layers import conversation, core, summary, archival
    monkeypatch.setattr(conversation, "_store", db_store)
    monkeypatch.setattr(core, "_store", db_store)
    monkeypatch.setattr(summary, "_store", db_store)
    # archival search not needed; patch to return empty
    monkeypatch.setattr(archival, "render_for_prompt", lambda *a, **kw: "")
    return db_store


def test_build_context_returns_messages(patched_layers):
    config = MemoryConfig(token_budget=3000)
    msgs = context_assembler.build_context("alice", "hello", config)
    assert len(msgs) >= 1
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "hello"


def test_build_context_budget_overflow_still_returns_valid_list(patched_layers):
    """Issue 2: even with a tiny budget the function must not crash or return
    an empty list — system + user message are always included."""
    config = MemoryConfig(token_budget=1)  # absurdly small
    msgs = context_assembler.build_context("alice", "hi", config)
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles


def test_available_for_history_never_negative(patched_layers):
    """With a tiny budget, history window is 0 messages — no crash, no negative slice."""
    from agent_memory.layers import conversation
    # Add some history first
    for i in range(5):
        conversation.save_message("alice", "user", f"message {i}")
        conversation.save_message("alice", "assistant", f"reply {i}")

    config = MemoryConfig(token_budget=10)
    # Should not raise
    msgs = context_assembler.build_context("alice", "new message", config)
    assert msgs[-1]["content"] == "new message"


def test_should_summarize_false_initially(patched_layers):
    config = MemoryConfig(summarize_after_turns=15)
    assert context_assembler.should_summarize("alice", config) is False


def test_should_summarize_true_after_threshold(patched_layers):
    from agent_memory.layers import conversation
    config = MemoryConfig(summarize_after_turns=4)
    for i in range(5):
        conversation.save_message("alice", "user", f"msg {i}")
    assert context_assembler.should_summarize("alice", config) is True
