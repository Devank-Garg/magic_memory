"""Unit tests for layers/summary.py"""

from agent_memory.layers import summary


def test_load_returns_empty_for_new_user(patch_summary):
    data = summary.load("alice")
    assert data["summary"] == ""
    assert data["turn_count"] == 0


def test_save_and_load_roundtrip(patch_summary):
    summary.save("alice", "we discussed Python and RAG", 10)
    data = summary.load("alice")
    assert data["summary"] == "we discussed Python and RAG"
    assert data["turn_count"] == 10


def test_save_overwrites_previous(patch_summary):
    summary.save("alice", "first summary", 5)
    summary.save("alice", "updated summary", 10)
    assert summary.load("alice")["summary"] == "updated summary"


def test_render_for_prompt_empty_when_no_summary(patch_summary):
    assert summary.render_for_prompt("alice") == ""


def test_render_for_prompt_includes_summary(patch_summary):
    summary.save("alice", "talked about agents", 8)
    rendered = summary.render_for_prompt("alice")
    assert "talked about agents" in rendered
    assert "CONVERSATION SUMMARY" in rendered


def test_separate_users_isolated(patch_summary):
    summary.save("alice", "alice summary", 5)
    assert summary.load("bob")["summary"] == ""
