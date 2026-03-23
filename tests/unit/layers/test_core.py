"""Unit tests for layers/core.py"""

from agent_memory.layers import core


def test_load_returns_defaults_for_new_user(patch_core):
    data = core.load("alice")
    assert data["user_name"] == "User"
    assert data["user_facts"] == []
    assert data["scratch"] == ""


def test_update_fact_persists(patch_core):
    core.update_fact("alice", "likes Python")
    data = core.load("alice")
    assert "likes Python" in data["user_facts"]


def test_update_fact_deduplicates(patch_core):
    core.update_fact("alice", "likes Python")
    core.update_fact("alice", "likes Python")
    assert core.load("alice")["user_facts"].count("likes Python") == 1


def test_update_fact_caps_at_max(patch_core):
    for i in range(15):
        core.update_fact("alice", f"fact {i}")
    assert len(core.load("alice")["user_facts"]) == 10


def test_update_scratch_persists(patch_core):
    core.update_scratch("alice", "working on RAG")
    assert core.load("alice")["scratch"] == "working on RAG"


def test_update_scratch_caps_at_500_chars(patch_core):
    core.update_scratch("alice", "x" * 600)
    assert len(core.load("alice")["scratch"]) == 500


def test_set_user_name(patch_core):
    core.set_user_name("alice", "Alice Smith")
    assert core.load("alice")["user_name"] == "Alice Smith"


def test_render_for_prompt_contains_facts(patch_core):
    core.update_fact("alice", "works in ML")
    rendered = core.render_for_prompt("alice")
    assert "works in ML" in rendered
    assert "CORE MEMORY" in rendered
