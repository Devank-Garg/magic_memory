"""Unit tests for MemoryConfig.from_env()"""

import os
from agent_memory.config import MemoryConfig


def test_from_env_defaults(monkeypatch):
    """from_env() with no env vars returns the same as the default config."""
    for key in [
        "AGENT_MEMORY_DB_PATH", "AGENT_MEMORY_CHROMA_PATH",
        "AGENT_MEMORY_TOKEN_BUDGET", "AGENT_MEMORY_RECENT_TURNS",
        "AGENT_MEMORY_SUMMARIZE_AFTER", "AGENT_MEMORY_MAX_FACTS",
        "AGENT_MEMORY_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)

    cfg = MemoryConfig.from_env()
    default = MemoryConfig()
    assert cfg.token_budget == default.token_budget
    assert cfg.model == default.model


def test_from_env_overrides(monkeypatch):
    """from_env() picks up every AGENT_MEMORY_* variable."""
    monkeypatch.setenv("AGENT_MEMORY_TOKEN_BUDGET", "5000")
    monkeypatch.setenv("AGENT_MEMORY_RECENT_TURNS", "5")
    monkeypatch.setenv("AGENT_MEMORY_SUMMARIZE_AFTER", "20")
    monkeypatch.setenv("AGENT_MEMORY_MAX_FACTS", "8")
    monkeypatch.setenv("AGENT_MEMORY_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_MEMORY_OLLAMA_BASE", "http://localhost:11434")
    monkeypatch.setenv("AGENT_MEMORY_ARCHIVAL_THRESHOLD", "0.85")
    monkeypatch.setenv("AGENT_MEMORY_ARCHIVAL_TOP_K", "5")
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER_MODEL", "all-mpnet-base-v2")
    monkeypatch.setenv("AGENT_MEMORY_TIMEOUT", "30.0")

    cfg = MemoryConfig.from_env()
    assert cfg.token_budget == 5000
    assert cfg.recent_turns_window == 5
    assert cfg.summarize_after_turns == 20
    assert cfg.core_memory_max_facts == 8
    assert cfg.model == "gpt-4o"
    assert cfg.ollama_base == "http://localhost:11434"
    assert cfg.archival_similarity_threshold == 0.85
    assert cfg.archival_top_k == 5
    assert cfg.embedder_model == "all-mpnet-base-v2"
    assert cfg.timeout == 30.0


def test_from_env_path_overrides(monkeypatch, tmp_path):
    """DB and chroma paths are converted to Path objects."""
    monkeypatch.setenv("AGENT_MEMORY_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("AGENT_MEMORY_CHROMA_PATH", str(tmp_path / "chroma"))

    cfg = MemoryConfig.from_env()
    assert cfg.db_path == tmp_path / "mem.db"
    assert cfg.chroma_path == tmp_path / "chroma"
