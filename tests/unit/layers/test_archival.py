"""Unit tests for layers/archival.py"""

import pytest
from unittest.mock import MagicMock, patch


def test_archive_message_non_fatal_on_chromadb_failure(monkeypatch):
    """archive_message must not raise even when ChromaDB is unavailable (Issue 4)."""
    from agent_memory.layers import archival
    from agent_memory.storage.chroma_store import ChromaStore

    broken = ChromaStore.__new__(ChromaStore)
    broken.similarity_threshold = 0.7
    broken.archival_top_k = 3

    def boom(*a, **kw):
        raise RuntimeError("ChromaDB unavailable")

    broken.get_embedder = boom
    broken.get_collection = boom

    monkeypatch.setattr(archival, "_chroma", broken)

    # Should not raise
    archival.archive_message("alice", "user", "hello world!", 1)


def test_archive_message_skips_short_content(monkeypatch):
    """Messages under 10 chars are silently skipped."""
    from agent_memory.layers import archival

    mock_chroma = MagicMock()
    monkeypatch.setattr(archival, "_chroma", mock_chroma)

    archival.archive_message("alice", "user", "hi", 1)
    mock_chroma.get_embedder.assert_not_called()


def test_search_returns_empty_on_empty_collection(monkeypatch):
    """search() returns [] when the collection has no documents."""
    from agent_memory.layers import archival
    from agent_memory.storage.chroma_store import ChromaStore

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    mock_chroma = MagicMock(spec=ChromaStore)
    mock_chroma.similarity_threshold = 0.7
    mock_chroma.archival_top_k = 3
    mock_chroma.get_collection.return_value = mock_collection

    monkeypatch.setattr(archival, "_chroma", mock_chroma)

    result = archival.search("alice", "anything")
    assert result == []


def test_search_non_fatal_on_failure(monkeypatch):
    """search() returns [] instead of raising on any exception."""
    from agent_memory.layers import archival
    from agent_memory.storage.chroma_store import ChromaStore

    broken = MagicMock(spec=ChromaStore)
    broken.similarity_threshold = 0.7
    broken.archival_top_k = 3
    broken.get_embedder.side_effect = RuntimeError("boom")

    monkeypatch.setattr(archival, "_chroma", broken)

    result = archival.search("alice", "query")
    assert result == []
