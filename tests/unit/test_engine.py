"""Unit tests for engine.py (MemoryEngine)"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_memory.engine import MemoryEngine
from agent_memory.config import MemoryConfig
from agent_memory.types import MemoryResponse, MemoryAction, MemoryState


@pytest.fixture
def db_store(tmp_path):
    from agent_memory.storage.sqlite_store import SQLiteStore
    return SQLiteStore(tmp_path / "engine_test.db")


@pytest.fixture
def patched_layers(tmp_path, monkeypatch):
    """Point all layer _store instances at the same temp DB the engine uses."""
    from agent_memory.storage.sqlite_store import SQLiteStore
    from agent_memory.layers import conversation, core, summary
    store = SQLiteStore(tmp_path / "engine_test.db")
    monkeypatch.setattr(conversation, "_store", store)
    monkeypatch.setattr(core, "_store", store)
    monkeypatch.setattr(summary, "_store", store)
    return store


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value="Hello there!")
    provider.summarize = AsyncMock(return_value="Summary text.")
    return provider


@pytest.fixture
def engine(tmp_path, mock_provider):
    config = MemoryConfig(
        token_budget=3000,
        summarize_after_turns=100,
        db_path=tmp_path / "engine_test.db",
        chroma_path=tmp_path / "chroma",
    )
    return MemoryEngine(config=config, provider=mock_provider)


# ── process_message ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_message_returns_memory_response(engine, patched_layers):
    result = await engine.process_message("alice", "hi")
    assert isinstance(result, MemoryResponse)
    assert result.response == "Hello there!"
    assert result.memory_actions == []


@pytest.mark.asyncio
async def test_process_message_strips_memory_commands(engine, patched_layers, mock_provider):
    mock_provider.chat = AsyncMock(return_value="Sure! [REMEMBER: likes cats]")
    result = await engine.process_message("alice", "I love cats")
    assert "[REMEMBER:" not in result.response
    assert len(result.memory_actions) == 1
    assert result.memory_actions[0].type == "remember"
    assert result.memory_actions[0].value == "likes cats"


@pytest.mark.asyncio
async def test_process_message_multiple_actions(engine, patched_layers, mock_provider):
    mock_provider.chat = AsyncMock(return_value="Hi! [NAME: Bob] [REMEMBER: loves Go]")
    result = await engine.process_message("bob", "my name is Bob")
    assert len(result.memory_actions) == 2
    types = {a.type for a in result.memory_actions}
    assert types == {"name", "remember"}


@pytest.mark.asyncio
async def test_process_message_persists_to_conversation(engine, patched_layers):
    from agent_memory.layers import conversation
    await engine.process_message("alice", "hello")
    count = conversation.get_message_count("alice")
    assert count == 2  # user + assistant


@pytest.mark.asyncio
async def test_process_message_triggers_summarization(tmp_path, patched_layers, mock_provider):
    """Summarization fires when turn count >= summarize_after_turns and there
    are old turns outside the recent window to compress."""
    config = MemoryConfig(
        token_budget=3000,
        summarize_after_turns=2,
        recent_turns_window=1,   # small window so old turns exist to summarize
        db_path=tmp_path / "engine_test.db",
        chroma_path=tmp_path / "chroma",
    )
    eng = MemoryEngine(config=config, provider=mock_provider)

    # 3 calls → 6 messages; cutoff = max(0, 6 - 1*2) = 4 → non-empty to summarize
    await eng.process_message("alice", "first")
    await eng.process_message("alice", "second")
    await eng.process_message("alice", "third")

    mock_provider.summarize.assert_called()


# ── get_memory_state ────────────────────────────────────────────────────────

def test_get_memory_state_defaults(engine, patched_layers):
    state = engine.get_memory_state("alice")
    assert isinstance(state, MemoryState)
    assert state.user_name == "User"   # _default() in core.py returns "User"
    assert state.facts == []
    assert state.message_count == 0


@pytest.mark.asyncio
async def test_get_memory_state_after_messages(engine, patched_layers, mock_provider):
    mock_provider.chat = AsyncMock(return_value="Hi! [NAME: Alice]")
    await engine.process_message("alice", "hi")
    state = engine.get_memory_state("alice")
    assert state.user_name == "Alice"
    assert state.message_count == 2


# ── reset_user ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_user_clears_messages(engine, patched_layers):
    from agent_memory.layers import conversation
    await engine.process_message("alice", "hello")
    assert conversation.get_message_count("alice") == 2

    engine.reset_user("alice")
    assert conversation.get_message_count("alice") == 0


# ── MemoryAction str ─────────────────────────────────────────────────────────

def test_memory_action_str():
    action = MemoryAction(type="remember", value="likes Python")
    assert str(action) == "[REMEMBER: likes Python]"
