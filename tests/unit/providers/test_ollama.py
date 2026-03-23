"""Unit tests for providers/ollama.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_memory.providers.ollama import OllamaProvider
from agent_memory.providers.base import LLMOptions
from agent_memory.config import MemoryConfig


@pytest.fixture
def config():
    return MemoryConfig(ollama_base="http://localhost:11434", model="test-model")


@pytest.fixture
def provider(config):
    return OllamaProvider(config=config)


# ── chat (non-streaming) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_non_streaming_returns_content(provider):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Hello!"}}

    provider._client = AsyncMock()
    provider._client.post = AsyncMock(return_value=mock_resp)

    result = await provider.chat([{"role": "user", "content": "hi"}], LLMOptions(stream=False))
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_chat_uses_shared_client(provider):
    """The same _client instance should be used across calls (Issue 8 fix)."""
    first_client = provider._client
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}}

    provider._client = AsyncMock()
    provider._client.post = AsyncMock(return_value=mock_resp)

    await provider.chat([{"role": "user", "content": "a"}], LLMOptions(stream=False))
    await provider.chat([{"role": "user", "content": "b"}], LLMOptions(stream=False))

    # Both calls hit the same client object (post called twice)
    assert provider._client.post.call_count == 2


# ── health_check ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check_true_when_model_present(provider):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "test-model:latest"}]}

    provider._client = AsyncMock()
    provider._client.get = AsyncMock(return_value=mock_resp)

    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_false_on_exception(provider):
    provider._client = AsyncMock()
    provider._client.get = AsyncMock(side_effect=Exception("connection refused"))

    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_health_check_false_when_model_absent(provider):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "other-model:latest"}]}

    provider._client = AsyncMock()
    provider._client.get = AsyncMock(return_value=mock_resp)

    assert await provider.health_check() is False


# ── summarize ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summarize_returns_content(provider):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Summary text."}}

    provider._client = AsyncMock()
    provider._client.post = AsyncMock(return_value=mock_resp)

    messages = [{"role": "user", "content": "tell me about Python"}]
    result = await provider.summarize(messages)
    assert result == "Summary text."
