"""Unit tests for providers/registry.py"""

import pytest
from agent_memory.providers.registry import create_provider
from agent_memory.providers.ollama import OllamaProvider
from agent_memory.providers.openai import OpenAIProvider
from agent_memory.providers.anthropic import AnthropicProvider
from agent_memory.config import MemoryConfig


@pytest.fixture
def config():
    return MemoryConfig()


def test_create_ollama_provider(config):
    provider = create_provider("ollama", config)
    assert isinstance(provider, OllamaProvider)


def test_create_ollama_case_insensitive(config):
    provider = create_provider("Ollama", config)
    assert isinstance(provider, OllamaProvider)


def test_create_openai_provider(config):
    pytest.importorskip("openai", reason="openai package not installed")
    provider = create_provider("openai", config, api_key="sk-test")
    assert isinstance(provider, OpenAIProvider)


def test_create_anthropic_provider(config):
    pytest.importorskip("anthropic", reason="anthropic package not installed")
    provider = create_provider("anthropic", config, api_key="ant-test")
    assert isinstance(provider, AnthropicProvider)


def test_unknown_provider_raises_value_error(config):
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("mistral", config)


def test_create_provider_without_config_uses_default():
    provider = create_provider("ollama")
    assert isinstance(provider, OllamaProvider)
