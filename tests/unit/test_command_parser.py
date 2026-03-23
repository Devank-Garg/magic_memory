"""Unit tests for command_parser.py"""

import pytest
from unittest.mock import patch
from agent_memory.command_parser import parse_and_apply


def _parse(text, user_id="test_user"):
    """Helper: patch core layer so tests don't touch SQLite."""
    with patch("agent_memory.command_parser.core") as mock_core:
        cleaned, actions = parse_and_apply(user_id, text)
    return cleaned, actions, mock_core


def test_no_commands_passthrough():
    cleaned, actions, _ = _parse("Hello, how are you?")
    assert cleaned == "Hello, how are you?"
    assert actions == []


def test_remember_command_parsed():
    cleaned, actions, mock_core = _parse(
        "Sure! [REMEMBER: likes dark mode]"
    )
    mock_core.update_fact.assert_called_once_with("test_user", "likes dark mode")
    assert "likes dark mode" not in cleaned
    assert len(actions) == 1


def test_note_command_parsed():
    cleaned, actions, mock_core = _parse(
        "Got it. [NOTE: currently debugging a RAG pipeline]"
    )
    mock_core.update_scratch.assert_called_once_with(
        "test_user", "currently debugging a RAG pipeline"
    )
    assert len(actions) == 1


def test_name_command_parsed():
    cleaned, actions, mock_core = _parse("[NAME: Devank]")
    mock_core.set_user_name.assert_called_once_with("test_user", "Devank")
    assert len(actions) == 1


def test_case_insensitive():
    _, actions, mock_core = _parse("[remember: eats pizza]")
    mock_core.update_fact.assert_called_once()


def test_command_stripped_from_response():
    cleaned, _, _ = _parse("Nice to meet you! [NAME: Alice] Hope you enjoy this.")
    assert "[NAME:" not in cleaned
    assert "Nice to meet you!" in cleaned


def test_multiple_commands_in_one_response():
    _, actions, _ = _parse("[REMEMBER: loves Go] [NAME: Bob]")
    assert len(actions) == 2


def test_malformed_tag_not_matched():
    cleaned, actions, _ = _parse("[REMEMBER]  [REMEMBER:]  plain text")
    assert actions == []
    assert cleaned.strip() != ""


def test_markdown_bold_command_stripped():
    """**[NAME: Alice]** and **[REMEMBER: x]** should be fully removed."""
    cleaned, actions, mock_core = _parse("Hello! **[NAME: Alice]** Hope you enjoy.")
    assert "**" not in cleaned
    assert "[NAME:" not in cleaned
    assert len(actions) == 1
    mock_core.set_user_name.assert_called_once_with("test_user", "Alice")


def test_markdown_italic_command_stripped():
    cleaned, actions, _ = _parse("Sure! *[REMEMBER: likes cats]*")
    assert "*" not in cleaned
    assert "[REMEMBER:" not in cleaned
    assert len(actions) == 1
