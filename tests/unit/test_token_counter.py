"""Unit tests for token_counter.py"""

from agent_memory.token_counter import count_tokens, count_messages_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0


def test_count_messages_tokens_empty_list():
    assert count_messages_tokens([]) == 0


def test_count_messages_tokens_sums_content_and_overhead():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    total = count_messages_tokens(msgs)
    # each message adds 4 overhead tokens
    expected = count_tokens("hello") + 4 + count_tokens("hi there") + 4
    assert total == expected


def test_count_messages_tokens_missing_content():
    """Messages without 'content' key should not raise."""
    msgs = [{"role": "system"}]
    total = count_messages_tokens(msgs)
    assert total == 4  # just the overhead
