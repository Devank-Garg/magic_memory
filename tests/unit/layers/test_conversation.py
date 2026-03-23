"""Unit tests for layers/conversation.py"""

from agent_memory.layers import conversation


def test_save_message_returns_incrementing_ids(patch_conversation):
    id1 = conversation.save_message("alice", "user", "hello")
    id2 = conversation.save_message("alice", "assistant", "hi there")
    assert id1 == 1
    assert id2 == 2


def test_get_message_count(patch_conversation):
    assert conversation.get_message_count("alice") == 0
    conversation.save_message("alice", "user", "hello")
    conversation.save_message("alice", "user", "world")
    assert conversation.get_message_count("alice") == 2


def test_get_recent_messages_returns_chronological_order(patch_conversation):
    conversation.save_message("alice", "user", "first")
    conversation.save_message("alice", "assistant", "second")
    conversation.save_message("alice", "user", "third")

    msgs = conversation.get_recent_messages("alice", 3)
    assert [m["content"] for m in msgs] == ["first", "second", "third"]


def test_get_recent_messages_limits_correctly(patch_conversation):
    for i in range(5):
        conversation.save_message("alice", "user", f"msg {i}")

    msgs = conversation.get_recent_messages("alice", 3)
    assert len(msgs) == 3
    # Should be the last 3, in ascending order
    assert msgs[0]["content"] == "msg 2"
    assert msgs[-1]["content"] == "msg 4"


def test_get_all_messages_ascending(patch_conversation):
    conversation.save_message("alice", "user", "a")
    conversation.save_message("alice", "assistant", "b")
    msgs = conversation.get_all_messages("alice")
    assert msgs[0]["content"] == "a"
    assert msgs[1]["content"] == "b"


def test_separate_users_are_isolated(patch_conversation):
    conversation.save_message("alice", "user", "alice msg")
    conversation.save_message("bob", "user", "bob msg")
    assert conversation.get_message_count("alice") == 1
    assert conversation.get_message_count("bob") == 1
    assert conversation.get_recent_messages("alice", 1)[0]["content"] == "alice msg"
