"""
token_counter.py
Simple token estimation — tiktoken uses cl100k (GPT-4 tokenizer).
Ministral uses its own tokenizer but cl100k is ~close enough for budgeting.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate token count for a string."""
    return len(_enc.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a list of {role, content} messages."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", ""))
        total += 4  # role overhead per message
    return total
