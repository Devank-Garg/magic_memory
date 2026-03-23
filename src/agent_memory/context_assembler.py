"""
context_assembler.py  —  The Context Budget Manager

This is the core of "infinite context". It assembles the final message list
to send to the LLM, respecting a hard token budget.

Priority order (highest to lowest):
  1. System prompt (core memory + persona)           ~300 tokens  [never dropped]
  2. Retrieved archival memories                     ~400 tokens  [dropped if tight]
  3. Conversation summary (older turns)              ~250 tokens  [dropped if tight]
  4. Recent messages (sliding window, last N turns)  ~800 tokens  [trimmed, not dropped]
  5. Current user message                            ~100 tokens  [never dropped]

Total budget: configurable via MemoryConfig (default 3000 tokens)
"""

import logging

from agent_memory.config import MemoryConfig
from agent_memory.token_counter import count_tokens, count_messages_tokens
from agent_memory.layers import core, summary, archival, conversation

logger = logging.getLogger(__name__)


def build_context(user_id: str, current_user_message: str, config: MemoryConfig = None) -> list[dict]:
    """
    Build the final messages list for the LLM call.

    Returns: list of {role, content} ready to send to Ollama
    """
    config = config or MemoryConfig()
    budget = config.token_budget
    messages = []

    # ── 1. SYSTEM PROMPT (Core Memory) — always present ───────────────────────
    core_block     = core.render_for_prompt(user_id)
    summary_block  = summary.render_for_prompt(user_id)
    archival_block = archival.render_for_prompt(user_id, current_user_message)

    system_parts = [
        core_block,
        "\n" + summary_block if summary_block else "",
        "\n" + archival_block if archival_block else "",
        """
## BEHAVIOUR
- Respond naturally and concisely to what the user actually said.
- Your memory context is background knowledge — do NOT recap, list, or discuss it
  unless the user explicitly asks (e.g. "what do you know about me?").
- Keep responses short unless the user asks for detail.

## MEMORY COMMANDS
You may append ONE command at the very end of your response (never mid-response):
  [REMEMBER: <fact>]   — only for facts the user explicitly told you this turn
  [NOTE: <text>]       — update your working notes
  [NAME: <name>]       — only when the user directly tells you their name

NEVER invent facts. NEVER use these as answers. NEVER wrap them in markdown formatting.
"""
    ]

    system_prompt = "\n".join(p for p in system_parts if p)
    system_tokens = count_tokens(system_prompt)
    budget -= system_tokens

    # Issue 2 fix: guard against negative budget if system prompt alone exceeds limit
    if budget <= 0:
        logger.warning(
            "System prompt (%d tokens) exceeds token_budget (%d). "
            "No history will be included. Consider raising token_budget.",
            system_tokens, config.token_budget,
        )

    messages.append({"role": "system", "content": system_prompt})

    # ── 2. RECENT MESSAGES (Sliding Window) ────────────────────────────────────
    recent = conversation.get_recent_messages(user_id, config.recent_turns_window * 2)

    current_msg_tokens = count_tokens(current_user_message) + 10
    # Issue 2 fix: clamp so available_for_history is never negative
    available_for_history = max(0, budget - current_msg_tokens - 100)  # 100 token safety margin

    history_window = []
    token_used = 0
    for msg in reversed(recent):  # newest first
        t = count_tokens(msg["content"]) + 4
        if token_used + t > available_for_history:
            break
        history_window.insert(0, msg)
        token_used += t

    messages.extend(history_window)
    budget -= token_used

    # ── 3. Current user message ────────────────────────────────────────────────
    messages.append({"role": "user", "content": current_user_message})

    return messages


def should_summarize(user_id: str, config: MemoryConfig = None) -> bool:
    """Returns True if we should trigger a summarization pass."""
    config = config or MemoryConfig()
    total_turns = conversation.get_message_count(user_id)
    existing_summary = summary.load(user_id)
    turns_since_summary = total_turns - existing_summary["turn_count"]
    return turns_since_summary >= config.summarize_after_turns


def get_turns_to_summarize(user_id: str, config: MemoryConfig = None) -> list[dict]:
    """Get the old turns that should be summarized (everything except recent window)."""
    config = config or MemoryConfig()
    all_msgs = conversation.get_all_messages(user_id)
    cutoff = max(0, len(all_msgs) - config.recent_turns_window * 2)
    return all_msgs[:cutoff]


def get_context_stats(user_id: str, current_message: str, config: MemoryConfig = None) -> dict:
    """Debug helper — returns token breakdown of current context."""
    config = config or MemoryConfig()
    msgs = build_context(user_id, current_message, config)
    return {
        "total_messages": len(msgs),
        "total_tokens": count_messages_tokens(msgs),
        "budget": config.token_budget,
        "breakdown": {m["role"]: count_tokens(m["content"]) for m in msgs}
    }
