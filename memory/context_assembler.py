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

Total budget: configurable, default 3000 tokens (leaving ~1000 for response)
"""

from memory.token_counter import count_tokens, count_messages_tokens
from memory import core_memory, summary_memory, archival_memory, conversation_store


# ── Configuration ──────────────────────────────────────────────────────────────
TOTAL_CONTEXT_BUDGET  = 3000   # tokens sent to model (input side)
RESPONSE_RESERVE      = 1000   # tokens reserved for model response
RECENT_TURNS_DEFAULT  = 10     # number of recent turns to include verbatim
SUMMARIZE_AFTER_TURNS = 15     # trigger summarization after N total turns


def build_context(user_id: str, current_user_message: str) -> list[dict]:
    """
    Build the final messages list for the LLM call.
    
    Returns: list of {role, content} ready to send to Ollama
    """
    budget = TOTAL_CONTEXT_BUDGET
    messages = []

    # ── 1. SYSTEM PROMPT (Core Memory) — always present ───────────────────────
    core_block    = core_memory.render_for_prompt(user_id)
    summary_block = summary_memory.render_for_prompt(user_id)
    archival_block = archival_memory.render_for_prompt(user_id, current_user_message)

    system_parts = [
        core_block,
        "\n" + summary_block if summary_block else "",
        "\n" + archival_block if archival_block else "",
        """
## MEMORY COMMANDS
You can update your memory by including ONE of these at the END of your response (on its own line):
  [REMEMBER: <fact about the user>]     — store a new user fact in core memory
  [NOTE: <working note to yourself>]    — update your scratch pad
  [NAME: <user's name>]                 — store the user's name
These commands will be parsed and executed after your response. Use them sparingly.
"""
    ]

    system_prompt = "\n".join(p for p in system_parts if p)
    system_tokens = count_tokens(system_prompt)
    budget -= system_tokens

    messages.append({"role": "system", "content": system_prompt})

    # ── 2. RECENT MESSAGES (Sliding Window) ────────────────────────────────────
    # Get recent turns, then trim to fit budget
    recent = conversation_store.get_recent_messages(user_id, RECENT_TURNS_DEFAULT * 2)
    
    # Build recent window that fits in budget (leave room for current message)
    current_msg_tokens = count_tokens(current_user_message) + 10
    available_for_history = budget - current_msg_tokens - 100  # 100 token safety margin
    
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


def should_summarize(user_id: str) -> bool:
    """Returns True if we should trigger a summarization pass."""
    total_turns = conversation_store.get_message_count(user_id)
    existing_summary = summary_memory.load(user_id)
    turns_since_summary = total_turns - existing_summary["turn_count"]
    return turns_since_summary >= SUMMARIZE_AFTER_TURNS


def get_turns_to_summarize(user_id: str) -> list[dict]:
    """Get the old turns that should be summarized (everything except recent window)."""
    all_msgs = conversation_store.get_all_messages(user_id)
    # Keep last RECENT_TURNS_DEFAULT*2 verbatim, summarize everything older
    cutoff = max(0, len(all_msgs) - RECENT_TURNS_DEFAULT * 2)
    return all_msgs[:cutoff]


def get_context_stats(user_id: str, current_message: str) -> dict:
    """Debug helper — returns token breakdown of current context."""
    msgs = build_context(user_id, current_message)
    return {
        "total_messages": len(msgs),
        "total_tokens": count_messages_tokens(msgs),
        "budget": TOTAL_CONTEXT_BUDGET,
        "breakdown": {m["role"]: count_tokens(m["content"]) for m in msgs}
    }
