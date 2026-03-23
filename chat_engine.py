"""
chat_engine.py  —  Core Chat Orchestrator

Ties all layers together:
  1. Build context (assemble memory layers)
  2. Call Ollama
  3. Parse memory commands from response
  4. Save messages to conversation store
  5. Archive to vector store
  6. Trigger summarization if needed
"""

import asyncio
from agent_memory.config import MemoryConfig
from agent_memory.layers import conversation, archival, summary
from agent_memory import context_assembler
from ollama_client import chat, summarize
from agent_memory.command_parser import parse_and_apply


async def process_message(
    user_id: str,
    user_message: str,
    stream: bool = True,
    show_stats: bool = False,
    config: MemoryConfig = None,
) -> tuple[str, list]:
    """
    Full pipeline for one user message → response.

    Args:
        user_id:      unique identifier for this user's memory
        user_message: the user's input text
        stream:       whether to stream tokens to stdout
        show_stats:   print token budget stats for debugging
        config:       MemoryConfig instance (defaults to MemoryConfig())

    Returns:
        (cleaned_response, memory_actions)
    """
    config = config or MemoryConfig()

    # ── Step 1: Show context stats (debug mode) ────────────────────────────────
    if show_stats:
        stats = context_assembler.get_context_stats(user_id, user_message, config)
        print(f"\n  [CTX] {stats['total_tokens']}/{stats['budget']} tokens | "
              f"{stats['total_messages']} messages in context\n")

    # ── Step 2: Build context window ───────────────────────────────────────────
    messages = context_assembler.build_context(user_id, user_message, config)

    # ── Step 3: Call LLM ───────────────────────────────────────────────────────
    raw_response = await chat(messages, stream=stream, config=config)

    # ── Step 4: Parse + apply memory commands ──────────────────────────────────
    cleaned_response, memory_actions = parse_and_apply(user_id, raw_response)

    # ── Step 5: Persist to raw conversation store ──────────────────────────────
    msg_id_user = conversation.save_message(user_id, "user",      user_message)
    msg_id_asst = conversation.save_message(user_id, "assistant", cleaned_response)

    # ── Step 6: Archive both to vector store ───────────────────────────────────
    archival.archive_message(user_id, "user",      user_message,     msg_id_user)
    archival.archive_message(user_id, "assistant", cleaned_response, msg_id_asst)

    # ── Step 7: Trigger summarization if needed ────────────────────────────────
    if context_assembler.should_summarize(user_id, config):
        await _run_summarization(user_id, config)

    return cleaned_response, memory_actions


async def _run_summarization(user_id: str, config: MemoryConfig = None):
    """Compress old conversation turns into a rolling summary."""
    config = config or MemoryConfig()
    turns_to_summarize = context_assembler.get_turns_to_summarize(user_id, config)
    if not turns_to_summarize:
        return

    summary_text = await summarize(turns_to_summarize, config)
    total_turns  = conversation.get_message_count(user_id)
    summary.save(user_id, summary_text, total_turns)
