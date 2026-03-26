"""
02_standalone_openai.py — MemoryEngine with OpenAI, no LangChain
=================================================================

Uses agent_memory directly. No LangChain dependency needed.

Requirements
------------
    pip install "agent-memory[cli,openai]"
    export OPENAI_API_KEY=sk-...   # or put it in a .env file

Run
---
    python examples/02_standalone_openai.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env if present

from agent_memory import MemoryEngine, MemoryConfig, MemoryResponse
from agent_memory.providers.openai import OpenAIProvider


async def main() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────────
    config = MemoryConfig(
        model="gpt-4o-mini",
        recent_turns_window=10,
        summarize_after_turns=15,
    )
    engine = MemoryEngine(
        config=config,
        provider=OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            config=config,
        ),
    )
    user_id = "bob"

    # ── Single turn ────────────────────────────────────────────────────────────
    result: MemoryResponse = await engine.process_message(user_id, "Hi! I'm Bob, a backend engineer.")

    # result.response       — plain text reply, memory commands already stripped
    # result.memory_actions — what the LLM stored in memory this turn
    print("Response:", result.response)

    if result.memory_actions:
        print("Memory actions this turn:")
        for action in result.memory_actions:
            # action.type  — "remember" | "note" | "name"
            # action.value — the text that was stored
            print(f"  [{action.type.upper()}] {action.value}")

    # ── Multi-turn conversation ────────────────────────────────────────────────
    messages = [
        "I mostly work with Python and Go.",
        "I prefer PostgreSQL over MongoDB.",
        "What stack do I use?",          # tests recall
        "What do you know about me?",    # tests full memory recall
    ]

    for msg in messages:
        print(f"\nUser : {msg}")
        result = await engine.process_message(user_id, msg)
        print(f"Agent: {result.response}")
        if result.memory_actions:
            for action in result.memory_actions:
                print(f"  → stored [{action.type}]: {action.value}")

    # ── Inspect memory state ───────────────────────────────────────────────────
    print("\n── Memory State ──────────────────────────────")
    state = engine.get_memory_state(user_id)
    print(f"Name         : {state.user_name or '(not set)'}")
    print(f"Facts        : {state.facts}")
    print(f"Summary      : {state.summary or '(none yet)'}")
    print(f"Total turns  : {state.message_count}")

    # ── Reset (wipes all memory for this user) ─────────────────────────────────
    # engine.reset_user(user_id)
    # print("Memory wiped.")


if __name__ == "__main__":
    asyncio.run(main())
