"""
01_standalone_ollama.py — MemoryEngine with Ollama, no LangChain
================================================================

Uses agent_memory directly. No LangChain dependency needed.

Requirements
------------
    pip install "agent-memory[cli]"
    ollama pull mistral          # or any model you prefer
    ollama serve                 # must be running

Run
---
    python examples/01_standalone_ollama.py
"""

import asyncio
from agent_memory import MemoryEngine, MemoryConfig, MemoryResponse
from agent_memory.providers import OllamaProvider


async def main() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────────
    config = MemoryConfig(
        model="mistral",          # any model pulled in ollama
        recent_turns_window=10,   # last 10 turns kept verbatim in context
        summarize_after_turns=15, # compress older turns into a rolling summary
    )
    engine = MemoryEngine(config=config, provider=OllamaProvider(config=config))
    user_id = "alice"

    # ── Single turn ────────────────────────────────────────────────────────────
    result: MemoryResponse = await engine.process_message(user_id, "Hi! My name is Alice.")

    # result.response       — plain text reply, memory commands already stripped
    # result.memory_actions — list of MemoryAction written this turn
    print("Response:", result.response)

    if result.memory_actions:
        print("Memory actions this turn:")
        for action in result.memory_actions:
            # action.type  — "remember" | "note" | "name"
            # action.value — the text that was stored
            print(f"  [{action.type.upper()}] {action.value}")

    # ── Multi-turn conversation ────────────────────────────────────────────────
    messages = [
        "I love Python and hate JavaScript.",
        "What languages do I like?",          # tests recall of fact above
        "I work as a data engineer.",
        "What do you know about me?",          # tests full memory recall
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
