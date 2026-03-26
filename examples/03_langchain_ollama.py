"""
03_langchain_ollama.py — AgentMemoryFullHistory with LangChain + Ollama
========================================================================

Plugs agent_memory into any LangChain LCEL chain via BaseChatMessageHistory.
The full 4-layer memory (core facts, rolling summary, archival hits, behaviour)
is injected as a SystemMessage automatically each turn.

Requirements
------------
    pip install "agent-memory[langchain]"
    pip install langchain-ollama
    ollama pull mistral          # or any model you prefer
    ollama serve                 # must be running

Run
---
    python examples/03_langchain_ollama.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage

from agent_memory import MemoryConfig
from agent_memory.integrations.langchain import AgentMemoryFullHistory


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="mistral")   # swap model name freely

# ── Prompt ────────────────────────────────────────────────────────────────────
# MessagesPlaceholder receives [SystemMessage(memory layers)] + recent chat turns.
# Do NOT add a separate ("system", "...") here — AgentMemoryFullHistory owns it.
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# ── Chain ─────────────────────────────────────────────────────────────────────
chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=lambda sid: AgentMemoryFullHistory(
        user_id=sid,
        config=MemoryConfig(model="mistral"),  # optional — defaults work fine
    ),
    input_messages_key="input",
    history_messages_key="history",
)


def chat(user_id: str, message: str) -> str:
    """Send one message and return the plain text reply."""
    resp: AIMessage = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": user_id}},
    )

    # ── Output parsing ────────────────────────────────────────────────────────
    # resp.content              — the text reply  (always use this)
    # resp.response_metadata    — model name, timing, token counts
    # resp.usage_metadata       — {"input_tokens": N, "output_tokens": N, "total_tokens": N}
    # resp.id                   — LangChain run ID
    return resp.content


def chat_verbose(user_id: str, message: str) -> None:
    """Send one message and print full output breakdown."""
    resp: AIMessage = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": user_id}},
    )
    print(f"Reply   : {resp.content}")
    print(f"Model   : {resp.response_metadata.get('model', 'unknown')}")
    tokens = resp.usage_metadata or {}
    print(f"Tokens  : {tokens.get('input_tokens', '?')} in / {tokens.get('output_tokens', '?')} out")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user = "alice"

    turns = [
        "Hi! My name is Alice.",
        "I love Python and work as a data engineer.",
        "I prefer dark mode and mechanical keyboards.",
        "What do you know about me?",    # recall test
    ]

    for msg in turns:
        print(f"\nUser : {msg}")
        reply = chat(user, msg)
        print(f"Agent: {reply}")

    # Verbose output example — shows token usage and model metadata
    print("\n── Verbose turn ─────────────────────────────")
    chat_verbose(user, "Summarise everything you know about me.")

    # ── Inspect / reset memory (no LangChain needed) ──────────────────────────
    from agent_memory import MemoryEngine
    from agent_memory.providers import OllamaProvider

    engine = MemoryEngine(config=MemoryConfig(), provider=OllamaProvider())
    state = engine.get_memory_state(user)
    print(f"\n── Memory State ──────────────────────────────")
    print(f"Name   : {state.user_name or '(not set)'}")
    print(f"Facts  : {state.facts}")
    print(f"Turns  : {state.message_count}")

    # engine.reset_user(user)  # wipe all memory for this user
