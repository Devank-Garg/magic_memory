"""
04_langchain_openai.py — AgentMemoryFullHistory with LangChain + OpenAI
========================================================================

Plugs agent_memory into any LangChain LCEL chain via BaseChatMessageHistory.
The full 4-layer memory (core facts, rolling summary, archival hits, behaviour)
is injected as a SystemMessage automatically each turn.

Requirements
------------
    pip install "agent-memory[langchain,openai]"
    pip install langchain-openai
    export OPENAI_API_KEY=sk-...   # or put it in a .env file

Run
---
    python examples/04_langchain_openai.py
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env if present

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage

from agent_memory import MemoryConfig
from agent_memory.integrations.langchain import AgentMemoryFullHistory


# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

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
        config=MemoryConfig(model="gpt-4o-mini"),  # optional — defaults work fine
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
    # resp.response_metadata    — finish_reason, model, system_fingerprint
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
    print(f"Model   : {resp.response_metadata.get('model_name', 'unknown')}")
    print(f"Finish  : {resp.response_metadata.get('finish_reason', 'unknown')}")
    tokens = resp.usage_metadata or {}
    print(f"Tokens  : {tokens.get('input_tokens', '?')} in / {tokens.get('output_tokens', '?')} out")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user = "carol"

    turns = [
        "Hi! I'm Carol, a machine learning engineer.",
        "I mainly use PyTorch and dislike TensorFlow.",
        "My favourite editor is Neovim.",
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
    from agent_memory.providers.openai import OpenAIProvider

    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
    )
    state = engine.get_memory_state(user)
    print(f"\n── Memory State ──────────────────────────────")
    print(f"Name   : {state.user_name or '(not set)'}")
    print(f"Facts  : {state.facts}")
    print(f"Turns  : {state.message_count}")

    # engine.reset_user(user)  # wipe all memory for this user
