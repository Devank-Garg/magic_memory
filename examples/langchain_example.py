"""
langchain_example.py — Using agent_memory as LangChain memory

Two approaches shown:

  1. AgentMemoryHistory  — BaseChatMessageHistory adapter (sliding window only)
     Works today, zero extra code needed in agent_memory.

  2. AgentMemoryFullHistory — enhanced adapter that also injects the
     assembled system prompt (core facts + summary + archival hits).
     Gives you the full 4-layer memory in any LangChain chain.

Install:
    pip install langchain-core langchain-openai   # or langchain-anthropic / langchain-ollama

Run:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_example.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Pick whichever LLM you have installed:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_ollama import ChatOllama

from agent_memory import MemoryConfig
from agent_memory.layers import conversation
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.chroma_store import ChromaStore
from agent_memory import context_assembler


# ─────────────────────────────────────────────────────────────────────────────
# Approach 1: Sliding-window adapter (works today, no changes to agent_memory)
#
# What it does:    stores every message in SQLite via conversation layer;
#                  returns the last N turns as LangChain messages.
# What it misses:  core memory facts, rolling summary, archival search hits
#                  are NOT injected — LangChain builds the system prompt.
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemoryHistory(BaseChatMessageHistory):
    """
    Drop-in LangChain chat history backed by agent_memory's conversation layer.

    Use this when you want persistent, multi-user storage but are happy
    letting LangChain handle the prompt assembly.
    """

    def __init__(self, user_id: str, config: MemoryConfig | None = None):
        self.user_id = user_id
        self._config = config or MemoryConfig()

    @property
    def messages(self) -> list[BaseMessage]:
        """Return recent conversation as LangChain message objects."""
        raw = conversation.get_recent_messages(
            self.user_id, self._config.recent_turns_window * 2
        )
        result: list[BaseMessage] = []
        for m in raw:
            if m["role"] == "user":
                result.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                result.append(AIMessage(content=m["content"]))
        return result

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Persist LangChain messages to the conversation log."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation.save_message(self.user_id, "user", msg.content)
            elif isinstance(msg, AIMessage):
                conversation.save_message(self.user_id, "assistant", msg.content)
            # SystemMessage is not persisted — it's rebuilt each turn

    def clear(self) -> None:
        """Wipe all stored messages for this user."""
        SQLiteStore(self._config.db_path).delete_user(self.user_id)
        ChromaStore(self._config.chroma_path).delete_collection(self.user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Approach 2: Full 4-layer adapter
#
# What it does:    all of Approach 1 PLUS injects the assembled system prompt
#                  (core memory facts + rolling summary + archival hits) as
#                  a SystemMessage prepended to the history.
# How:             overrides messages property to prepend a SystemMessage
#                  built from context_assembler's layer renders.
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemoryFullHistory(AgentMemoryHistory):
    """
    Full agent_memory adapter — injects all 4 memory layers into LangChain.

    Pass the current user message via the `current_input` kwarg when
    constructing so archival search can find semantically relevant hits.
    """

    def __init__(
        self,
        user_id: str,
        current_input: str = "",
        config: MemoryConfig | None = None,
    ):
        super().__init__(user_id, config)
        self._current_input = current_input

    @property
    def messages(self) -> list[BaseMessage]:
        """Return [SystemMessage(all layers)] + recent chat history."""
        from agent_memory.layers import core, summary, archival

        # Build the same system prompt that MemoryEngine would build
        core_block     = core.render_for_prompt(self.user_id)
        summary_block  = summary.render_for_prompt(self.user_id)
        archival_block = archival.render_for_prompt(self.user_id, self._current_input)

        parts = [core_block]
        if summary_block:
            parts.append(summary_block)
        if archival_block:
            parts.append(archival_block)
        parts.append(
            "## MEMORY COMMANDS\n"
            "You can update memory: [REMEMBER: fact]  [NOTE: text]  [NAME: name]"
        )
        system_prompt = "\n\n".join(parts)

        return [SystemMessage(content=system_prompt)] + super().messages


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────

def build_chain_approach1(llm):
    """Approach 1: simple persistent history, LangChain owns the system prompt."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with an excellent memory."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history=lambda session_id: AgentMemoryHistory(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )


def build_chain_approach2(llm):
    """
    Approach 2: agent_memory owns the system prompt (all 4 layers injected).

    The history placeholder will receive [SystemMessage + chat turns],
    so we do NOT add a separate system message in the prompt template.
    """

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),   # includes SystemMessage from our adapter
        ("human", "{input}"),
    ])

    chain = prompt | llm

    def get_full_history(session_id: str, **kwargs) -> AgentMemoryFullHistory:
        # current_input is passed via configurable so archival search is accurate
        current_input = kwargs.get("current_input", "")
        return AgentMemoryFullHistory(session_id, current_input=current_input)

    return RunnableWithMessageHistory(
        chain,
        get_session_history=lambda session_id: get_full_history(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI   # swap for ChatOllama / ChatAnthropic

    llm = ChatOpenAI(model="gpt-4o")

    # --- Approach 1 ---
    chain1 = build_chain_approach1(llm)
    config = {"configurable": {"session_id": "alice"}}

    resp = chain1.invoke({"input": "Hi! My name is Alice."}, config=config)
    print("Approach 1:", resp.content)

    resp = chain1.invoke({"input": "What's my name?"}, config=config)
    print("Approach 1:", resp.content)   # remembers "Alice" from sliding window

    # --- Approach 2 (full memory layers) ---
    chain2 = build_chain_approach2(llm)

    resp = chain2.invoke({"input": "I love Python and hate JavaScript."}, config=config)
    print("Approach 2:", resp.content)

    resp = chain2.invoke({"input": "What do you know about my preferences?"}, config=config)
    print("Approach 2:", resp.content)   # uses core memory facts + archival hits
