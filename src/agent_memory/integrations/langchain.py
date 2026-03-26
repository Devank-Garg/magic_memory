"""
agent_memory.integrations.langchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LangChain adapter that gives any LCEL chain persistent, multi-layer memory.

Install
-------
    pip install "agent-memory[langchain]"

Single class: AgentMemoryFullHistory
    A ``BaseChatMessageHistory`` that stores every turn in SQLite and returns
    the full 4-layer memory context (core facts, rolling summary, archival hits,
    behaviour instructions) as a ``SystemMessage`` prepended to the history.

Quick start
-----------
::

    from langchain_openai import ChatOpenAI          # or ChatOllama / ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from agent_memory.integrations.langchain import AgentMemoryFullHistory

    llm    = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),   # SystemMessage + chat turns injected here
        ("human", "{input}"),
    ])
    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=lambda sid: AgentMemoryFullHistory(sid),
        input_messages_key="input",
        history_messages_key="history",
    )
    resp = chain.invoke(
        {"input": "I love Python."},
        config={"configurable": {"session_id": "alice"}},
    )
    print(resp.content)

Custom config
-------------
::

    from agent_memory import MemoryConfig
    from agent_memory.integrations.langchain import AgentMemoryFullHistory

    history = AgentMemoryFullHistory(
        "alice",
        current_input="What do I like?",
        config=MemoryConfig(db_path="/tmp/myapp.db", recent_turns_window=5),
    )

Swap LLMs
---------
The adapter is LLM-agnostic — pass any LangChain chat model::

    from langchain_ollama import ChatOllama          # local, free
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
"""

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required for the agent_memory LangChain integration.\n"
        "Install it with:  pip install \"agent-memory[langchain]\""
    ) from exc

from agent_memory.config import MemoryConfig
from agent_memory.context_assembler import build_system_prompt
from agent_memory.layers import conversation
from agent_memory.storage.chroma_store import ChromaStore
from agent_memory.storage.sqlite_store import SQLiteStore


class AgentMemoryFullHistory(BaseChatMessageHistory):
    """
    Full 4-layer LangChain chat history for agent_memory.

    Every human/AI turn is persisted to SQLite.  On each read, the
    ``messages`` property returns:

    1. A ``SystemMessage`` assembled from all memory layers:
       - core facts (always present)
       - rolling summary of older turns
       - semantically relevant archival hits
       - behaviour instructions
    2. The last ``recent_turns_window`` turns verbatim as
       ``HumanMessage`` / ``AIMessage`` objects.

    Parameters
    ----------
    user_id:
        Unique identifier for this user's memory.  Separate users get
        completely isolated SQLite tables and ChromaDB collections.
    current_input:
        The user's current message text.  Pass this so the archival layer
        can run semantic search and surface relevant past context.
        If omitted, archival hits will not be retrieved for this turn.
    config:
        Optional ``MemoryConfig``.  Defaults to ``MemoryConfig()`` which
        stores data under ``./data/``.  Override to change storage paths,
        token budget, window sizes, or any other setting.

    Example
    -------
    ::

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from agent_memory.integrations.langchain import AgentMemoryFullHistory

        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]) | ChatOpenAI(model="gpt-4o-mini"),
            get_session_history=lambda sid: AgentMemoryFullHistory(sid),
            input_messages_key="input",
            history_messages_key="history",
        )
        resp = chain.invoke(
            {"input": "What do you know about me?"},
            config={"configurable": {"session_id": "alice"}},
        )
        print(resp.content)
    """

    def __init__(
        self,
        user_id: str,
        current_input: str = "",
        config: MemoryConfig | None = None,
    ) -> None:
        self.user_id = user_id
        self._current_input = current_input
        self._config = config or MemoryConfig()

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Return [SystemMessage(all memory layers)] + recent chat turns.

        The SystemMessage is rebuilt on every call so it always reflects
        the latest core facts, summary, and archival hits.
        """
        system_prompt = build_system_prompt(
            self.user_id, self._current_input, self._config
        )

        raw = conversation.get_recent_messages(
            self.user_id, self._config.recent_turns_window * 2
        )
        history: list[BaseMessage] = []
        for m in raw:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                history.append(AIMessage(content=m["content"]))

        return [SystemMessage(content=system_prompt)] + history

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Persist LangChain messages to the SQLite conversation log."""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation.save_message(self.user_id, "user", msg.content)
            elif isinstance(msg, AIMessage):
                conversation.save_message(self.user_id, "assistant", msg.content)
            # SystemMessage is not persisted — it is rebuilt from memory layers each turn

    def clear(self) -> None:
        """Wipe all stored messages and vectors for this user."""
        SQLiteStore(self._config.db_path).delete_user(self.user_id)
        ChromaStore(self._config.chroma_path).delete_collection(self.user_id)
