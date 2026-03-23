"""
engine.py  —  MemoryEngine: the single public entry point for library consumers

Usage example::

    from agent_memory import MemoryEngine, MemoryConfig
    from agent_memory.providers import OllamaProvider

    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=OllamaProvider(),
    )

    result = await engine.process_message("alice", "hello!")
    print(result.response)          # LLM reply, commands stripped
    print(result.memory_actions)    # list[MemoryAction]

    state = engine.get_memory_state("alice")
    print(state.facts)
"""

from __future__ import annotations

from agent_memory.config import MemoryConfig
from agent_memory.providers.base import BaseLLMProvider, LLMOptions
from agent_memory.providers.ollama import OllamaProvider
from agent_memory.types import MemoryAction, MemoryResponse, MemoryState
from agent_memory import context_assembler
from agent_memory.layers import conversation, archival, summary, core
from agent_memory.command_parser import parse_and_apply
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.chroma_store import ChromaStore


class MemoryEngine:
    """
    Orchestrates all memory layers for a given config + provider.

    Parameters
    ----------
    config   : MemoryConfig — all tuning knobs in one place
    provider : BaseLLMProvider — LLM backend (defaults to OllamaProvider)
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self._config = config or MemoryConfig()
        self._provider = provider or OllamaProvider(config=self._config)

    # ── public API ──────────────────────────────────────────────────────────

    async def process_message(
        self,
        user_id: str,
        user_message: str,
        stream: bool = False,
    ) -> MemoryResponse:
        """
        Full pipeline for one user message → response.

        1. Assemble context window from all memory layers
        2. Call LLM provider
        3. Parse + apply memory commands
        4. Persist messages to conversation log
        5. Archive to vector store
        6. Trigger rolling summarisation if threshold reached

        Returns
        -------
        MemoryResponse with the cleaned reply and any MemoryActions applied.
        """
        # 1. Build context
        messages = context_assembler.build_context(
            user_id, user_message, self._config
        )

        # 2. Call LLM
        raw_response = await self._provider.chat(
            messages, LLMOptions(stream=stream)
        )

        # 3. Parse memory commands
        cleaned_response, memory_actions = parse_and_apply(user_id, raw_response)

        # 4. Persist
        msg_id_user = conversation.save_message(user_id, "user",      user_message)
        msg_id_asst = conversation.save_message(user_id, "assistant", cleaned_response)

        # 5. Archive
        archival.archive_message(user_id, "user",      user_message,     msg_id_user)
        archival.archive_message(user_id, "assistant", cleaned_response, msg_id_asst)

        # 6. Summarise if needed
        if context_assembler.should_summarize(user_id, self._config):
            await self._run_summarization(user_id)

        return MemoryResponse(
            response=cleaned_response,
            memory_actions=memory_actions,
        )

    def get_memory_state(self, user_id: str) -> MemoryState:
        """Return a snapshot of all memory layers for a user."""
        core_data    = core.load(user_id)
        summary_data = summary.load(user_id)
        msg_count    = conversation.get_message_count(user_id)

        return MemoryState(
            user_name=core_data["user_name"],
            facts=core_data["user_facts"],
            scratch=core_data["scratch"],
            summary=summary_data["summary"],
            summary_turn=summary_data["turn_count"],
            message_count=msg_count,
        )

    def reset_user(self, user_id: str) -> None:
        """Wipe all memory (SQLite + ChromaDB) for a user."""
        SQLiteStore(self._config.db_path).delete_user(user_id)
        ChromaStore(self._config.chroma_path).delete_collection(user_id)

    # ── internal ────────────────────────────────────────────────────────────

    async def _run_summarization(self, user_id: str) -> None:
        turns = context_assembler.get_turns_to_summarize(user_id, self._config)
        if not turns:
            return
        summary_text = await self._provider.summarize(turns)
        total_turns  = conversation.get_message_count(user_id)
        summary.save(user_id, summary_text, total_turns)
