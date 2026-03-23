# Change Log

Continuous record of changes made during the `magic_memory` library refactor.
Each entry maps to a phase and a specific task.

---

## Phase 1 — Restructure into `src/agent_memory/` package layout
*Completed. No logic changes.*

| Task | Change |
|---|---|
| Directory structure | Created `src/agent_memory/`, `src/agent_memory/layers/`, `src/agent_memory/providers/`, `cli/` |
| Move layer files | `memory/conversation_store.py` → `src/agent_memory/layers/conversation.py` |
| | `memory/core_memory.py` → `src/agent_memory/layers/core.py` |
| | `memory/summary_memory.py` → `src/agent_memory/layers/summary.py` |
| | `memory/archival_memory.py` → `src/agent_memory/layers/archival.py` |
| | `memory/token_counter.py` → `src/agent_memory/token_counter.py` |
| Move logic files | `command_parser.py` → `src/agent_memory/command_parser.py` |
| | `memory/context_assembler.py` → `src/agent_memory/context_assembler.py` |
| DB_PATH fix | Updated `Path(__file__)` depth in all layer files to resolve correctly from new location (`parents[3]` instead of `parent.parent`) |
| Import updates | Updated all imports in `main.py`, `chat_engine.py`, `ollama_client.py` to use `agent_memory.*` paths |
| Packaging | Added `pyproject.toml` with `setuptools` src layout; `pip install -e .` now works |
| Cleanup | Deleted old `memory/` directory and root `command_parser.py` |

---

## Phase 2 — `MemoryConfig` dataclass
*In progress.*

| Task | Change |
|---|---|
| `src/agent_memory/config.py` | **New file.** `MemoryConfig` dataclass centralises all configuration: `db_path`, `chroma_path`, `token_budget`, `response_reserve`, `recent_turns_window`, `summarize_after_turns`, `core_memory_max_facts`, `core_memory_max_scratch_chars`, `archival_similarity_threshold`, `archival_top_k`, `embedder_model`, `ollama_base`, `model`, `timeout`. Includes `MemoryConfig.from_env()` classmethod. |
| `layers/conversation.py` | Replaced `DB_PATH = Path(__file__).parents[3] / ...` with `DB_PATH = MemoryConfig().db_path` |
| `layers/core.py` | Same — `DB_PATH` now sourced from `MemoryConfig` |
| `layers/summary.py` | Same — `DB_PATH` now sourced from `MemoryConfig` |
| `layers/archival.py` | `DB_PATH` → `MemoryConfig().chroma_path`; hardcoded `"all-MiniLM-L6-v2"` → `MemoryConfig().embedder_model` |
| `context_assembler.py` | Removed 4 module-level constants (`TOTAL_CONTEXT_BUDGET`, `RESPONSE_RESERVE`, `RECENT_TURNS_DEFAULT`, `SUMMARIZE_AFTER_TURNS`). All functions now accept optional `config: MemoryConfig` parameter; fall back to `MemoryConfig()` if not provided. |
| `ollama_client.py` | Removed `OLLAMA_BASE`, `MODEL`, `TIMEOUT` constants. All functions accept optional `config: MemoryConfig`; use `config.ollama_base`, `config.model`, `config.timeout`. |
| `chat_engine.py` | `process_message()` and `_run_summarization()` accept optional `config: MemoryConfig`; pass it through to all callees. Fixed return type annotation to `tuple[str, list]`. |
| `main.py` | Constructs `config = MemoryConfig.from_env()` at startup; passes to `check_ollama()`, `process_message()`, and `print_banner()`. Banner now shows `config.model` instead of hardcoded string. |
