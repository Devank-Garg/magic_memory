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
*Completed.*

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

---

## Phase 3 — `SQLiteStore` + `ChromaStore` storage layer
*Completed. Fixed 5 bugs from the original code review. 32 unit tests passing.*

**Bugs fixed in this phase:**

| Issue | Description | Fix |
|---|---|---|
| #3 | `get_recent_messages` used `ORDER BY id DESC` + Python `reversed()` — fragile if row IDs drift | Replace with `SELECT * FROM (... ORDER BY id DESC LIMIT n) ORDER BY id ASC` subquery |
| #4 | `archive_message` had no error handling — a ChromaDB failure killed the whole message pipeline | Wrap embed+store in `try/except`; log warning, return `None`, treat as non-fatal |
| #5 | `reset_user()` in `main.py` duplicated `_safe()` sanitization from `conversation.py` | Add `delete_user()` method to `SQLiteStore`; `main.py` calls it instead of raw SQL |
| #6 | Every function opened and closed its own `sqlite3.Connection` with no context manager | `SQLiteStore` class wraps connection lifecycle in `with` blocks; shared across all layer functions |
| #7 | `archive_message` did a redundant `collection.get(ids)` check before every insert | Replace check-then-add with `collection.upsert()` — atomic and eliminates the extra round-trip |

**New files:**

| File | Description |
|---|---|
| `src/agent_memory/storage/__init__.py` | Storage package |
| `src/agent_memory/storage/sqlite_store.py` | `SQLiteStore` class — single context-manager-based SQLite connection holder shared across all three SQLite layers. Provides `connection()` context manager, `ensure_table()`, and `delete_user()` |
| `src/agent_memory/storage/chroma_store.py` | `ChromaStore` class — wraps ChromaDB client and collection access with error boundaries. Provides `get_collection()`, lazy embedder singleton, and `delete_collection()` |

**Modified files:**

| File | Change |
|---|---|
| `layers/conversation.py` | Use `SQLiteStore` internally; fix `get_recent_messages` ordering (Issue 3) |
| `layers/core.py` | Use `SQLiteStore` internally (Issue 6) |
| `layers/summary.py` | Use `SQLiteStore` internally (Issue 6) |
| `layers/archival.py` | Use `ChromaStore`; replace `collection.get()` + `add()` with `upsert()` (Issue 7); wrap `archive_message` in `try/except` (Issue 4) |
| `main.py` | `reset_user()` calls `SQLiteStore.delete_user()` and `ChromaStore.delete_collection()` instead of duplicating `_safe()` (Issue 5) |

**New tests:**

| File | What it tests |
|---|---|
| `tests/conftest.py` | Shared fixtures: in-memory SQLite `SQLiteStore`, `MockChromaStore` |
| `tests/unit/layers/test_conversation.py` | `save_message`, `get_recent_messages` ordering, `get_message_count` |
| `tests/unit/layers/test_core.py` | `update_fact` dedup + cap, `update_scratch` cap, `set_user_name`, `render_for_prompt` |
| `tests/unit/layers/test_summary.py` | `save`/`load` roundtrip, `render_for_prompt` empty state |
| `tests/unit/layers/test_archival.py` | `archive_message` non-fatal on failure, `search` returns empty on empty collection |
| `tests/unit/test_command_parser.py` | All three command types, case insensitivity, no commands passthrough, malformed tag ignored |
