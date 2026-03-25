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

---

## Phase 4 — `BaseLLMProvider` abstraction
*Completed. Fixed 1 bug. 12 new tests (44 total passing).*

**Goal:** Decouple LLM calls from Ollama-specific code so any backend (OpenAI, Anthropic, Ollama) can be swapped in via config.

**Bug fixed in this phase:**

| Issue | Description | Fix |
|---|---|---|
| #8 | `ollama_client.py` creates a new `httpx.AsyncClient` on every single call — expensive and leaks connections under load | `OllamaProvider` holds a single shared `httpx.AsyncClient` as an instance attribute |

**New files:**

| File | Description |
|---|---|
| `src/agent_memory/providers/base.py` | `BaseLLMProvider` ABC with `chat()`, `summarize()`, `health_check()` abstract methods; `LLMOptions` dataclass (temperature, num_ctx, stream) |
| `src/agent_memory/providers/ollama.py` | `OllamaProvider` — migrates all logic from root `ollama_client.py`; shared `httpx.AsyncClient` (Issue 8 fix) |
| `src/agent_memory/providers/openai.py` | `OpenAIProvider` stub — raises `NotImplementedError`; ready for Phase 7 implementation |
| `src/agent_memory/providers/anthropic.py` | `AnthropicProvider` stub — raises `NotImplementedError`; ready for Phase 7 implementation |
| `src/agent_memory/providers/registry.py` | `create_provider(name, config)` factory — maps `"ollama"` / `"openai"` / `"anthropic"` to provider instances |

**Modified files:**

| File | Change |
|---|---|
| `src/agent_memory/providers/__init__.py` | Export `BaseLLMProvider`, `OllamaProvider`, `create_provider` |
| `chat_engine.py` | Import `OllamaProvider` from `agent_memory.providers`; remove `from ollama_client import chat, summarize`; provider passed in or created from config |

**New tests:**

| File | What it tests |
|---|---|
| `tests/unit/providers/test_ollama.py` | `OllamaProvider.chat` mocks `httpx`; verifies shared client reuse; streaming path |
| `tests/unit/providers/test_registry.py` | `create_provider("ollama", ...)` returns `OllamaProvider`; unknown name raises `ValueError` |

---

## Phase 5 — `MemoryEngine` class
*Completed. Fixed 1 bug. 14 new tests (58 total passing).*

**Goal:** Expose a clean, single-class public API so library consumers never need to touch layers, assembler, or chat_engine directly.

**Bug fixed in this phase:**

| Issue | Description | Fix |
|---|---|---|
| #2 | `context_assembler.build_context` does not guard against a negative token budget — if the system prompt alone exceeds `token_budget`, `available_for_history` goes negative, causing `reversed()` to silently include zero messages | Clamp `available_for_history = max(0, budget - current_msg_tokens - 100)` and log a warning when system prompt exceeds budget |

**New files:**

| File | Description |
|---|---|
| `src/agent_memory/types.py` | `MemoryAction` dataclass (type, value); `MemoryResponse` dataclass (response, memory_actions); `MemoryState` dataclass (core, summary, message_count) — clean return types for the public API |
| `src/agent_memory/engine.py` | `MemoryEngine` class — single entry point for library consumers. `__init__(config, provider)`, `async process_message(user_id, user_message, stream) -> MemoryResponse`, `get_memory_state(user_id) -> MemoryState`, `reset_user(user_id)` |

**Modified files:**

| File | Change |
|---|---|
| `src/agent_memory/context_assembler.py` | Clamp `available_for_history` to `>= 0`; log warning when budget is exceeded (Issue 2 fix) |
| `src/agent_memory/__init__.py` | Export `MemoryEngine`, `MemoryConfig`, `MemoryResponse`, `MemoryAction`, `MemoryState`, `create_provider` |

**New tests:**

| File | What it tests |
|---|---|
| `tests/unit/test_engine.py` | `process_message` returns `MemoryResponse`; `memory_actions` list populated when commands present; `get_memory_state` returns correct counts; `reset_user` clears state |
| `tests/unit/test_context_assembler.py` | Budget overflow: system prompt > budget still returns valid messages list; `available_for_history` never negative |

---

## Phase 9 — Custom system prompt support
*Allows library consumers and framework integrations (LangChain, etc.) to supply their own system instructions without forking the assembler.*

**Design:**
The system prompt has two distinct parts:
1. **Memory context** (always generated from layers — core memory, rolling summary, archival results) — never overridable, ensures the memory system works regardless.
2. **Behaviour instructions** (previously hardcoded in `context_assembler.py`) — now fully configurable.

**Changes:**

| File | Change |
|---|---|
| `src/agent_memory/context_assembler.py` | Extracted hardcoded BEHAVIOUR + MEMORY COMMANDS block to module-level constant `DEFAULT_BEHAVIOUR_PROMPT`. `build_context` now uses `config.system_prompt` when set, falling back to `DEFAULT_BEHAVIOUR_PROMPT`. |
| `src/agent_memory/config.py` | Added `system_prompt: str \| None = None` field. `from_env()` reads `AGENT_MEMORY_SYSTEM_PROMPT`. |
| `src/agent_memory/__init__.py` | Exports `DEFAULT_BEHAVIOUR_PROMPT` so callers can compose on top of it. |

**Usage examples:**

```python
# Full replacement — your own instructions, memory context still prepended
config = MemoryConfig(system_prompt="You are a customer support agent for Acme Corp.")

# Compose: keep the default memory commands, add your persona
from agent_memory import DEFAULT_BEHAVIOUR_PROMPT
config = MemoryConfig(
    system_prompt=f"{DEFAULT_BEHAVIOUR_PROMPT}\n\nYou speak only in haiku."
)

# Via env var (useful for deployment / framework injection)
# AGENT_MEMORY_SYSTEM_PROMPT="You are a strict JSON-only responder." agent-memory
```

**New tests:**

| Test | What it verifies |
|---|---|
| `test_custom_system_prompt_replaces_default_behaviour` | Custom prompt appears; default `## BEHAVIOUR` block absent |
| `test_default_behaviour_prompt_used_when_no_override` | Default behaviour injected when `system_prompt=None` |
| `test_memory_context_always_prepended_with_custom_prompt` | `CORE MEMORY` block always present even with a custom prompt |
| `test_from_env_system_prompt` | `AGENT_MEMORY_SYSTEM_PROMPT` env var is picked up |
| `test_system_prompt_none_by_default` | Default is `None` (built-in behaviour applies) |

---

## Phase 8 — Bug fixes (code review)
*Addresses 7 bugs found during post-refactor code review.*

**Bugs fixed:**

| Issue | Description | Fix |
|---|---|---|
| #9 | `MemoryEngine` config not propagated to layers — layers bound `_store` at import time using default `MemoryConfig()`, so any custom `db_path` / `chroma_path` passed to `MemoryEngine` was silently ignored | `MemoryEngine.__init__` now creates `SQLiteStore` and `ChromaStore` from its own config and rebinds the module-level `_store` / `_chroma` / `_cfg` in each layer. `core.update_fact` and `core.update_scratch` accept an optional `config` arg; `parse_and_apply` forwards it from the engine. |
| #10 | `MemoryConfig.from_env()` silently dropped numeric/float env vars set to `0` — `if v := _int(k):` treats `0` as falsy | Changed all walrus-based guards to `if (v := ...) is not None:` |
| #11 | Empty/blank `user_id` mapped to bare table name `conv_` causing all callers with empty IDs to share data | `SQLiteStore._safe()` and `ChromaStore._safe()` now raise `ValueError` for empty/blank user IDs |
| #12 | `response_reserve` field defined in `MemoryConfig` but never subtracted from the token budget in `context_assembler` | `build_context` now starts with `budget = config.token_budget - config.response_reserve` |
| #13 | No error boundary around `parse_and_apply` in `engine.process_message` — a parser crash (e.g. unusual LLM output) would abort the entire pipeline | Wrapped in `try/except`; on failure logs a warning and returns the raw response with empty `memory_actions` |
| #14 | `_ensure_table` issued `CREATE TABLE IF NOT EXISTS` SQL on every single DB read/write | Added `_tables_ensured: set[str]` to `SQLiteStore`; each layer's `_ensure_table` helper now skips the SQL if the table has already been ensured in this process |
| #15 | CLI imported and called `chat_engine.process_message` — a parallel orchestration path that duplicated all of `MemoryEngine.process_message` and would diverge | `cli.py` now imports `MemoryEngine` directly; `chat_engine.py` retained as a legacy shim but no longer on the main code path |

**Modified files:**

| File | Change |
|---|---|
| `src/agent_memory/engine.py` | Own `SQLiteStore` + `ChromaStore` instances; rebind layer module-level singletons on init (#9); pass config to `parse_and_apply` (#9); `try/except` around parser (#13); `reset_user` uses owned stores |
| `src/agent_memory/layers/core.py` | `update_fact` + `update_scratch` accept optional `config` param (#9); `_ensure_table` one-shot guard (#14) |
| `src/agent_memory/command_parser.py` | `parse_and_apply` accepts optional `config` param and forwards to core functions (#9) |
| `src/agent_memory/config.py` | `from_env()` walrus guards use `is not None` (#10) |
| `src/agent_memory/storage/sqlite_store.py` | `_safe()` raises on empty user_id (#11); `_tables_ensured` set added (#14) |
| `src/agent_memory/storage/chroma_store.py` | `_safe()` raises on empty user_id (#11) |
| `src/agent_memory/context_assembler.py` | Subtract `response_reserve` from initial budget (#12) |
| `src/agent_memory/layers/conversation.py` | `_ensure_table` one-shot guard (#14) |
| `src/agent_memory/layers/summary.py` | `_ensure_table` one-shot guard (#14) |
| `src/agent_memory/cli.py` | Import and use `MemoryEngine` directly; remove `chat_engine` dependency (#15) |

**New tests:**

| File | What it tests |
|---|---|
| `tests/unit/test_engine.py` | `test_engine_uses_configured_db_path` — custom db_path respected (#9); `test_engine_config_respected_by_update_fact` — max_facts cap honoured (#9); `test_process_message_handles_parser_error` — parser crash returns raw response (#13) |
| `tests/unit/test_config.py` | `test_from_env_zero_numeric_not_dropped` — zero values not silently dropped (#10) |
| `tests/unit/test_sqlite_store.py` | Empty/blank user_id raises ValueError (#11); normal sanitization; `delete_user` roundtrip |
| `tests/unit/test_context_assembler.py` | `test_response_reserve_reduces_history_budget` — reserve shrinks available history tokens (#12) |

---

## Phase 7 — Packaging
*Completed. 8 new tests (66 passing, 82% coverage).*

**Goal:** Make the library installable as `pip install agent-memory` with optional provider extras, add type information, and enforce a coverage gate.

| Task | Change |
|---|---|
| Optional deps | Added `openai = ["openai>=1.0.0"]` and `anthropic = ["anthropic>=0.20.0"]` extras to `pyproject.toml` — `pip install agent-memory[openai]`, `pip install agent-memory[anthropic]` |
| `py.typed` | Added empty `src/agent_memory/py.typed` marker + `[tool.setuptools.package-data]` entry so mypy/pyright recognise the package as typed |
| Coverage gate | `[tool.pytest.ini_options] addopts = "--cov=agent_memory --cov-fail-under=80"`; `[tool.coverage.run]` config added; 82% coverage achieved |
| Dead file cleanup | Deleted orphaned root `ollama_client.py` (replaced by `OllamaProvider` in Phase 4) |
| Version bump | `pyproject.toml` version `0.1.0` → `0.2.0` |

**New tests:**

| File | What it tests |
|---|---|
| `tests/unit/test_config.py` | `MemoryConfig.from_env()` — defaults, all env var overrides, path coercion |
| `tests/unit/test_token_counter.py` | `count_tokens`, `count_messages_tokens` — empty input, overhead accounting, missing content key |
