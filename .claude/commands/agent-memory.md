You are continuing development on **agent_memory** — a pip-installable MemGPT-style memory layer for AI agents. Load this context fully before doing anything.

---

## What This Project Is

A Python library (`src/agent_memory/`) that gives any LLM agent a persistent, multi-layer memory system. Inspired by MemGPT.

**Memory layers:**
- Layer 0: SQLite raw log — every message ever, full fidelity
- Layer 1: Core memory — key facts always in system prompt
- Layer 2: Rolling summary — LLM compresses old turns every 15 messages
- Layer 3: Sliding window — last ~10 turns verbatim in context
- Layer 4: Archival memory — semantic search over all messages via ChromaDB

**The LLM updates its own memory mid-conversation** by appending commands:
```
[REMEMBER: fact]   → stored in Layer 1 core memory
[NOTE: text]       → updates scratch/notes field
[NAME: name]       → sets the user's name
```
These are parsed by `command_parser.py` and stripped from the displayed response.

---

## Current State

All 7 phases complete. Version `0.2.0`.

| Phase | Description | Status |
|---|---|---|
| 1 | `src/agent_memory/` package layout | ✅ Done |
| 2 | `MemoryConfig` dataclass | ✅ Done |
| 3 | `SQLiteStore` + `ChromaStore` | ✅ Done |
| 4 | `BaseLLMProvider` — Ollama, OpenAI, Anthropic | ✅ Done |
| 5 | `MemoryEngine` public API | ✅ Done |
| 6 | CLI `--provider` flag, multi-provider support | ✅ Done |
| 7 | Packaging — extras, `py.typed`, coverage gate | ✅ Done |
| 8 | PyPI publish + multi-user abstractions | ⬜ Next |

---

## Key Files

```
src/agent_memory/
├── __init__.py          # exports: MemoryEngine, MemoryConfig, MemoryAction, MemoryResponse, MemoryState, create_provider
├── engine.py            # MemoryEngine — main public API
├── config.py            # MemoryConfig dataclass (all 14 fields, from_env() classmethod)
├── cli.py               # agent-memory console script entry point
├── chat_engine.py       # CLI orchestration (process_message pipeline)
├── context_assembler.py # builds token-budgeted message list for LLM
├── command_parser.py    # parses/strips [REMEMBER:] [NOTE:] [NAME:] from LLM output
├── token_counter.py     # tiktoken cl100k_base token estimation
├── types.py             # MemoryAction, MemoryResponse, MemoryState dataclasses
├── layers/
│   ├── conversation.py  # Layer 0: SQLite raw log
│   ├── core.py          # Layer 1: always-present facts
│   ├── summary.py       # Layer 2: rolling summary
│   └── archival.py      # Layer 4: ChromaDB vector store
├── providers/
│   ├── base.py          # BaseLLMProvider ABC + LLMOptions dataclass
│   ├── ollama.py        # OllamaProvider (shared httpx.AsyncClient)
│   ├── openai.py        # OpenAIProvider (AsyncOpenAI, streaming)
│   ├── anthropic.py     # AnthropicProvider (splits system prompt per API spec)
│   └── registry.py      # create_provider(name, config) factory
└── storage/
    ├── sqlite_store.py  # SQLiteStore — context-manager connection lifecycle
    └── chroma_store.py  # ChromaStore — lazy embedder singleton, cosine similarity
```

Root files:
- `main.py` — thin shim: `from agent_memory.cli import main`
- `pyproject.toml` — version 0.2.0, extras: `[cli]` `[openai]` `[anthropic]` `[dev]`
- `tests/` — 66 unit tests, 82% coverage, gate at 80%
- `CHANGES.md` — per-phase change log

---

## Public API

```python
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers import OllamaProvider

engine = MemoryEngine(config=MemoryConfig(), provider=OllamaProvider())
result = await engine.process_message("alice", "I love Python!")
result.response        # str — LLM reply, commands stripped
result.memory_actions  # list[MemoryAction]

state = engine.get_memory_state("alice")
state.user_name        # str
state.facts            # list[str]
state.summary          # str
state.message_count    # int

engine.reset_user("alice")
```

---

## MemoryConfig — all 14 fields

| Field | Default | Env var |
|---|---|---|
| `db_path` | `./data/conversations.db` | `AGENT_MEMORY_DB_PATH` |
| `chroma_path` | `./data/chroma` | `AGENT_MEMORY_CHROMA_PATH` |
| `token_budget` | `3000` | `AGENT_MEMORY_TOKEN_BUDGET` |
| `response_reserve` | `1000` | `AGENT_MEMORY_RESPONSE_RESERVE` |
| `recent_turns_window` | `10` | `AGENT_MEMORY_RECENT_TURNS` |
| `summarize_after_turns` | `15` | `AGENT_MEMORY_SUMMARIZE_AFTER` |
| `core_memory_max_facts` | `10` | `AGENT_MEMORY_MAX_FACTS` |
| `core_memory_max_scratch_chars` | `500` | `AGENT_MEMORY_MAX_SCRATCH` |
| `archival_similarity_threshold` | `0.7` | `AGENT_MEMORY_ARCHIVAL_THRESHOLD` |
| `archival_top_k` | `3` | `AGENT_MEMORY_ARCHIVAL_TOP_K` |
| `embedder_model` | `all-MiniLM-L6-v2` | `AGENT_MEMORY_EMBEDDER_MODEL` |
| `model` | `ministral-3:3b` | `AGENT_MEMORY_MODEL` |
| `ollama_base` | `http://localhost:11434` | `AGENT_MEMORY_OLLAMA_BASE` |
| `timeout` | `120.0` | `AGENT_MEMORY_TIMEOUT` |

---

## Development Rules

1. **Never push directly to main** — always create a branch and raise a PR
2. **Branch naming**: `feat/`, `fix/`, `docs/`, `phase-N/` prefixes
3. **Update `CHANGES.md`** when completing a phase or significant fix
4. **Coverage gate**: `--cov-fail-under=80` is enforced — tests must stay ≥ 80%
5. **`cli.py` and `chat_engine.py` are excluded from coverage** (CLI/orchestration code)
6. **`python main.py` must always work** — it's a shim to `agent_memory.cli:main`

---

## Dev Setup

```bash
git clone https://github.com/Devank-Garg/magic_memory.git
cd magic_memory
python -m venv env && source env/bin/activate
pip install -e ".[cli,dev]"
```

Run tests:
```bash
python -m pytest tests/ -v
# 66 passed, 2 skipped, coverage ≥ 80%
```

Run CLI:
```bash
agent-memory --user devank                              # Ollama
agent-memory --user devank --provider openai            # OpenAI
agent-memory --user devank --provider anthropic         # Anthropic
```

---

## Phase 8 — What's Next

Planned work:
- **PyPI publish**: `python -m build` + `twine upload dist/*`
- **`BaseStore` ABC**: abstract SQLiteStore so it can be swapped for PostgresStore
- **`BaseVectorStore` ABC**: abstract ChromaStore for remote embedding APIs
- **DI for singletons**: module-level `_store` instances in layers should be injected, not global
- **Embedding API abstraction**: replace local sentence-transformers with pluggable embedding backend

---

## If $ARGUMENTS is provided

Focus on that specific task, file, or feature. Otherwise, read the current git status and open PRs (`gh pr list`) to understand what's in flight before suggesting next steps.
