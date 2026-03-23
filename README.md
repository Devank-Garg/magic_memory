# magic_memory

A modular, pip-installable memory layer for AI agents — inspired by MemGPT.

Gives any agent a persistent, multi-layer memory system: core facts always in context, rolling summaries of older history, semantic search over everything ever said. Plug it into any LLM backend (Ollama, OpenAI, Anthropic).

> **Status:** Active refactor — transforming from a standalone Ollama CLI into a reusable `agent_memory` library. See [Roadmap](#roadmap) below.

---

## How It Works

```
CONTEXT WINDOW (3000 token budget)
┌──────────────────────────────────────────────────────────────┐
│ [System Prompt]                                              │
│   Layer 1: Core Memory     ~300 tok  always present         │
│   Layer 2: Rolling Summary ~250 tok  compressed old turns   │
│   Layer 4: Archival hits   ~400 tok  semantically relevant  │
│   Memory command instructions                                │
├──────────────────────────────────────────────────────────────┤
│   Layer 3: Sliding Window  ~800 tok  last N turns verbatim  │
├──────────────────────────────────────────────────────────────┤
│   Current user message     ~100 tok                         │
└──────────────────────────────────────────────────────────────┘
         ↕ archive everything                ↕ query
   ChromaDB (all messages, embedded)    SQLite (raw log)
```

### Memory Layers

| Layer | Name | What It Does | Storage |
|---|---|---|---|
| 0 | Conversation Log | Every message ever, full fidelity | SQLite |
| 1 | Core Memory | Key user facts, always in system prompt | SQLite |
| 2 | Rolling Summary | LLM compresses old turns every 15 messages | SQLite |
| 3 | Sliding Window | Last ~10 turns verbatim | In-context |
| 4 | Archival Memory | Semantic search over ALL past messages | ChromaDB |

### Self-Updating Memory

The LLM can issue memory commands mid-response:
```
[REMEMBER: User prefers Python over JavaScript]
[NOTE: Currently discussing RAG architecture]
[NAME: Arjun]
```
These are parsed, applied to core memory, and stripped from the displayed response.

---

## Quick Start (current CLI)

```bash
# 1. Clone and set up
git clone https://github.com/Devank-Garg/magic_memory.git
cd magic_memory
python -m venv env && source env/bin/activate
pip install -e ".[cli]"

# 2. Pull the model
ollama pull ministral-3:3b
OLLAMA_NUM_PARALLEL=2 ollama serve &

# 3. Run
python main.py
python main.py --user alice           # named user (separate memory)
python main.py --user alice --debug   # show token budget each turn
python main.py --user alice --reset   # wipe alice's memory
```

### In-Chat Commands
```
/memory   — inspect all 4 memory layers
/reset    — wipe memory for current user
/help     — show commands
exit      — quit
```

---

## Target API (after refactor)

Once the refactor is complete, using `agent_memory` in your own agent will look like this:

```python
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers.ollama import OllamaProvider

provider = OllamaProvider(base_url="http://localhost:11434", model="mistral")
engine   = MemoryEngine(config=MemoryConfig(), provider=provider)

result = await engine.process_message(user_id="alice", user_message="hello")
print(result.response)          # str — LLM response with commands stripped
print(result.memory_actions)    # list[MemoryAction] — what was stored
```

Switch providers without changing anything else:

```python
from agent_memory.providers.openai import OpenAIProvider
provider = OpenAIProvider(api_key="...", model="gpt-4o")
```

Configure everything from one place:

```python
config = MemoryConfig(
    token_budget=4000,
    recent_turns_window=15,
    enabled_layers={0, 1, 2, 3},   # disable ChromaDB (layer 4)
    db_path="~/.myapp/memory.db",
)
```

---

## Project Structure

```
magic_memory/
├── src/
│   └── agent_memory/               ← installable library (in progress)
│       ├── __init__.py
│       ├── context_assembler.py
│       ├── command_parser.py
│       ├── token_counter.py
│       ├── layers/
│       │   ├── conversation.py     # Layer 0: SQLite raw log
│       │   ├── core.py             # Layer 1: always-present facts
│       │   ├── summary.py          # Layer 2: rolling summary
│       │   ├── archival.py         # Layer 4: ChromaDB vector store
│       └── providers/              # LLM backends (Phase 4)
├── cli/                            # CLI entry point (Phase 6)
├── tests/                          # (Phase 7)
├── main.py                         # current CLI entry point
├── chat_engine.py                  # orchestration
├── ollama_client.py                # Ollama API wrapper
└── pyproject.toml
```

---

## Roadmap

The project is being refactored in 7 phases. Each phase ends with a working, committed state.

| Phase | Description | Status |
|---|---|---|
| 1 | Restructure into `src/agent_memory/` package layout | ✅ Done |
| 2 | `MemoryConfig` dataclass — eliminate all scattered constants | ✅ Done |
| 3 | `SQLiteStore` + `ChromaStore` — fix connection/ordering/error bugs | ⬜ Next |
| 4 | `BaseLLMProvider` abstraction — Ollama, OpenAI, Anthropic backends | ⬜ |
| 5 | `MemoryEngine` class — clean public API, fix budget overflow | ⬜ |
| 6 | CLI separation — `cli/main.py` as thin wrapper | ⬜ |
| 7 | Packaging, tests, `pip install agent-memory` | ⬜ |

---

## Configuration Reference

All constants will move to `MemoryConfig` in Phase 2. Currently in `src/agent_memory/context_assembler.py`:

```python
TOTAL_CONTEXT_BUDGET  = 3000   # total tokens sent to model
RESPONSE_RESERVE      = 1000   # reserved for model's response
RECENT_TURNS_DEFAULT  = 10     # verbatim recent turns
SUMMARIZE_AFTER_TURNS = 15     # trigger summarization interval
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) (for the CLI demo)
- Dependencies: `httpx`, `chromadb`, `sentence-transformers`, `tiktoken`, `rich`
