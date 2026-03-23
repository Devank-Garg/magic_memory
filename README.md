# magic_memory

A modular, pip-installable memory layer for AI agents — inspired by MemGPT.

Gives any agent a persistent, multi-layer memory system: core facts always in context, rolling summaries of older history, semantic search over everything ever said. Plug it into any LLM backend — Ollama, OpenAI, or Anthropic.

> **Status:** Core library complete (Phases 1–5 ✅). CLI provider flags shipped. Packaging in progress (Phase 7).

---

## How It Works

```
CONTEXT WINDOW (3000 token budget)
┌──────────────────────────────────────────────────────────────┐
│ [System Prompt]                                              │
│   Layer 1: Core Memory     ~300 tok  always present         │
│   Layer 2: Rolling Summary ~250 tok  compressed old turns   │
│   Layer 4: Archival hits   ~400 tok  semantically relevant  │
│   Memory instructions                                        │
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

The LLM can issue memory commands at the end of its response:
```
[REMEMBER: User prefers Python over JavaScript]
[NOTE: Currently discussing RAG architecture]
[NAME: Devank]
```
These are parsed, applied to core memory, and stripped from the displayed response.

---

## Setup

```bash
git clone https://github.com/Devank-Garg/magic_memory.git
cd magic_memory
python -m venv env && source env/bin/activate
pip install -e ".[cli]"
```

---

## CLI Usage

### Ollama (local, free)

```bash
# 1. Start Ollama
ollama pull ministral-3:3b
OLLAMA_NUM_PARALLEL=2 ollama serve &

# 2. Chat
python main.py --user devank
python main.py --user devank --model llama3.2   # different model
```

### OpenAI

```bash
# API key via flag
python main.py --user devank --provider openai --api-key sk-...

# or via env var
export OPENAI_API_KEY=sk-...
python main.py --user devank --provider openai

# different model
python main.py --user devank --provider openai --model gpt-4o-mini
```

### Anthropic

```bash
# API key via flag
python main.py --user devank --provider anthropic --api-key ant-...

# or via env var
export ANTHROPIC_API_KEY=ant-...
python main.py --user devank --provider anthropic

# different model
python main.py --user devank --provider anthropic --model claude-haiku-4-5-20251001
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--user` | `default` | User ID — each user gets isolated memory |
| `--provider` | `ollama` | LLM backend: `ollama`, `openai`, `anthropic` |
| `--api-key` | `` | API key for openai / anthropic (or set env var) |
| `--model` | provider default | Override model name |
| `--debug` | off | Show token budget breakdown each turn |
| `--reset` | off | Wipe memory for this user and exit |

### In-chat commands

| Command | What it does |
|---|---|
| `/memory` | Inspect all 4 memory layers |
| `/reset` | Wipe memory for current user |
| `/help` | Show commands |
| `exit` | Quit |

---

## Library API

Use `agent_memory` directly inside your own agent — no CLI needed.

```python
import asyncio
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers import OllamaProvider

async def main():
    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=OllamaProvider(),
    )

    result = await engine.process_message("alice", "I love Python!")
    print(result.response)          # LLM reply, memory commands stripped
    print(result.memory_actions)    # list[MemoryAction] — what was stored

asyncio.run(main())
```

Switch provider — everything else stays the same:

```python
from agent_memory.providers.openai import OpenAIProvider
from agent_memory.providers.anthropic import AnthropicProvider

provider = OpenAIProvider(api_key="sk-...")          # OpenAI
provider = AnthropicProvider(api_key="ant-...")      # Anthropic
provider = OllamaProvider()                          # Ollama (default)
```

Inspect or reset memory:

```python
state = engine.get_memory_state("alice")
print(state.user_name)      # "Alice"
print(state.facts)          # ["loves Python", ...]
print(state.summary)        # rolling summary of older turns
print(state.message_count)  # total messages logged

engine.reset_user("alice")  # wipe everything for this user
```

---

## Configuration

```python
from agent_memory import MemoryConfig

config = MemoryConfig(
    db_path="~/.myapp/memory.db",       # SQLite location
    chroma_path="~/.myapp/chroma",      # ChromaDB location
    token_budget=4000,                  # total tokens per request
    recent_turns_window=10,             # verbatim recent turns
    summarize_after_turns=15,           # compress older turns after N new turns
    core_memory_max_facts=10,           # max facts in core memory
)
```

Or from environment variables:

```bash
export AGENT_MEMORY_DB_PATH=~/.myapp/memory.db
export AGENT_MEMORY_TOKEN_BUDGET=4000
```

```python
config = MemoryConfig.from_env()
```

---

## Project Structure

```
magic_memory/
├── src/
│   └── agent_memory/           ← installable library
│       ├── __init__.py         # public API: MemoryEngine, MemoryConfig, ...
│       ├── engine.py           # MemoryEngine — main entry point
│       ├── config.py           # MemoryConfig dataclass
│       ├── context_assembler.py
│       ├── command_parser.py
│       ├── token_counter.py
│       ├── types.py            # MemoryAction, MemoryResponse, MemoryState
│       ├── layers/
│       │   ├── conversation.py # Layer 0: SQLite raw log
│       │   ├── core.py         # Layer 1: always-present facts
│       │   ├── summary.py      # Layer 2: rolling summary
│       │   └── archival.py     # Layer 4: ChromaDB vector store
│       ├── providers/
│       │   ├── base.py         # BaseLLMProvider ABC
│       │   ├── ollama.py       # OllamaProvider
│       │   ├── openai.py       # OpenAIProvider
│       │   ├── anthropic.py    # AnthropicProvider
│       │   └── registry.py     # create_provider() factory
│       └── storage/
│           ├── sqlite_store.py # SQLiteStore
│           └── chroma_store.py # ChromaStore
├── tests/                      # 58 unit tests
├── examples/                   # LangChain adapter, usage examples
├── main.py                     # CLI entry point
├── chat_engine.py              # orchestration
├── GUIDE.md                    # detailed usage guide
├── CHANGES.md                  # per-phase change log
└── pyproject.toml
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Restructure into `src/agent_memory/` package layout | ✅ Done |
| 2 | `MemoryConfig` dataclass — eliminate all scattered constants | ✅ Done |
| 3 | `SQLiteStore` + `ChromaStore` — fix connection/ordering/error bugs | ✅ Done |
| 4 | `BaseLLMProvider` abstraction — Ollama, OpenAI, Anthropic backends | ✅ Done |
| 5 | `MemoryEngine` class — clean public API, fix budget overflow | ✅ Done |
| 6 | CLI cleanup — `--provider` flag, remove dead root files | 🔄 In progress |
| 7 | Packaging — optional deps, `py.typed`, coverage gate, `pip install` | ⬜ Next |

---

## Requirements

- Python 3.11+
- Core deps: `httpx`, `chromadb`, `sentence-transformers`, `tiktoken`
- CLI: `rich`
- OpenAI: `pip install openai`
- Anthropic: `pip install anthropic`
- [Ollama](https://ollama.com) (for local models only)

---

## Running Tests

```bash
source env/bin/activate
python -m pytest tests/ -v
# 58 passed, 2 skipped (openai/anthropic — skipped unless packages installed)
```
