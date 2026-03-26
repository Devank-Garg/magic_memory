# agent_memory

A pip-installable memory layer for AI agents — inspired by MemGPT.

Drop persistent, multi-layer memory into any agent or chatbot. Core facts stay in context forever, older conversations are compressed into rolling summaries, and everything ever said is searchable via semantic search. Swap between Ollama, OpenAI, or Anthropic with a single flag.

---

## How It Works

```
CONTEXT WINDOW (3000 token budget)
┌──────────────────────────────────────────────────────────────┐
│ [System Prompt]                                              │
│   Layer 1: Core Memory     ~300 tok  always present         │
│   Layer 2: Rolling Summary ~250 tok  compressed old turns   │
│   Layer 4: Archival hits   ~400 tok  semantically relevant  │
├──────────────────────────────────────────────────────────────┤
│   Layer 3: Sliding Window  ~800 tok  last N turns verbatim  │
├──────────────────────────────────────────────────────────────┤
│   Current user message     ~100 tok                         │
└──────────────────────────────────────────────────────────────┘
         ↕ archive everything              ↕ query
   ChromaDB (all messages, embedded)   SQLite (raw log)
```

| Layer | Name | What It Does | Storage |
|---|---|---|---|
| 0 | Conversation Log | Every message, full fidelity | SQLite |
| 1 | Core Memory | Key facts — always in system prompt | SQLite |
| 2 | Rolling Summary | LLM compresses old turns every 15 messages | SQLite |
| 3 | Sliding Window | Last ~10 turns verbatim | In-context |
| 4 | Archival Memory | Semantic search over all past messages | ChromaDB |

The LLM can update memory mid-conversation by appending commands to its response:

```
[REMEMBER: prefers dark mode]
[NOTE: currently debugging a RAG pipeline]
[NAME: Devank]
```

These are parsed, applied to core memory, and stripped before the response is displayed.

---

## Setup

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) (for local models)

```bash
git clone https://github.com/Devank-Garg/magic_memory.git
cd magic_memory
python -m venv env && source env/bin/activate
pip install -e ".[cli]"
```

For OpenAI, Anthropic, or LangChain integration, install the relevant extra:

```bash
pip install -e ".[cli,openai]"               # OpenAI
pip install -e ".[cli,anthropic]"            # Anthropic
pip install -e ".[cli,langchain]"            # LangChain adapter
pip install -e ".[cli,openai,anthropic]"     # all native providers
```

---

## CLI Usage

After installing, the `agent-memory` command is available. `python main.py` also works as a shorthand.

### Ollama (local, free)

```bash
ollama pull mistral
OLLAMA_NUM_PARALLEL=2 ollama serve &

agent-memory --user devank
agent-memory --user devank --model llama3.2
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
agent-memory --user devank --provider openai
agent-memory --user devank --provider openai --model gpt-4o-mini
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=ant-...
agent-memory --user devank --provider anthropic
agent-memory --user devank --provider anthropic --model claude-haiku-4-5-20251001
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--user` | `default` | User ID — each user gets isolated memory |
| `--provider` | `ollama` | LLM backend: `ollama`, `openai`, `anthropic` |
| `--api-key` | — | API key (or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `--model` | provider default | Override model name |
| `--debug` | off | Show token budget breakdown each turn |
| `--reset` | off | Wipe memory for this user and exit |

### In-chat commands

| Command | What it does |
|---|---|
| `/memory` | Inspect all memory layers |
| `/reset` | Wipe memory for the current user |
| `/help` | Show available commands |
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
    print(result.response)        # LLM reply, memory commands stripped
    print(result.memory_actions)  # list[MemoryAction] — what was stored

asyncio.run(main())
```

### Swap providers

```python
from agent_memory.providers.openai import OpenAIProvider
from agent_memory.providers.anthropic import AnthropicProvider

provider = OllamaProvider()                         # local (default)
provider = OpenAIProvider(api_key="sk-...")         # OpenAI
provider = AnthropicProvider(api_key="ant-...")     # Anthropic
```

### Inspect or reset memory

```python
state = engine.get_memory_state("alice")
print(state.user_name)      # "Alice"
print(state.facts)          # ["loves Python", ...]
print(state.summary)        # rolling summary of older turns
print(state.message_count)  # total messages logged

engine.reset_user("alice")  # wipe everything for this user
```

---

## LangChain Integration

Install the extra and import a single class:

```bash
pip install "agent-memory[langchain]"
pip install langchain-ollama       # or langchain-openai / langchain-anthropic
```

```python
from langchain_ollama import ChatOllama   # swap for ChatOpenAI / ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from agent_memory.integrations.langchain import AgentMemoryFullHistory

chain = RunnableWithMessageHistory(
    ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),   # SystemMessage(memory) + chat turns
        ("human", "{input}"),
    ]) | ChatOllama(model="mistral"),
    get_session_history=lambda sid: AgentMemoryFullHistory(sid),
    input_messages_key="input",
    history_messages_key="history",
)

resp = chain.invoke(
    {"input": "I love Python."},
    config={"configurable": {"session_id": "alice"}},
)
print(resp.content)           # plain text reply
print(resp.usage_metadata)    # {"input_tokens": N, "output_tokens": N, ...}
```

`AgentMemoryFullHistory` injects all four memory layers as a `SystemMessage` automatically each turn — core facts, rolling summary, archival hits, and behaviour instructions. See `examples/` for full working scripts.

Also exported for standalone use in custom chains:

```python
from agent_memory import build_system_prompt

system_prompt: str = build_system_prompt("alice", current_user_message, config)
```

---

## Configuration

Pass a `MemoryConfig` when constructing `MemoryEngine`, or let the CLI pick it up via `MemoryConfig.from_env()`.

```python
from agent_memory import MemoryConfig, MemoryEngine
from agent_memory.providers import OllamaProvider

config = MemoryConfig(
    token_budget=4000,
    recent_turns_window=10,
    summarize_after_turns=20,
)
engine = MemoryEngine(config=config, provider=OllamaProvider(config=config))
```

### All options

**Storage**

| Field | Default | Env var | Description |
|---|---|---|---|
| `db_path` | `./data/conversations.db` | `AGENT_MEMORY_DB_PATH` | SQLite file path |
| `chroma_path` | `./data/chroma` | `AGENT_MEMORY_CHROMA_PATH` | ChromaDB directory |

**Token budget**

| Field | Default | Env var | Description |
|---|---|---|---|
| `token_budget` | `3000` | `AGENT_MEMORY_TOKEN_BUDGET` | Total input tokens sent to the model per turn |
| `response_reserve` | `1000` | `AGENT_MEMORY_RESPONSE_RESERVE` | Tokens reserved for the model's reply |
| `recent_turns_window` | `10` | `AGENT_MEMORY_RECENT_TURNS` | How many recent turns to include verbatim (Layer 3) |
| `summarize_after_turns` | `15` | `AGENT_MEMORY_SUMMARIZE_AFTER` | Compress older turns into a summary after N total turns |

**Core memory**

| Field | Default | Env var | Description |
|---|---|---|---|
| `core_memory_max_facts` | `10` | `AGENT_MEMORY_MAX_FACTS` | Max number of facts stored in core memory (Layer 1) |
| `core_memory_max_scratch_chars` | `500` | `AGENT_MEMORY_MAX_SCRATCH` | Max characters in the scratch/notes field |

**Archival memory**

| Field | Default | Env var | Description |
|---|---|---|---|
| `archival_similarity_threshold` | `0.7` | `AGENT_MEMORY_ARCHIVAL_THRESHOLD` | Minimum cosine similarity for a result to be included |
| `archival_top_k` | `3` | `AGENT_MEMORY_ARCHIVAL_TOP_K` | Max archival results injected per turn |
| `embedder_model` | `all-MiniLM-L6-v2` | `AGENT_MEMORY_EMBEDDER_MODEL` | Sentence-transformer model used for embeddings |

**LLM backend**

| Field | Default | Env var | Description |
|---|---|---|---|
| `model` | `ministral-3:3b` | `AGENT_MEMORY_MODEL` | Model name passed to the provider |
| `ollama_base` | `http://localhost:11434` | `AGENT_MEMORY_OLLAMA_BASE` | Ollama server URL |
| `timeout` | `120.0` | `AGENT_MEMORY_TIMEOUT` | Request timeout in seconds |

### Via environment variables

Any field can be set with an env var — useful for deployments or keeping secrets out of code:

```bash
export AGENT_MEMORY_TOKEN_BUDGET=4000
export AGENT_MEMORY_MODEL=llama3.2
export AGENT_MEMORY_DB_PATH=~/.myapp/memory.db
```

```python
config = MemoryConfig.from_env()  # unset vars fall back to defaults
```

---

## Project Structure

```
magic_memory/
├── src/
│   └── agent_memory/           ← installable library
│       ├── engine.py           # MemoryEngine — public API entry point
│       ├── config.py           # MemoryConfig dataclass
│       ├── cli.py              # agent-memory console script
│       ├── chat_engine.py      # CLI orchestration layer
│       ├── context_assembler.py# build_context(), build_system_prompt()
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
│       ├── integrations/
│       │   └── langchain.py    # AgentMemoryFullHistory — BaseChatMessageHistory adapter
│       └── storage/
│           ├── sqlite_store.py
│           └── chroma_store.py
├── tests/                      # 91 unit tests, 86% coverage
├── examples/
│   ├── 01_standalone_ollama.py # MemoryEngine + Ollama, no LangChain
│   ├── 02_standalone_openai.py # MemoryEngine + OpenAI, no LangChain
│   ├── 03_langchain_ollama.py  # AgentMemoryFullHistory + ChatOllama
│   └── 04_langchain_openai.py  # AgentMemoryFullHistory + ChatOpenAI
├── main.py                     # thin shim → agent_memory.cli:main
├── CHANGES.md                  # per-phase change log
└── pyproject.toml
```

---

## Running Tests

```bash
source env/bin/activate
python -m pytest tests/ -v
# 91 passed — coverage ≥ 80% enforced (currently 86%)
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Restructure into `src/agent_memory/` package layout | ✅ Done |
| 2 | `MemoryConfig` dataclass — centralise all config | ✅ Done |
| 3 | `SQLiteStore` + `ChromaStore` — fix connection/ordering/error bugs | ✅ Done |
| 4 | `BaseLLMProvider` — Ollama, OpenAI, Anthropic backends | ✅ Done |
| 5 | `MemoryEngine` — clean public API, fix token budget overflow | ✅ Done |
| 6 | CLI — `--provider` flag, multi-provider support | ✅ Done |
| 7 | Packaging — optional extras, `py.typed`, coverage gate, v0.2.0 | ✅ Done |
| 8 | PyPI publish + multi-user abstractions (`BaseStore`, `BaseVectorStore`) | ⬜ Planned |
| 10a | LangChain adapter — `AgentMemoryFullHistory`, `build_system_prompt()`, `[langchain]` extra | ✅ Done |
| 10b | LangChain tools, LangGraph store integration | ⬜ Planned |
