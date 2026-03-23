# agent_memory — User Guide

How to use the `agent_memory` library in your own projects.

---

## Installation

```bash
git clone https://github.com/Devank-Garg/magic_memory.git
cd magic_memory
python -m venv env && source env/bin/activate
pip install -e .
```

### Optional provider dependencies

```bash
pip install openai       # for OpenAIProvider
pip install anthropic    # for AnthropicProvider
```

---

## 1. CLI (Quick Demo)

The bundled CLI lets you chat instantly using Ollama:

```bash
# Start Ollama first
OLLAMA_NUM_PARALLEL=2 ollama serve &
ollama pull ministral-3:3b

# Run
python main.py                        # default user
python main.py --user alice           # named user (separate memory)
python main.py --user alice --debug   # show token budget each turn
python main.py --user alice --reset   # wipe alice's memory
```

**In-chat commands:**

| Command    | What it does                      |
|------------|-----------------------------------|
| `/memory`  | Inspect all memory layers         |
| `/reset`   | Wipe memory for current user      |
| `/help`    | Show commands                     |
| `exit`     | Quit                              |

---

## 2. Library API

### Minimal example — Ollama

```python
import asyncio
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers import OllamaProvider

async def main():
    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=OllamaProvider(),
    )

    result = await engine.process_message("alice", "Hello! My name is Alice.")
    print(result.response)          # LLM reply, memory commands stripped
    print(result.memory_actions)    # list of MemoryAction applied this turn

asyncio.run(main())
```

### Minimal example — OpenAI

```python
import asyncio
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers.openai import OpenAIProvider

async def main():
    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=OpenAIProvider(api_key="sk-..."),   # or set OPENAI_API_KEY env var
    )
    result = await engine.process_message("alice", "Hello!")
    print(result.response)

asyncio.run(main())
```

### Minimal example — Anthropic (Claude)

```python
import asyncio
from agent_memory import MemoryEngine, MemoryConfig
from agent_memory.providers.anthropic import AnthropicProvider

async def main():
    engine = MemoryEngine(
        config=MemoryConfig(),
        provider=AnthropicProvider(api_key="ant-..."),  # or set ANTHROPIC_API_KEY env var
    )
    result = await engine.process_message("alice", "Hello!")
    print(result.response)

asyncio.run(main())
```

### Using create_provider() factory

```python
from agent_memory import MemoryEngine, MemoryConfig, create_provider

engine = MemoryEngine(
    config=MemoryConfig(),
    provider=create_provider("openai", api_key="sk-..."),
    # provider=create_provider("anthropic", api_key="ant-..."),
    # provider=create_provider("ollama"),
)
```

---

## 3. Reading memory state

```python
state = engine.get_memory_state("alice")

print(state.user_name)      # "Alice"
print(state.facts)          # ["likes cats", "prefers Python"]
print(state.scratch)        # LLM's working notes
print(state.summary)        # rolling summary of older turns
print(state.message_count)  # total messages in log
```

---

## 4. Resetting a user's memory

```python
engine.reset_user("alice")  # wipes SQLite + ChromaDB for this user
```

---

## 5. Streaming responses

```python
result = await engine.process_message("alice", "Tell me a story", stream=True)
# tokens print to stdout as they arrive; result.response has the full text
```

---

## 6. Configuration

All settings live in `MemoryConfig`. Override what you need:

```python
from agent_memory import MemoryConfig

config = MemoryConfig(
    db_path="~/.myapp/memory.db",       # SQLite location
    chroma_path="~/.myapp/chroma",      # ChromaDB location
    token_budget=4000,                  # total tokens per request
    response_reserve=1000,              # tokens reserved for LLM reply
    recent_turns_window=10,             # verbatim recent turns kept in context
    summarize_after_turns=15,           # compress older turns after N new turns
    core_memory_max_facts=10,           # max facts stored in core memory
    archival_top_k=3,                   # semantic search results per query
)
```

**Via environment variables** (no code change needed):

```bash
export AGENT_MEMORY_DB_PATH=~/.myapp/memory.db
export AGENT_MEMORY_TOKEN_BUDGET=4000
export AGENT_MEMORY_MODEL=gpt-4o
```

```python
config = MemoryConfig.from_env()
```

---

## 7. Self-updating memory

The LLM can update its own memory mid-response using these commands:

```
[REMEMBER: user prefers dark mode]   → stored as a core memory fact
[NOTE: currently debugging auth]     → updates the LLM's scratch pad
[NAME: Alice]                        → stores the user's name
```

These are automatically parsed, applied, and stripped from the displayed response.
`result.memory_actions` contains a `MemoryAction` for each one that fired.

```python
result = await engine.process_message("alice", "I love Python!")
for action in result.memory_actions:
    print(action.type, action.value)
    # e.g. "remember"  "loves Python"
```

---

## 8. Memory layers

| Layer | What it stores | Always in context? |
|---|---|---|
| 0 — Conversation Log | Every message, full fidelity (SQLite) | No |
| 1 — Core Memory | Key facts + user name + scratch pad | Yes |
| 2 — Rolling Summary | LLM-compressed summary of older turns | Yes |
| 3 — Sliding Window | Last N turns verbatim | Yes |
| 4 — Archival Memory | Semantic search over all past messages (ChromaDB) | Relevant hits only |

---

## 9. Multiple users

Each `user_id` is fully isolated — separate SQLite tables and ChromaDB collections:

```python
await engine.process_message("alice", "I hate bugs")
await engine.process_message("bob", "I love bugs")

alice_state = engine.get_memory_state("alice")
bob_state   = engine.get_memory_state("bob")
# completely separate memories
```

---

## 10. Running tests

```bash
source env/bin/activate
python -m pytest tests/ -v
```
