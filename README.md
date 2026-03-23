# ∞ Infinite Context Chat

A local LLM chat system that simulates infinite memory using layered context engineering — running on `ministral-3:3b` via Ollama.

## How It Works

```
CONTEXT WINDOW (3000 tokens budget)
┌──────────────────────────────────────────────────────────────┐
│ [System Prompt]                                              │
│   Layer 1: Core Memory     ~300 tok  always present         │
│   Layer 2: Rolling Summary ~250 tok  compressed old turns   │
│   Layer 4: Archival hits   ~400 tok  semantically relevant  │
│   Memory command instructions                                │
├──────────────────────────────────────────────────────────────│
│   Layer 3: Sliding Window  ~800 tok  last N turns verbatim  │
├──────────────────────────────────────────────────────────────│
│   Current user message     ~100 tok                         │
└──────────────────────────────────────────────────────────────┘
         ↕ archive everything                ↕ query
   ChromaDB (all messages, embedded)    SQLite (raw log)
```

### The 4 Memory Layers

| Layer | Name | What It Does | Storage |
|---|---|---|---|
| 1 | Core Memory | Key user facts, always in system prompt | SQLite |
| 2 | Rolling Summary | LLM compresses old turns every 15 messages | SQLite |
| 3 | Sliding Window | Last ~10 turns verbatim | In-context |
| 4 | Archival Memory | Semantic search over ALL past messages | ChromaDB |

### The LLM Can Update Its Own Memory

The LLM learns to call memory commands mid-response:
```
[REMEMBER: User prefers Python over JavaScript]
[NOTE: Currently discussing RAG architecture]
[NAME: Arjun]
```
These get parsed, applied to core memory, and stripped from the displayed response.

---

## Installation

```bash
# 1. Pull the model
ollama pull ministral-3:3b

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run (with parallel request support)
OLLAMA_NUM_PARALLEL=2 ollama serve &
python main.py
```

## Usage

```bash
python main.py                        # default user
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

## File Structure

```
infinite_context/
├── main.py                    # CLI entry point
├── chat_engine.py             # orchestrates all layers
├── ollama_client.py           # async Ollama API wrapper
├── command_parser.py          # parses [REMEMBER:] etc. from LLM
├── requirements.txt
├── memory/
│   ├── token_counter.py       # token budget tracking
│   ├── conversation_store.py  # Layer 0: SQLite raw log
│   ├── core_memory.py         # Layer 1: always-present facts
│   ├── summary_memory.py      # Layer 2: rolling summary
│   ├── archival_memory.py     # Layer 4: ChromaDB vector store
│   └── context_assembler.py   # builds final context under budget
└── data/                      # auto-created
    ├── conversations.db       # SQLite
    └── chroma/                # ChromaDB persistent store
```

## Key Config (context_assembler.py)

```python
TOTAL_CONTEXT_BUDGET  = 3000   # total tokens sent to model
RESPONSE_RESERVE      = 1000   # reserved for model's response  
RECENT_TURNS_DEFAULT  = 10     # verbatim recent turns
SUMMARIZE_AFTER_TURNS = 15     # trigger summarization interval
```
