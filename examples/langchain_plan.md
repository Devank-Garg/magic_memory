# LangChain Integration Plan

## Current state (works today)

`examples/langchain_example.py` shows two working adapters built from the
existing `agent_memory` layer APIs. No changes to the library are needed.

| Adapter | What works | What's missing |
|---|---|---|
| `AgentMemoryHistory` (Approach 1) | Persistent sliding-window storage, multi-user isolation | Core facts / summary / archival hits not in context |
| `AgentMemoryFullHistory` (Approach 2) | All 4 layers injected as a SystemMessage | Archival search uses empty query (current input not threaded through) |

---

## Gap: archival search needs the current input

In `AgentMemoryFullHistory.messages`, we need the *current user message* to
run a semantic search (`archival.render_for_prompt(user_id, current_input)`).
But `BaseChatMessageHistory.messages` is a property — it doesn't receive
the current input.

**Fix needed in agent_memory:** add a `get_session_history_factory` helper
that accepts a `current_input` argument and returns a pre-configured history object.

```python
# What this looks like from the caller's side
def get_history(session_id: str) -> AgentMemoryFullHistory:
    # current input not available here — limitation of RunnableWithMessageHistory
    return AgentMemoryFullHistory(session_id)
```

LangChain's `RunnableWithMessageHistory` does not pass the current message to
`get_session_history`. A proper fix requires either:
- A `RunnableLambda` pre-step that sets `current_input` on the history object
- Or a custom `Runnable` that wraps the chain and calls `history.set_input()`

---

## Recommended: add `agent_memory.integrations.langchain` module

To ship a clean, pip-installable LangChain integration, add:

```
src/agent_memory/integrations/
    __init__.py
    langchain.py       ← AgentMemoryHistory, AgentMemoryFullHistory, build_chain()
```

### Tasks

| # | Task | What it does |
|---|---|---|
| 1 | Create `src/agent_memory/integrations/langchain.py` | Houses `AgentMemoryHistory` and `AgentMemoryFullHistory` as proper importable classes |
| 2 | Fix archival query threading | Pre-step `RunnableLambda` sets `current_input` before history is loaded |
| 3 | Add `build_chain(llm, user_id, config)` factory | Returns a ready `RunnableWithMessageHistory` — one-line setup for consumers |
| 4 | Handle memory command parsing | After `AIMessage` is saved, run `parse_and_apply()` so LLM can still issue `[REMEMBER:]` commands |
| 5 | Add rolling summarisation trigger | After `add_messages()`, check `should_summarize()` and run async summarisation |
| 6 | Add `pyproject.toml` optional dep: `langchain` | `pip install agent-memory[langchain]` |
| 7 | Tests | Mock LangChain LLM, assert history round-trip, assert commands parsed |

### Estimated effort: Phase 8 (after Phase 7 packaging)

---

## Minimal workaround today (no changes needed)

```python
from agent_memory.integrations_draft import AgentMemoryFullHistory

# Before each invoke, pass current input so archival search works:
history = AgentMemoryFullHistory("alice", current_input=user_message)

# Call chain directly (bypassing RunnableWithMessageHistory):
messages = history.messages + [HumanMessage(content=user_message)]
response = await llm.ainvoke(messages)

# Save back
history.add_messages([HumanMessage(content=user_message), AIMessage(content=response.content)])
```

This gives you full 4-layer memory with any LangChain LLM today, at the cost
of manually managing the invoke loop.
