"""
cli.py  —  Infinite Context Chat CLI entry point

Run:
    agent-memory                                        # Ollama, default user
    agent-memory --user alice                           # named user
    agent-memory --user alice --debug                   # show token stats
    agent-memory --user alice --reset                   # wipe memory and exit

    agent-memory --provider openai --api-key sk-...
    agent-memory --provider anthropic --api-key ant-...
    agent-memory --provider ollama --model llama3.2     # different Ollama model

    # API keys can also come from env vars:
    # OPENAI_API_KEY=sk-...  agent-memory --provider openai
    # ANTHROPIC_API_KEY=ant-...  agent-memory --provider anthropic

Memory layers active:
  Layer 0: SQLite raw log (never deleted)
  Layer 1: Core memory  (always in context)
  Layer 2: Rolling summary (compresses old turns)
  Layer 3: Sliding window (recent N turns verbatim)
  Layer 4: Archival memory (semantic search over all history)
"""

import asyncio
import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from agent_memory.config import MemoryConfig
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.chroma_store import ChromaStore
from agent_memory.providers.base import BaseLLMProvider
from agent_memory.chat_engine import process_message
from agent_memory.layers import core, summary, conversation

console = Console()


# ── provider factory ────────────────────────────────────────────────────────

def _build_provider(provider_name: str, api_key: str, model: str | None, config: MemoryConfig) -> BaseLLMProvider:
    name = provider_name.lower()

    if name == "ollama":
        from agent_memory.providers import OllamaProvider
        return OllamaProvider(config=config)

    if name == "openai":
        from agent_memory.providers.openai import OpenAIProvider
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        m = model or "gpt-4o"
        return OpenAIProvider(api_key=key, model=m, config=config)

    if name == "anthropic":
        from agent_memory.providers.anthropic import AnthropicProvider
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        m = model or "claude-sonnet-4-6"
        return AnthropicProvider(api_key=key, model=m, config=config)

    console.print(f"[red]Unknown provider '{provider_name}'. Choose: ollama, openai, anthropic[/red]")
    raise SystemExit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def _seed_user_name(user_id: str) -> None:
    """If no name is stored yet, initialise it from the user_id."""
    if user_id == "default":
        return
    data = core.load(user_id)
    if data["user_name"] == "User":
        core.set_user_name(user_id, user_id.capitalize())


def print_banner(user_id: str, provider_name: str, model_label: str) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]∞ Infinite Context Chat[/bold cyan]\n"
        f"[dim]User: [bold]{user_id}[/bold]  |  Provider: {provider_name}  |  Model: {model_label}[/dim]\n"
        "[dim]Type [bold white]exit[/bold white] to quit  |  "
        "[bold white]/memory[/bold white] to inspect memory  |  "
        "[bold white]/reset[/bold white] to wipe[/dim]",
        border_style="cyan"
    ))
    console.print()


def print_memory_state(user_id: str) -> None:
    console.print(Rule("[bold yellow]Memory State[/bold yellow]"))

    data = core.load(user_id)
    console.print(f"\n[bold cyan]Layer 1 — Core Memory[/bold cyan]")
    console.print(f"  Name:    {data['user_name']}")
    console.print(f"  Facts:   {data['user_facts'] or '(none)'}")
    console.print(f"  Scratch: {data['scratch'] or '(empty)'}")

    s = summary.load(user_id)
    console.print(f"\n[bold magenta]Layer 2 — Rolling Summary[/bold magenta]")
    if s['summary']:
        console.print(f"  [dim](covers turns 1–{s['turn_count']})[/dim]")
        console.print(f"  {s['summary'][:200]}{'...' if len(s['summary']) > 200 else ''}")
    else:
        console.print("  [dim](no summary yet — triggers after 15 turns)[/dim]")

    total = conversation.get_message_count(user_id)
    console.print(f"\n[bold green]Layer 3 — Sliding Window[/bold green]")
    console.print(f"  Total messages in log: {total}")

    console.print()
    console.print(Rule())


def reset_user(user_id: str, config: MemoryConfig) -> None:
    SQLiteStore(config.db_path).delete_user(user_id)
    ChromaStore(config.chroma_path).delete_collection(user_id)
    console.print(f"[yellow]⚠ Memory wiped for user '{user_id}'[/yellow]")


# ── main loop ────────────────────────────────────────────────────────────────

async def chat_loop(user_id: str, provider_name: str, api_key: str, model: str | None, debug: bool) -> None:
    config = MemoryConfig.from_env()
    _seed_user_name(user_id)

    try:
        provider = _build_provider(provider_name, api_key, model, config)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        return

    model_label = model or {"ollama": config.model, "openai": "gpt-4o", "anthropic": "claude-sonnet-4-6"}.get(provider_name, provider_name)
    print_banner(user_id, provider_name, model_label)

    console.print(f"[dim]Checking {provider_name}...[/dim]", end=" ")
    ok = await provider.health_check()
    if not ok:
        console.print("[red]✗[/red]")
        _print_health_hint(provider_name, config)
        return
    console.print("[green]✓ Connected[/green]")
    console.print()

    while True:
        try:
            user_input = console.input("[bold white]You:[/bold white] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/memory":
            print_memory_state(user_id)
            continue

        if user_input.lower() == "/reset":
            reset_user(user_id, config)
            continue

        if user_input.lower() == "/help":
            console.print(
                "[dim]Commands: exit | /memory | /reset | /help[/dim]\n"
                "[dim]Memory cmds LLM can use: [REMEMBER: fact] [NOTE: text] [NAME: name][/dim]"
            )
            continue

        console.print(f"\n[bold green]Assistant:[/bold green] ", end="")

        try:
            cleaned_response, memory_actions = await process_message(
                user_id=user_id,
                user_message=user_input,
                stream=True,
                show_stats=debug,
                config=config,
                provider=provider,
            )
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue

        if memory_actions:
            console.print()
            for action in memory_actions:
                icon = {"remember": "📌", "note": "📝", "name": "👤"}.get(action.type, "•")
                console.print(f"  [dim]{icon} {action}[/dim]")

        console.print()


def _print_health_hint(provider_name: str, config: MemoryConfig) -> None:
    if provider_name == "ollama":
        console.print(
            f"[red bold]Ollama not running or model not found.[/red bold]\n"
            f"Run: [bold white]ollama pull {config.model}[/bold white]\n"
            "Then: [bold white]OLLAMA_NUM_PARALLEL=2 ollama serve[/bold white]"
        )
    elif provider_name == "openai":
        console.print("[red bold]OpenAI connection failed. Check your API key and network.[/red bold]")
    elif provider_name == "anthropic":
        console.print("[red bold]Anthropic connection failed. Check your API key and network.[/red bold]")


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Infinite Context Chat")
    parser.add_argument("--user",     default="default",  help="User ID (separate memory per user)")
    parser.add_argument("--provider", default="ollama",   help="LLM provider: ollama | openai | anthropic")
    parser.add_argument("--api-key",  default="",         help="API key for openai / anthropic (or set env var)")
    parser.add_argument("--model",    default=None,       help="Model name override (e.g. gpt-4o-mini, llama3.2)")
    parser.add_argument("--debug",    action="store_true", help="Show token budget stats each turn")
    parser.add_argument("--reset",    action="store_true", help="Wipe memory for this user and exit")
    args = parser.parse_args()

    config = MemoryConfig.from_env()

    if args.reset:
        reset_user(args.user, config)
        return

    asyncio.run(chat_loop(
        user_id=args.user,
        provider_name=args.provider,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
