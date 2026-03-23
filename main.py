"""
main.py  —  Infinite Context Chat CLI

Run:
    python main.py                     # default user
    python main.py --user alice        # named user (separate memory)
    python main.py --user alice --debug  # show token stats each turn
    python main.py --user alice --reset  # wipe memory and start fresh

Memory layers active:
  Layer 0: SQLite raw log (never deleted)
  Layer 1: Core memory  (always in context)
  Layer 2: Rolling summary (compresses old turns)
  Layer 3: Sliding window (recent N turns verbatim)
  Layer 4: Archival memory (semantic search over all history)
"""

import asyncio
import argparse
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from agent_memory.config import MemoryConfig
from ollama_client import check_ollama
from chat_engine import process_message
from agent_memory.layers import core, summary, conversation

console = Console()


def print_banner(user_id: str, config: MemoryConfig = None):
    config = config or MemoryConfig()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]∞ Infinite Context Chat[/bold cyan]\n"
        f"[dim]User: [bold]{user_id}[/bold]  |  Model: {config.model}[/dim]\n"
        "[dim]Type [bold white]exit[/bold white] to quit  |  "
        "[bold white]/memory[/bold white] to inspect memory  |  "
        "[bold white]/reset[/bold white] to wipe[/dim]",
        border_style="cyan"
    ))
    console.print()


def print_memory_state(user_id: str):
    """Show current state of all memory layers."""
    console.print(Rule("[bold yellow]Memory State[/bold yellow]"))
    
    # Core memory
    data = core.load(user_id)
    console.print(f"\n[bold cyan]Layer 1 — Core Memory[/bold cyan]")
    console.print(f"  Name:   {data['user_name']}")
    console.print(f"  Facts:  {data['user_facts'] or '(none)'}")
    console.print(f"  Scratch: {data['scratch'] or '(empty)'}")

    # Summary
    s = summary.load(user_id)
    console.print(f"\n[bold magenta]Layer 2 — Rolling Summary[/bold magenta]")
    if s['summary']:
        console.print(f"  [dim](covers turns 1–{s['turn_count']})[/dim]")
        console.print(f"  {s['summary'][:200]}{'...' if len(s['summary']) > 200 else ''}")
    else:
        console.print("  [dim](no summary yet — triggers after 15 turns)[/dim]")

    # Message count
    total = conversation.get_message_count(user_id)
    console.print(f"\n[bold green]Layer 3 — Sliding Window[/bold green]")
    console.print(f"  Total messages in log: {total}")
    
    console.print()
    console.print(Rule())


def reset_user(user_id: str):
    """Wipe all memory for a user."""
    db_path = Path("data/conversations.db")
    chroma_path = Path("data/chroma")
    
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        safe = "".join(c if c.isalnum() else "_" for c in user_id)
        conn.execute(f"DROP TABLE IF EXISTS conv_{safe}")
        conn.execute("DELETE FROM core_memory WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM summaries WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
    
    # ChromaDB reset
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        safe = "".join(c if c.isalnum() else "_" for c in user_id)
        client.delete_collection(f"memory_{safe}")
    except Exception:
        pass
    
    console.print(f"[yellow]⚠ Memory wiped for user '{user_id}'[/yellow]")


async def chat_loop(user_id: str, debug: bool = False):
    config = MemoryConfig.from_env()
    print_banner(user_id, config)

    # Check Ollama
    console.print("[dim]Checking Ollama...[/dim]", end=" ")
    ok = await check_ollama(config)
    if not ok:
        console.print(f"[red]✗[/red]")
        console.print(
            f"[red bold]Ollama not running or {config.model} not found.[/red bold]\n"
            f"Run: [bold white]ollama pull {config.model}[/bold white]\n"
            "Then: [bold white]OLLAMA_NUM_PARALLEL=2 ollama serve[/bold white]"
        )
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

        # Built-in commands
        if user_input.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower() == "/memory":
            print_memory_state(user_id)
            continue

        if user_input.lower() == "/reset":
            reset_user(user_id)
            continue

        if user_input.lower() == "/help":
            console.print(
                "[dim]Commands: exit | /memory | /reset | /help[/dim]\n"
                "[dim]Memory cmds LLM can use: [REMEMBER: fact] [NOTE: text] [NAME: name][/dim]"
            )
            continue

        # Process through memory pipeline
        console.print(f"\n[bold green]Assistant:[/bold green] ", end="")

        try:
            cleaned_response, memory_actions = await process_message(
                user_id=user_id,
                user_message=user_input,
                stream=True,
                show_stats=debug,
                config=config,
            )
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue

        # Show memory actions if any were triggered
        if memory_actions:
            console.print()
            for action in memory_actions:
                console.print(f"  [dim]{action}[/dim]")

        console.print()


def main():
    parser = argparse.ArgumentParser(description="Infinite Context Chat")
    parser.add_argument("--user",  default="default", help="User ID (separate memory per user)")
    parser.add_argument("--debug", action="store_true", help="Show token budget stats")
    parser.add_argument("--reset", action="store_true", help="Wipe memory for this user and exit")
    args = parser.parse_args()

    if args.reset:
        reset_user(args.user)
        return

    asyncio.run(chat_loop(args.user, debug=args.debug))


if __name__ == "__main__":
    main()
