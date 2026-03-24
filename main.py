"""Thin shim — delegates to the installed agent_memory.cli entry point."""
from agent_memory.cli import main

if __name__ == "__main__":
    main()
