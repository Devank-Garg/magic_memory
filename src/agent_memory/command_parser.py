"""
command_parser.py

Parses memory update commands from LLM responses and applies them.

Commands the LLM can issue (at end of response):
  [REMEMBER: <fact>]   → saved to core memory user_facts
  [NOTE: <text>]       → saved to core memory scratch
  [NAME: <name>]       → saved as user_name in core memory
"""

import re
from agent_memory.layers import core
from agent_memory.types import MemoryAction


COMMAND_PATTERN = re.compile(
    r'\[(REMEMBER|NOTE|NAME):\s*(.+?)\]',
    re.IGNORECASE
)

# Matches the command wrapped in markdown bold/italic (e.g. **[NAME: x]** or *[NOTE: y]*)
# Used only for stripping — capturing is done by COMMAND_PATTERN above
_MARKDOWN_COMMAND_PATTERN = re.compile(
    r'\*{1,2}\[(REMEMBER|NOTE|NAME):\s*.+?\]\*{1,2}',
    re.IGNORECASE
)


def parse_and_apply(user_id: str, response_text: str) -> tuple[str, list[MemoryAction]]:
    """
    Extract and apply memory commands from LLM response.

    Returns:
      - cleaned_response: response with commands stripped out
      - actions: list of MemoryAction describing what was stored
    """
    actions: list[MemoryAction] = []
    commands = COMMAND_PATTERN.findall(response_text)

    for cmd_type, cmd_value in commands:
        cmd_type  = cmd_type.upper().strip()
        cmd_value = cmd_value.strip()

        if cmd_type == "REMEMBER":
            core.update_fact(user_id, cmd_value)
            actions.append(MemoryAction(type="remember", value=cmd_value))

        elif cmd_type == "NOTE":
            core.update_scratch(user_id, cmd_value)
            actions.append(MemoryAction(type="note", value=cmd_value))

        elif cmd_type == "NAME":
            core.set_user_name(user_id, cmd_value)
            actions.append(MemoryAction(type="name", value=cmd_value))

    # Strip markdown-wrapped variants first (**[CMD: ...]**), then bare ones
    cleaned = _MARKDOWN_COMMAND_PATTERN.sub("", response_text)
    cleaned = COMMAND_PATTERN.sub("", cleaned).strip()

    return cleaned, actions
