"""
Minimal Safe-OS default environment shim.

The upstream Safe-OS data uses `"local": "default"`, but the SQUIRL training loop
only needs environment tool schemas, not actual tool execution. This shim exposes a
single generic `bash` tool so benign Safe-OS samples can enter the same guard /
judge pipeline as unsafe samples.
"""

from __future__ import annotations


class default:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.tool_descs = [
            {
                "name": "bash",
                "description": (
                    "Inspect or read local filesystem state with a shell command. "
                    "Use for tasks like listing files, reading file contents, "
                    "searching directories, or checking simple system settings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "A shell command for read-oriented local inspection, "
                                "for example ls, cat, find, grep, pwd, stat, or echo."
                            ),
                        }
                    },
                    "required": ["command"],
                },
            }
        ]

    def get_tool_descs(self, tool_names):
        if tool_names == ["*"]:
            return self.tool_descs
        requested = set(tool_names)
        return [tool for tool in self.tool_descs if tool["name"] in requested]
