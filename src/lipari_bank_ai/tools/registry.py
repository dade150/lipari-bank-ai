from __future__ import annotations

from typing import Any

from lipari_bank_ai.tools.base import Tool


class ToolRegistry:
    """Registra e gestisce i tool disponibili per il LLM."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Converte i tool in formato OpenAI function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def to_prompt_section(self) -> str:
        """Genera una sezione di testo per il system prompt con i tool disponibili."""
        lines = ["## Tools Disponibili"]
        for tool in self._tools.values():
            lines.append(f"\n### {tool.name}")
            lines.append(f"Descrizione: {tool.description}")
            lines.append(f"Parametri: {tool.parameters_schema}")
        return "\n".join(lines)
