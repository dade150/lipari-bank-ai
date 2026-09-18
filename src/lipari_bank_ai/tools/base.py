from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    """Protocollo per un tool utilizzabile dal LLM."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    async def execute(self, params: dict[str, Any], user_id: str) -> dict[str, Any]: ...
