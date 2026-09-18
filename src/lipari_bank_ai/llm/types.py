from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """Una richiesta di tool call dal LLM."""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    tokens_used: int
    cost_eur: float
    model: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
