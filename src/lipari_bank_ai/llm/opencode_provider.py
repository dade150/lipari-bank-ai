from __future__ import annotations

import json
import re

from opencode_ai import AsyncOpencode
from opencode_ai.types import TextPartInputParam

from lipari_bank_ai.llm.types import LLMResponse, Message, ToolCall


def _parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Estrae tool call dal testo del LLM.

    Cerca blocchi ```tool ... ``` nel testo e li converte in ToolCall.
    Restituisce (testo_pulito, lista_tool_calls).
    """
    tool_calls: list[ToolCall] = []
    pattern = r"```tool\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            tool_calls.append(ToolCall(
                id=data.get("id", f"tc-{len(tool_calls)}"),
                name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return clean_text, tool_calls


class OpencodeProvider:
    PRICING = {
        "opencode-big-pickle": (0.0, 0.0),
    }

    def __init__(self, api_key: str, model: str = "opencode-big-pickle") -> None:
        self.client = AsyncOpencode(base_url="http://localhost:4096")
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 500) -> LLMResponse:
        session = await self.client.session.create()

        parts = [TextPartInputParam(text=m.content, type="text") for m in messages]

        response = await self.client.session.chat(
            id=session.id,
            model_id="opencode-big-pickle",
            provider_id="opencode/big-pickle",
            parts=parts,
        )

        text = next(
            (p["text"] for p in response.parts if p.get("type") == "text"),
            "",
        )

        tokens = response.info.get("tokens", {})
        input_tokens = tokens.get("input", 0)
        output_tokens = tokens.get("output", 0)

        clean_text, tool_calls = _parse_tool_calls(text)

        return LLMResponse(
            content=clean_text,
            tokens_used=input_tokens + output_tokens,
            cost_eur=0.0,
            model=self.model,
            tool_calls=tool_calls,
        )
