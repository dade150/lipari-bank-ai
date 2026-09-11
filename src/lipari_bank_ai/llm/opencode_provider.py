from opencode_ai import AsyncOpencode
from opencode_ai.types import TextPartInputParam

from lipari_bank_ai.llm.types import LLMResponse, Message


class OpencodeProvider:
    PRICING = {
        "opencode-big-pickle": (0.0, 0.0),  # Big Pickle è gratuito
    }

    def __init__(self, api_key: str, model: str = "opencode-big-pickle") -> None:
        self.client = AsyncOpencode(
            base_url="http://localhost:4096"
        )
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

        # print(f"{response}")

        text = next(
            (p["text"] for p in response.parts if p.get("type") == "text"),
            "",
        )

        # Questi al momento sono Any. Vanno modellati e validati
        tokens = response.info.get("tokens", {})
        input_tokens = tokens.get("input", 0)
        output_tokens = tokens.get("output", 0)

        return LLMResponse(
            content=text,
            tokens_used=input_tokens + output_tokens,
            cost_eur=0.0,
            model=self.model,
        )
