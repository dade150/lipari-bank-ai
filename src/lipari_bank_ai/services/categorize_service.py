import json

from opencode_ai import AsyncOpencode
from opencode_ai.types import TextPartInputParam

from lipari_bank_ai.models.categorize_model import CategorizeRequest, CategorizeResponse

CATEGORIZE_SYSTEM = """You are an expert at categorizing Italian bank transactions.

Categories:
- UTILITIES (luce, gas, acqua, internet, telefono)
- GROCERIES (supermercati, alimentari, market)
- TRANSPORT (carburante, treno, mezzi, parcheggi)
- RESTAURANTS (ristoranti, bar, fast food)
- ENTERTAINMENT (cinema, palestra, abbonamenti streaming)
- OTHER (tutto il resto)

Subcategory: specifica più precisa in italiano (es. "ENERGY", "SUPERMARKET", "FUEL").
Confidence: tua sicurezza 0.0-1.0.
Reasoning: 1-2 frasi spiegando la scelta.

Return ONLY valid JSON with keys: category, subcategory, confidence, reasoning."""


async def categorize(req: CategorizeRequest) -> CategorizeResponse:
    client = AsyncOpencode(base_url="http://localhost:4096")
    session = await client.session.create()

    user_msg = f"Description: {req.description}\nAmount: \u20ac{req.amount} {req.currency}"

    parts = [
        TextPartInputParam(text=f"{CATEGORIZE_SYSTEM}\n\n{user_msg}", type="text"),
    ]

    response = await client.session.chat(
        id=session.id,
        model_id="opencode-big-pickle",
        provider_id="opencode/big-pickle",
        parts=parts,
    )

    text = next(
        (p["text"] for p in response.parts if p.get("type") == "text"),
        "",
    )

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    data = json.loads(text)
    return CategorizeResponse(**data)
