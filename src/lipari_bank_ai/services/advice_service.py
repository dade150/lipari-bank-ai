from opencode_ai import AsyncOpencode
from opencode_ai.types import TextPartInputParam

from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse, Citation

ADVICE_SYSTEM = """Sei LipariBank Assistant, esperto di prodotti bancari italiani.

Regole:
- Rispondi SEMPRE in italiano.
- Basa la risposta SOLO sul contesto fornito.
- Se il contesto non contiene informazioni sufficienti, dillo chiaramente.
- Cita sempre le fonti usando [1], [2], ecc.
- Sii conciso: max 3 paragrafi.

Formato risposta:
- Risposta diretta alla domanda
- Citazioni tra parentesi quadre [1], [2]
- Fonti in fondo"""


async def get_advice(req: AdviceRequest, context_chunks: list[dict]) -> AdviceResponse:
    context_text = ""
    citations = []
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[{i}] {chunk['excerpt']}\n"
        citations.append(
            Citation(
                document_id=chunk["document_id"],
                chunk_id=chunk["chunk_id"],
                excerpt=chunk["excerpt"],
                similarity=chunk["similarity"],
            )
        )

    user_msg = f"Contesto:\n{context_text}\n\nDomanda: {req.question}"

    client = AsyncOpencode(base_url="http://localhost:4096")
    session = await client.session.create()

    parts = [TextPartInputParam(text=f"{ADVICE_SYSTEM}\n\n{user_msg}", type="text")]

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

    tokens = response.info.get("tokens", {})
    tokens_used = tokens.get("input", 0) + tokens.get("output", 0)

    return AdviceResponse(
        answer=text,
        citations=citations,
        tokens_used=tokens_used,
        cost_eur=0.0,
    )
