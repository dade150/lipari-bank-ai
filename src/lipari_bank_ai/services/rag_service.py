from pathlib import Path

from lipari_bank_ai.llm.client import LLMProvider, Message
from lipari_bank_ai.services.retrieval_service import RetrievalService
from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse, Citation

ADVICE_SYSTEM = (Path(__file__).parent.parent / "prompts" / "advice_system_v1.md").read_text()


class RAGService:
    def __init__(self, retrieval: RetrievalService, llm: LLMProvider) -> None:
        self.retrieval = retrieval
        self.llm = llm

    async def answer(self, req: AdviceRequest) -> AdviceResponse:
        # 1. Retrieve top-k chunks
        chunks = await self.retrieval.retrieve(req.question, top_k=5)

        if not chunks:
            return AdviceResponse(
                answer="Non ho documenti correlati alla tua domanda.",
                citations=[],
                tokens_used=0,
                cost_eur=0,
            )

        # 2. Build context string
        context_parts = [
            f"[{c.document_id}, {c.chunk_id}, sim:{c.similarity:.2f}]\n{c.content}"
            for c in chunks
        ]
        context = "\n\n---\n\n".join(context_parts)

        # 3. Generate
        user_prompt = f"""CONTESTI:
{context}

DOMANDA: {req.question}

RISPOSTA (con citazioni):"""

        llm_response = await self.llm.complete(
            messages=[
                Message(role="system", content=ADVICE_SYSTEM),
                Message(role="user", content=user_prompt),
            ],
            max_tokens=800,
        )

        # 4. Build citations from retrieved chunks
        citations = [
            Citation(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                excerpt=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                similarity=c.similarity,
            )
            for c in chunks
        ]

        return AdviceResponse(
            answer=llm_response.content,
            citations=citations,
            tokens_used=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
        )
