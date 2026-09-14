from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.services.rag_service import RAGService
from lipari_bank_ai.services.retrieval_service import RetrievalService
from lipari_bank_ai.types.advice import AdviceRequest, AdviceResponse

router = APIRouter(prefix="/api/ai", tags=["Advice"])


@router.post(
    "/advice",
    response_model=AdviceResponse,
    summary="Get banking advice with RAG",
    description="Retrieval Augmented Generation con citazioni.",
)
async def advice_endpoint(
    req: AdviceRequest,
    db: AsyncSession = Depends(get_db),
) -> AdviceResponse:
    embedding_client = EmbeddingClient()
    retrieval = RetrievalService(db, embedding_client)
    llm = get_llm_provider()
    rag = RAGService(retrieval, llm)
    return await rag.answer(req)
