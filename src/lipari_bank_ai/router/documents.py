from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.services.ingest_service import IngestService

router = APIRouter(prefix="/api/ai", tags=["Documents"])


class IngestRequest(BaseModel):
    document_id: str = Field(..., max_length=100)
    content: str = Field(..., min_length=10)
    metadata: dict | None = None


class IngestResponse(BaseModel):
    chunk_count: int
    embedding_dim: int


@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    summary="Ingest document into vector store",
)
async def ingest_document(
    req: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    embedding_client = EmbeddingClient()
    service = IngestService(db, embedding_client)
    chunk_count = await service.ingest_document(
        document_id=req.document_id,
        content=req.content,
        metadata=req.metadata,
    )
    return IngestResponse(chunk_count=chunk_count, embedding_dim=embedding_client.dim)
