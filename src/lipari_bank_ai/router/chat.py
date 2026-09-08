from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.models.chat_model import ChatRequest, ChatResponse
from lipari_bank_ai.db.chat_service import ChatService


router = APIRouter(prefix="/api/ai", tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send message to AI assistant",
    description="Multi-turn conversation with PostgreSQL persistence.",
)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    service = ChatService(db)

    return await service.chat(
        req=req,
        user_id="demo-user",
    )
