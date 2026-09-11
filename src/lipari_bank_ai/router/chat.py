from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.chat_service import ChatService
from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.models.chat_model import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/ai", tags=["Chat"])

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "chat_system_v1.md").read_text()


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
    service = ChatService(db, get_llm_provider(), SYSTEM_PROMPT)
    return await service.chat(req, user_id="demo-user")
