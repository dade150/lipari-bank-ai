from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.factory import get_llm_provider
from lipari_bank_ai.models.chat_model import ChatRequest, ChatResponse
from lipari_bank_ai.services.chat_service import ChatService
from lipari_bank_ai.tools.account import AccountTool
from lipari_bank_ai.tools.registry import ToolRegistry
from lipari_bank_ai.tools.transactions import TransactionTool

router = APIRouter(prefix="/api/ai", tags=["Chat"])

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "chat_system_v1.md").read_text()


def _build_tool_registry(session: AsyncSession) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(AccountTool(session))
    registry.register(TransactionTool(session))
    return registry


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send message to AI assistant",
    description="Multi-turn conversation with PostgreSQL persistence + tool use.",
)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    registry = _build_tool_registry(db)
    service = ChatService(db, get_llm_provider(), SYSTEM_PROMPT, tool_registry=registry)
    return await service.chat(req, user_id="u1")
