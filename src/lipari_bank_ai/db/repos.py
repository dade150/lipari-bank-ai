# lipari_bank_ai/db/repos.py
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lipari_bank_ai.db.models import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, user_id: str) -> ChatSession:
        chat_session = ChatSession(user_id=user_id)
        self.session.add(chat_session)
        await self.session.flush()
        return chat_session

    async def find_session(self, session_id: str) -> ChatSession | None:
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int,
        cost_eur: float,
        model_used: str,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tokens=tokens,
            cost_eur=cost_eur,
            model_used=model_used,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def list_messages(self, session_id: str) -> Sequence[ChatMessage]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_user_sessions(self, user_id: str) -> int:
        stmt = select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()
