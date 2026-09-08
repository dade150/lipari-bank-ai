from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.repos import ChatRepository
from lipari_bank_ai.exception.exception import ChatSessionNotFoundError
from lipari_bank_ai.models.chat_model import ChatRequest, ChatResponse


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.repo = ChatRepository(session)

    async def chat(self, req: ChatRequest, user_id: str) -> ChatResponse:
        # Trova la sessione esistente oppure creane una nuova.
        if req.session_id == "new":
            chat_session = await self.repo.create_session(user_id=user_id)
        else:
            chat_session = await self.repo.find_session(req.session_id)

            if chat_session is None:
                raise ChatSessionNotFoundError(req.session_id)

        # Salva il messaggio dell'utente.
        await self.repo.add_message(
            session_id=chat_session.id,
            role="user",
            content=req.message,
            tokens=0,
            cost_eur=0.0,
            model_used="dummy",
        )

        # Per il Giorno 3 utilizziamo ancora una risposta dummy.
        # In G4 verrà sostituita dalla chiamata al modello LLM.
        assistant_reply = f"Echo: {req.message}"

        # Salva la risposta dell'assistant.
        await self.repo.add_message(
            session_id=chat_session.id,
            role="assistant",
            content=assistant_reply,
            tokens=10,
            cost_eur=0.0001,
            model_used="dummy",
        )

        return ChatResponse(
            session_id=chat_session.id,
            reply=assistant_reply,
            tokens_used=10,
            cost_eur=0.0001,
            model_used="dummy",
            created_at=datetime.now(UTC),
        )
