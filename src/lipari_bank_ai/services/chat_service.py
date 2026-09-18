from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.repos import ChatRepository
from lipari_bank_ai.exception.exception import ChatSessionNotFoundError
from lipari_bank_ai.llm.client import LLMProvider
from lipari_bank_ai.llm.types import Message
from lipari_bank_ai.models.chat_model import ChatRequest, ChatResponse
from lipari_bank_ai.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

BALANCE_KEYWORDS = [
    "saldo", "bilancio", "quanto ho", "quanto money", "conta",
    "balance", "soldi sul conto", "disponibilità", "giacenza",
]
TRANSACTIONS_KEYWORDS = [
    "transazioni", "movimenti", "spese", "ultime operazioni",
    "cronologia", "storico", "dove ho speso", "acquisti",
    "transactions", "operazioni",
]


def _detect_intent(message: str) -> str | None:
    lower = message.lower()
    if any(kw in lower for kw in BALANCE_KEYWORDS):
        return "balance"
    if any(kw in lower for kw in TRANSACTIONS_KEYWORDS):
        return "transactions"
    return None


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        llm: LLMProvider,
        system_prompt: str,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.session = session
        self.repo = ChatRepository(session)
        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry

    async def chat(self, req: ChatRequest, user_id: str) -> ChatResponse:
        if req.session_id == "new":
            chat_session = await self.repo.create_session(user_id=user_id)
        else:
            chat_session = await self.repo.find_session(req.session_id)
            if chat_session is None:
                raise ChatSessionNotFoundError(req.session_id)

        history_messages = await self.repo.list_messages(chat_session.id)
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
        ]
        for m in history_messages:
            messages.append(Message(role=m.role, content=m.content))  # type: ignore[arg-type]
        messages.append(Message(role="user", content=req.message))

        await self.repo.add_message(
            session_id=chat_session.id,
            role="user",
            content=req.message,
            tokens=0,
            cost_eur=0.0,
            model_used="pending",
        )

        tool_result_text = ""
        intent = _detect_intent(req.message)
        if intent and self.tool_registry:
            tool_result_text = await self._run_tool(intent, user_id)
            logger.info("Tool '%s' result: %s", intent, tool_result_text[:200])

        if tool_result_text:
            messages.append(Message(
                role="user",
                content=(
                    f"I dati dell'utente sono:\n{tool_result_text}\n\n"
                    "Rispondi in italiano in modo conciso e professionale "
                    "usando questi dati reali. Non inventare altri numeri."
                ),
            ))

        llm_response = await self.llm.complete(messages, max_tokens=500)

        final_content = llm_response.content

        await self.repo.add_message(
            session_id=chat_session.id,
            role="assistant",
            content=final_content,
            tokens=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
            model_used=llm_response.model,
        )

        await self.session.commit()

        return ChatResponse(
            session_id=chat_session.id,
            reply=final_content,
            tokens_used=llm_response.tokens_used,
            cost_eur=llm_response.cost_eur,
            model_used=llm_response.model,
            created_at=datetime.now(UTC),
        )

    async def _run_tool(self, intent: str, user_id: str) -> str:
        if self.tool_registry is None:
            return ""
        try:
            if intent == "balance":
                tool = self.tool_registry.get("get_account_balance")
                if tool:
                    result = await tool.execute({}, user_id)
                    return json.dumps(result, ensure_ascii=False)
            elif intent == "transactions":
                tool = self.tool_registry.get("get_recent_transactions")
                if tool:
                    result = await tool.execute({"limit": 5}, user_id)
                    return json.dumps(result, ensure_ascii=False)
        except Exception:
            logger.exception("Tool execution failed for intent=%s", intent)
        return ""
