
import pytest

from lipari_bank_ai.exception.exception import ChatSessionNotFoundError
from lipari_bank_ai.models.chat_model import ChatRequest
from lipari_bank_ai.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_chat_creates_new_session(test_session, mock_llm):
    service = ChatService(test_session, mock_llm, "system prompt")

    response = await service.chat(ChatRequest(session_id="new", message="Ciao"), user_id="u1")

    assert response.session_id != "new"
    assert response.reply == "OK"
    mock_llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_finds_existing_session(test_session, mock_llm):
    # Crea session manualmente
    service = ChatService(test_session, mock_llm, "...")
    first = await service.chat(ChatRequest(session_id="new", message="msg1"), "u1")

    # Second message stesso session
    second = await service.chat(
        ChatRequest(session_id=first.session_id, message="msg2"), "u1"
    )

    assert second.session_id == first.session_id


@pytest.mark.asyncio
async def test_chat_raises_on_unknown_session(test_session, mock_llm):
    service = ChatService(test_session, mock_llm, "...")
    with pytest.raises(ChatSessionNotFoundError):
        await service.chat(ChatRequest(session_id="not-existing", message="x"), "u1")
