import pytest
from httpx import AsyncClient, ASGITransport

from lipari_bank_ai.main import app


@pytest.mark.asyncio
async def test_chat_endpoint_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/ai/chat",
            json={"session_id": "new", "message": "Ciao"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "reply" in body
        assert "session_id" in body