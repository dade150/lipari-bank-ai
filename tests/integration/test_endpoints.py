# tests/integration/test_endpoints.py
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.session import get_db
from lipari_bank_ai.llm.types import LLMResponse
from lipari_bank_ai.main import app


def _make_opencode_response(category: str, subcategory: str, confidence: float) -> MagicMock:
    data = json.dumps({
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "reasoning": "test reasoning",
    })
    resp = MagicMock()
    resp.parts = [{"type": "text", "text": data}]
    return resp


async def _override_get_db():
    mock = AsyncMock(spec=AsyncSession)
    mock.commit = AsyncMock()

    def _add(obj):
        if hasattr(obj, "id") and obj.id is None:
            obj.id = str(uuid.uuid4())

    mock.add = MagicMock(side_effect=_add)
    mock.flush = AsyncMock()
    mock.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
    yield mock


def _override_llm():
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content="Risposta test dall'assistente.",
        tokens_used=50,
        cost_eur=0.0,
        model="mock",
    ))
    return mock


# ── Health ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "UP"}


# ── Chat ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("lipari_bank_ai.router.chat.get_llm_provider")
async def test_chat_returns_200_with_required_fields(mock_llm_factory):
    mock_llm_factory.return_value = _override_llm()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/ai/chat", json={"session_id": "new", "message": "Ciao"})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert body["session_id"] != "new"
        assert "reply" in body
        assert "tokens_used" in body
        assert "cost_eur" in body
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_chat_returns_422_on_missing_message():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/ai/chat", json={"session_id": "new"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "VALIDATION_ERROR"


# ── Categorize ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_returns_200(mock_cls):
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s1"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "UTILITIES", "ENERGY", 0.95
    ))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/ai/categorize", json={
            "description": "Enel bolletta luce",
            "amount": 85.0,
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "UTILITIES"
    assert body["confidence"] == 0.95


@pytest.mark.asyncio
async def test_categorize_returns_422_on_invalid_amount():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/ai/categorize", json={
            "description": "Test",
            "amount": -10,
        })
    assert resp.status_code == 422


# ── Documents / Ingest ──────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("lipari_bank_ai.router.documents.EmbeddingClient")
async def test_ingest_returns_200(mock_embedding_cls):
    mock_inst = mock_embedding_cls.return_value
    mock_inst.dim = 384
    mock_inst.embed = AsyncMock(return_value=[[0.0] * 384])

    app.dependency_overrides[get_db] = _override_get_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/ai/documents/ingest",
                json={
                    "document_id": "test-doc",
                    "content": (
                        "Questo e un documento di test lungo "
                        "abbastanza per essere processato."
                    ),
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "chunk_count" in body
        assert body["embedding_dim"] == 384
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_ingest_returns_422_on_short_content():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/ai/documents/ingest", json={
            "document_id": "x",
            "content": "short",
        })
    assert resp.status_code == 422


# ── Cost Report ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cost_report_returns_200():
    app.dependency_overrides[get_db] = _override_get_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/admin/cost-report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["period"] == "last_24h"
        assert isinstance(body["by_model"], list)
    finally:
        app.dependency_overrides.pop(get_db, None)
