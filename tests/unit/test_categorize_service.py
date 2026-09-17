# tests/unit/test_categorize_service.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lipari_bank_ai.models.categorize_model import CategorizeRequest
from lipari_bank_ai.services.categorize_service import categorize


def _make_opencode_response(
    category: str, subcategory: str, confidence: float, reasoning: str
) -> MagicMock:
    """Crea una fake response AsyncOpencode con JSON parsabile."""
    data = json.dumps({
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    resp = MagicMock()
    resp.parts = [{"type": "text", "text": data}]
    return resp


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_utilities_energy(mock_cls):
    """Caso 1: Categoria UTILITIES con successo."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s1"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "UTILITIES", "ENERGY", 0.92, "enel keyword"
    ))

    result = await categorize(CategorizeRequest(description="Enel bolletta", amount=85.0))

    assert result.category == "UTILITIES"
    assert result.subcategory == "ENERGY"
    assert result.confidence == 0.92
    mock_instance.session.chat.assert_awaited_once()


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_restaurants(mock_cls):
    """Caso 2: Categoria RESTAURANTS."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s2"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "RESTAURANTS", "TRATTORIA", 0.98, "pizzeria keyword"
    ))

    result = await categorize(CategorizeRequest(description="Pizzeria da Mario", amount=35.0))

    assert result.category == "RESTAURANTS"
    assert result.subcategory == "TRATTORIA"
    assert result.confidence == 0.98


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_low_confidence(mock_cls):
    """Caso 3: Confidence bassa su transazione ambigua."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s3"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "OTHER", "UNKNOWN", 0.35, "Description is too vague"
    ))

    result = await categorize(CategorizeRequest(description="XYZ generico", amount=12.0))

    assert result.category == "OTHER"
    assert result.confidence < 0.5


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_groceries(mock_cls):
    """Caso 4: Categoria GROCERIES."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s4"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "GROCERIES", "SUPERMARKET", 0.95, "coop keyword"
    ))

    result = await categorize(CategorizeRequest(description="Coop supermercato", amount=62.5))

    assert result.category == "GROCERIES"
    assert result.confidence >= 0.9


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_strips_markdown_code_block(mock_cls):
    """Caso 5: LLM wrappa la risposta in ```json ... ```."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s5"))

    wrapped = (
        '```json\n{"category":"TRANSPORT","subcategory":"FUEL",'
        '"confidence":0.88,"reasoning":"eni keyword"}\n```'
    )
    resp = MagicMock()
    resp.parts = [{"type": "text", "text": wrapped}]
    mock_instance.session.chat = AsyncMock(return_value=resp)

    result = await categorize(CategorizeRequest(description="Eni carburante", amount=45.0))

    assert result.category == "TRANSPORT"
    assert result.subcategory == "FUEL"


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_provider_error(mock_cls):
    """Caso 6: Eccezione dal provider LLM."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s6"))
    mock_instance.session.chat = AsyncMock(side_effect=Exception("LLM Provider Timeout"))

    with pytest.raises(Exception, match="LLM Provider Timeout"):
        await categorize(CategorizeRequest(description="Test", amount=10.0))


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_invalid_json_response(mock_cls):
    """Caso 7: LLM restituisce JSON non valido."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s7"))

    resp = MagicMock()
    resp.parts = [{"type": "text", "text": "not valid json at all"}]
    mock_instance.session.chat = AsyncMock(return_value=resp)

    with pytest.raises(json.JSONDecodeError):
        await categorize(CategorizeRequest(description="Test", amount=10.0))


@pytest.mark.asyncio
@patch("lipari_bank_ai.services.categorize_service.AsyncOpencode")
async def test_categorize_entertainment(mock_cls):
    """Caso 8: Categoria ENTERTAINMENT."""
    mock_instance = mock_cls.return_value
    mock_instance.session.create = AsyncMock(return_value=MagicMock(id="s8"))
    mock_instance.session.chat = AsyncMock(return_value=_make_opencode_response(
        "ENTERTAINMENT", "STREAMING", 0.97, "netflix keyword"
    ))

    result = await categorize(CategorizeRequest(description="Netflix abbonamento", amount=15.99))

    assert result.category == "ENTERTAINMENT"
    assert result.subcategory == "STREAMING"
