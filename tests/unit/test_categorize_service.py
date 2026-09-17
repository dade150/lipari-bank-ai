# tests/unit/test_categorize_service.py
import pytest
from unittest.mock import AsyncMock, patch

from lipari_bank_ai.models.categorize_model import CategorizeRequest, CategorizeResponse
from lipari_bank_ai.services.categorize_service import categorize


@pytest.mark.asyncio
async def test_categorize_calls_llm_with_correct_prompt():
    """Caso 1: Categoria standard (UTILITIES) con successo."""
    fake_response = CategorizeResponse(
        category="UTILITIES",
        subcategory="ENERGY",
        confidence=0.92,
        reasoning="enel keyword",
    )

    with patch("src.services.categorize_service.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        result = await categorize(CategorizeRequest(description="Enel bolletta", amount=85.0))

        assert result.category == "UTILITIES"
        assert result.confidence == 0.92
        mock_client.chat.completions.create.assert_awaited_once()

        # Verifica che la descrizione dell'utente sia finita nel prompt inviato all'LLM
        call_args = mock_client.chat.completions.create.call_args
        content_sent = call_args.kwargs["messages"][1]["content"]
        assert "Enel" in content_sent


@pytest.mark.asyncio
async def test_categorize_different_category():
    """Caso 2: Categoria diversa (es. ristoranti/cibo)."""
    fake_response = CategorizeResponse(
        category="FOOD",
        subcategory="RESTAURANT",
        confidence=0.98,
        reasoning="Pizzeria keyword",
    )

    with patch("src.services.categorize_service.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        result = await categorize(CategorizeRequest(description="Pizzeria da Mario", amount=35.0))

        assert result.category == "FOOD"
        assert result.subcategory == "RESTAURANT"
        mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_categorize_low_confidence():
    """Caso 3: Risposta con confidence bassa (es. transazione ambigua)."""
    fake_response = CategorizeResponse(
        category="MISC",
        subcategory="UNKNOWN",
        confidence=0.45,
        reasoning="Description is too vague to be sure",
    )

    with patch("src.services.categorize_service.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        result = await categorize(CategorizeRequest(description="Pagamento generico XYZ", amount=12.0))

        assert result.category == "MISC"
        assert result.confidence < 0.5  # Verifica che la confidence bassa venga restituita correttamente
        mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_categorize_provider_error():
    """Caso 4: Errore del provider LLM (es. timeout o eccezione API)."""
    with patch("src.services.categorize_service.client") as mock_client:
        # Simula un'eccezione lanciata dal client LLM (es. errore di rete o API down)
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("LLM Provider Timeout"))

        with pytest.raises(Exception, match="LLM Provider Timeout"):
            await categorize(CategorizeRequest(description="Transazione test", amount=10.0))

        mock_client.chat.completions.create.assert_awaited_once()