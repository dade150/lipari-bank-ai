# tests/unit/test_rag_service.py
from unittest.mock import AsyncMock

import pytest

from lipari_bank_ai.llm.types import LLMResponse
from lipari_bank_ai.services.rag_service import RAGService
from lipari_bank_ai.services.retrieval_service import RetrievalResult
from lipari_bank_ai.types.advice import AdviceRequest


def _fake_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="commissioni_bonifico",
            content="Bonifici SEPA: commissione standard 0.50 EUR",
            similarity=0.62,
            metadata={},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="regolamento_conti",
            content="Conto Corrente: canone mensile 5 EUR",
            similarity=0.58,
            metadata={},
        ),
    ]


@pytest.mark.asyncio
async def test_rag_answer_with_chunks():
    """RAG con chunks trovati: risposta + citazioni."""
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=_fake_chunks())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Per un bonifico SEPA la commissione e 0.50 EUR.",
        tokens_used=120,
        cost_eur=0.0,
        model="mock",
    ))

    service = RAGService(mock_retrieval, mock_llm)
    result = await service.answer(AdviceRequest(question="Commissioni bonifico SEPA?"))

    assert "0.50" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].document_id == "commissioni_bonifico"
    assert result.citations[1].document_id == "regolamento_conti"
    assert result.tokens_used == 120
    mock_retrieval.retrieve.assert_awaited_once_with("Commissioni bonifico SEPA?", top_k=5)


@pytest.mark.asyncio
async def test_rag_answer_no_chunks():
    """RAG senza chunks: risposta di fallback, nessuna citazione."""
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=[])

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock()

    service = RAGService(mock_retrieval, mock_llm)
    result = await service.answer(AdviceRequest(question="Qualcosa di non presente nei documenti?"))

    assert "documenti correlati" in result.answer.lower() or "non ho" in result.answer.lower()
    assert result.citations == []
    assert result.tokens_used == 0
    mock_llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_builds_correct_context():
    """RAG costruisce il contesto corretto per il prompt."""
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=_fake_chunks())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Risposta test",
        tokens_used=50,
        cost_eur=0.0,
        model="mock",
    ))

    service = RAGService(mock_retrieval, mock_llm)
    await service.answer(AdviceRequest(question="Domanda test?"))

    call_args = mock_llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args[0][0]

    user_msg = [m for m in messages if m.role == "user"][0]
    assert "commissioni_bonifico" in user_msg.content
    assert "regolamento_conti" in user_msg.content
    assert "Domanda test?" in user_msg.content
    assert "CONTESTI:" in user_msg.content


@pytest.mark.asyncio
async def test_rag_citations_have_excerpt():
    """Le citazioni contengono un excerpt troncato a 200 char."""
    long_content = "x" * 300
    chunk = RetrievalResult(
        chunk_id="c1",
        document_id="doc1",
        content=long_content,
        similarity=0.7,
        metadata={},
    )

    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=[chunk])

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="ok", tokens_used=10, cost_eur=0.0, model="mock"
    ))

    service = RAGService(mock_retrieval, mock_llm)
    result = await service.answer(AdviceRequest(question="Test excerpt?"))

    assert len(result.citations) == 1
    assert len(result.citations[0].excerpt) <= 203  # 200 + "..."
    assert result.citations[0].excerpt.endswith("...")
