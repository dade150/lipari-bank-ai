# tests/unit/test_retrieval.py
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lipari_bank_ai.db.models import DocumentChunk
from lipari_bank_ai.services.retrieval_service import RetrievalService


def _embedding(values: list[float]) -> list[float]:
    """Helper: normalizza un vettore per cosine similarity."""
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


async def _insert_chunks(session, chunks_data: list[tuple[str, str, int, str, list[float]]]):
    """Inserisce document_chunks nel DB di test."""
    for doc_id, chunk_idx, idx, content, emb in chunks_data:
        db_chunk = DocumentChunk(
            id=f"chunk-{doc_id}-{idx}",
            document_id=doc_id,
            chunk_index=chunk_idx,
            content=content,
            embedding=emb,
            chunk_metadata={},
        )
        session.add(db_chunk)
    await session.commit()


@pytest.mark.asyncio
async def test_retrieve_returns_top_k(test_session, mock_embedding):
    """Verifica che retrieve restituisca al massimo top_k risultati."""
    emb_similar = _embedding([1.0, 0.0, 0.0] + [0.0] * 381)
    emb_different = _embedding([0.0, 0.0, 1.0] + [0.0] * 381)

    await _insert_chunks(test_session, [
        ("doc-a", 0, 0, "Conto Corrente: canone mensile 5 EUR", emb_similar),
        ("doc-a", 1, 1, "Bonifici SEPA: commissione 0.50 EUR", emb_similar),
        ("doc-b", 0, 2, "Carta Classic: canone annuale 20 EUR", emb_different),
        ("doc-b", 1, 3, "Mutuo tasso fisso: TAEG 3.5%", emb_different),
        ("doc-c", 0, 4, "Fondo pensione: rendimento annuo", emb_different),
    ])

    mock_embedding.embed_one = AsyncMock(return_value=emb_similar)

    service = RetrievalService(test_session, mock_embedding)
    results = await service.retrieve("conto corrente", top_k=3)

    assert len(results) <= 3
    assert all(r.similarity >= 0 for r in results)


@pytest.mark.asyncio
async def test_retrieve_similarity_ordering(test_session, mock_embedding):
    """Verifica che i risultati siano ordinati per similarità decrescente."""
    emb_a = _embedding([1.0, 0.0, 0.0] + [0.0] * 381)
    emb_b = _embedding([0.5, 0.5, 0.0] + [0.0] * 381)
    emb_c = _embedding([0.0, 0.0, 1.0] + [0.0] * 381)

    await _insert_chunks(test_session, [
        ("doc-a", 0, 0, "Testo identico al query", emb_a),
        ("doc-b", 0, 1, "Testo parzialmente simile", emb_b),
        ("doc-c", 0, 2, "Testo completamente diverso", emb_c),
    ])

    mock_embedding.embed_one = AsyncMock(return_value=emb_a)

    service = RetrievalService(test_session, mock_embedding)
    results = await service.retrieve("test query", top_k=3)

    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].similarity >= results[i + 1].similarity


@pytest.mark.asyncio
async def test_retrieve_empty_when_no_chunks(test_session, mock_embedding):
    """Verifica che retrieve ritorni lista vuota se non ci sono chunks."""
    mock_embedding.embed_one = AsyncMock(return_value=[0.0] * 384)

    service = RetrievalService(test_session, mock_embedding)
    results = await service.retrieve("query vuota", top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_retrieve_returns_retrieval_result_fields(test_session, mock_embedding):
    """Verifica che i campi di RetrievalResult siano popolati correttamente."""
    emb = _embedding([1.0, 0.0, 0.0] + [0.0] * 381)

    await _insert_chunks(test_session, [
        ("commissioni", 0, 0, "Bonifico SEPA: 0.50 EUR", emb),
    ])

    mock_embedding.embed_one = AsyncMock(return_value=emb)

    service = RetrievalService(test_session, mock_embedding)
    results = await service.retrieve("bonifico", top_k=1)

    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "chunk-commissioni-0"
    assert r.document_id == "commissioni"
    assert "Bonifico SEPA" in r.content
    assert r.similarity >= 0.99
    assert isinstance(r.metadata, dict)
