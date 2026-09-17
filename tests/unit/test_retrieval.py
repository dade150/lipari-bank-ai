import pytest
from lipari_bank_ai.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_retrieve_top_k(test_session, mock_embedding):
    # Setup: insert chunks con embedding
    # ...
    retrieval = RetrievalService(test_session, mock_embedding)
    chunks = await retrieval.retrieve("query test", top_k=3)
    assert len(chunks) <= 3
    # similarity decrescente
    for i in range(len(chunks) - 1):
        assert chunks[i].similarity >= chunks[i + 1].similarity