# tests/conftest.py
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from lipari_bank_ai.llm.types import LLMResponse
from lipari_bank_ai.db.session import Base


# URL del database di test isolato (porta 5433 e nome con _test per sicurezza)
TEST_DB_URL = "postgresql+asyncpg://lipari:lipari@localhost:5438/lipari_ai_test"


@pytest.fixture
def mock_llm():
    """Mock LLM provider; ritorna 'OK' di default."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content="OK",
        tokens_used=10,
        cost_eur=0.0001,
        model="mock",
    ))
    return mock


@pytest.fixture
def mock_embedding():
    """Mock embedding provider; ritorna un vettore di 1536 zeri."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[[0.0] * 1536])
    mock.embed_one = AsyncMock(return_value=[0.0] * 1536)
    return mock


@pytest_asyncio.fixture
async def test_engine():
    """Crea un engine asincrono isolato e ricrea lo schema del DB a ogni test."""
    # Controllo di sicurezza per evitare di eseguire drop_all su database di produzione/sviluppo
    if "_test" not in TEST_DB_URL:
        raise ValueError("L'URL del database di test deve contenere '_test' per sicurezza!")

    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine):
    """Fornisce una sessione di database transazionale e isolata per i test."""
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session