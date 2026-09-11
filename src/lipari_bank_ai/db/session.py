# lipari_bank_ai/db/session.py
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from lipari_bank_ai.config import settings


class Base(DeclarativeBase):
    """Unica base dichiarativa del progetto. Tutti i modelli devono ereditare da qui
    (importarla da questo modulo, non ridefinirla altrove), altrimenti Alembic
    non li vede in target_metadata.
    """

    pass


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # log SQL in dev
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency FastAPI: una sessione per request.

    Commit automatico se l'endpoint completa senza errori, rollback se
    viene sollevata un'eccezione (es. AppError). Cosi' i repository possono
    limitarsi a flush() senza doversi ricordare di committare.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
