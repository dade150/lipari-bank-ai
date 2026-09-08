# alembic/env.py
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from lipari_bank_ai.config import settings
from lipari_bank_ai.db import models  # noqa: F401  registra i modelli su Base.metadata
from lipari_bank_ai.db.session import Base

# Oggetto di configurazione di Alembic, legge alembic.ini
config = context.config

# Logging da alembic.ini (sezioni [loggers], [handlers], ...)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Sovrascrive il placeholder di alembic.ini con l'URL reale letto da .env
config.set_main_option("sqlalchemy.url", settings.database_url)

# target_metadata e' quello che autogenerate confronta con lo stato del DB
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Genera SQL senza connettersi al DB (alembic upgrade --sql)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Crea un engine async e ci gira sopra le migration in modalita' sync
    tramite connection.run_sync (Alembic stesso non e' async-nativo).
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())