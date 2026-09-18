"""Seed database with fake Italian banking data."""
import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from random import choice, randint, uniform

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from lipari_bank_ai.config import settings
from lipari_bank_ai.db.models import BankAccount, Transaction, Base

USERS = ["u1", "u2", "u3"]

ACCOUNTS = [
    {"user_id": "u1", "account_type": "checking", "balance": 5230.50, "currency": "EUR"},
    {"user_id": "u1", "account_type": "savings", "balance": 15420.00, "currency": "EUR"},
    {"user_id": "u2", "account_type": "checking", "balance": 1850.75, "currency": "EUR"},
    {"user_id": "u3", "account_type": "checking", "balance": 3100.00, "currency": "EUR"},
    {"user_id": "u3", "account_type": "savings", "balance": 28750.20, "currency": "EUR"},
]

TRANSACTION_TEMPLATES = [
    {"desc": "Esselunga Supermercato", "category": "GROCERIES", "min": -85.0, "max": -15.0},
    {"desc": "Enel Bolletta Luce", "category": "UTILITIES", "min": -120.0, "max": -60.0},
    {"desc": "Eni Distribuzione Carburante", "category": "TRANSPORT", "min": -60.0, "max": -30.0},
    {"desc": "Ristorante Da Luigi", "category": "RESTAURANTS", "min": -55.0, "max": -20.0},
    {"desc": "Netflix Abbonamento", "category": "ENTERTAINMENT", "min": -15.99, "max": -15.99},
    {"desc": "Coop Supermercato", "category": "GROCERIES", "min": -70.0, "max": -25.0},
    {"desc": "TIM Telefonia", "category": "UTILITIES", "min": -29.99, "max": -29.99},
    {"desc": "Amazon Acquisto", "category": "OTHER", "min": -95.0, "max": -10.0},
    {"desc": "Bar Caffetteria Roma", "category": "RESTAURANTS", "min": -8.0, "max": -2.50},
    {"desc": "Stazione Trenitalia", "category": "TRANSPORT", "min": -45.0, "max": -12.0},
    {"desc": "Bonifico Ricevuto Stipendio", "category": "OTHER", "min": 1800.0, "max": 2500.0},
    {"desc": "Spotify Premium", "category": "ENTERTAINMENT", "min": -9.99, "max": -9.99},
    {"desc": "Pharmacy Farmacia", "category": "UTILITIES", "min": -35.0, "max": -8.0},
    {"desc": "Carrefour Market", "category": "GROCERIES", "min": -55.0, "max": -18.0},
    {"desc": "Uber Rides", "category": "TRANSPORT", "min": -25.0, "max": -8.0},
]


def _random_amount(template: dict) -> float:
    return round(uniform(template["min"], template["max"]), 2)


def _random_date(days_back: int = 30) -> datetime:
    delta = timedelta(days=randint(0, days_back), hours=randint(6, 22), minutes=randint(0, 59))
    return datetime.now(UTC) - delta


async def seed() -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        account_ids: list[str] = []

        for acc_data in ACCOUNTS:
            acc_id = str(uuid.uuid4())
            account_ids.append(acc_id)
            session.add(BankAccount(
                id=acc_id,
                user_id=acc_data["user_id"],
                account_type=acc_data["account_type"],
                balance=acc_data["balance"],
                currency=acc_data["currency"],
            ))

        await session.flush()

        tx_count = 0
        for acc_id in account_ids:
            for _ in range(randint(4, 8)):
                template = choice(TRANSACTION_TEMPLATES)
                session.add(Transaction(
                    id=str(uuid.uuid4()),
                    account_id=acc_id,
                    amount=_random_amount(template),
                    description=template["desc"],
                    category=template["category"],
                    created_at=_random_date(),
                ))
                tx_count += 1

        await session.commit()
        print(f"Seeded {len(account_ids)} accounts and {tx_count} transactions.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
