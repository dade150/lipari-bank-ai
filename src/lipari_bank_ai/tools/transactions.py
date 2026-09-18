from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.models import BankAccount, Transaction


class TransactionTool:
    """Tool per recuperare le transazioni recenti."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @property
    def name(self) -> str:
        return "get_recent_transactions"

    @property
    def description(self) -> str:
        return (
            "Recupera le transazioni recenti del conto dell'utente. "
            "Usalo quando l'utente chiede le ultime spese, il movimento del conto, "
            "o vuole sapere dove ha speso i soldi di recente."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di transazioni. Default: 5, Max: 20",
                },
                "category": {
                    "type": "string",
                    "description": "Filtra per categoria (es. GROCERIES, TRANSPORT, UTILITIES)",
                },
            },
            "required": [],
        }

    async def execute(self, params: dict[str, Any], user_id: str) -> dict[str, Any]:
        limit = min(params.get("limit", 5), 20)
        category = params.get("category")

        stmt_account = (
            select(BankAccount.id)
            .where(BankAccount.user_id == user_id)
            .where(BankAccount.account_type == "checking")
        )
        result = await self.session.execute(stmt_account)
        account_id = result.scalar_one_or_none()

        if account_id is None:
            return {"error": f"Nessun conto corrente trovato per l'utente {user_id}."}

        stmt = (
            select(Transaction)
            .where(Transaction.account_id == account_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
        )
        if category:
            stmt = stmt.where(Transaction.category == category.upper())

        result = await self.session.execute(stmt)
        transactions = result.scalars().all()

        return {
            "account_id": account_id,
            "count": len(transactions),
            "transactions": [
                {
                    "amount": float(tx.amount),
                    "description": tx.description,
                    "category": tx.category,
                    "date": tx.created_at.isoformat(),
                }
                for tx in transactions
            ],
        }
