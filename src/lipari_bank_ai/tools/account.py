from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.models import BankAccount


class AccountTool:
    """Tool per recuperare il saldo del conto corrente."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @property
    def name(self) -> str:
        return "get_account_balance"

    @property
    def description(self) -> str:
        return (
            "Recupera il saldo del conto corrente dell'utente. "
            "Usalo quando l'utente chiede il saldo, quanto ha sul conto, "
            "o informazioni sul proprio conto corrente."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "account_type": {
                    "type": "string",
                    "description": "Tipo di conto: 'checking' (corrente) o 'savings' (risparmio). "
                    "Default: 'checking'",
                    "enum": ["checking", "savings"],
                },
            },
            "required": [],
        }

    async def execute(self, params: dict[str, Any], user_id: str) -> dict[str, Any]:
        account_type = params.get("account_type", "checking")

        stmt = (
            select(BankAccount)
            .where(BankAccount.user_id == user_id)
            .where(BankAccount.account_type == account_type)
        )
        result = await self.session.execute(stmt)
        account = result.scalar_one_or_none()

        if account is None:
            return {
                "error": f"Nessun conto '{account_type}' trovato per l'utente {user_id}.",
            }

        return {
            "account_id": account.id,
            "account_type": account.account_type,
            "balance": float(account.balance),
            "currency": account.currency,
        }
