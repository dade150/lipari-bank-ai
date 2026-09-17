from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.models import ChatMessage


class CostTracker:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def cost_by_model(self, days: int = 1) -> list[dict]:
        since = datetime.now(UTC) - timedelta(days=days)
        stmt = (
            select(
                ChatMessage.model_used,
                func.count().label("calls"),
                func.sum(ChatMessage.tokens).label("tokens"),
                func.sum(ChatMessage.cost_eur).label("cost"),
            )
            .where(ChatMessage.created_at >= since)
            .group_by(ChatMessage.model_used)
        )
        result = await self.session.execute(stmt)
        return [
            {
                "model": r.model_used or "unknown",
                "calls": r.calls,
                "tokens": int(r.tokens or 0),
                "cost_eur": float(r.cost or 0),
            }
            for r in result
        ]
