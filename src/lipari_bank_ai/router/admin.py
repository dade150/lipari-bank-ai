from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.db.models import ChatMessage
from lipari_bank_ai.db.session import get_db

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/cost-report")
async def cost_report(db: AsyncSession = Depends(get_db)) -> dict:
    since = datetime.now(UTC) - timedelta(days=1)
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
    result = await db.execute(stmt)
    return {
        "period": "last_24h",
        "by_model": [
            {
                "model": r.model_used or "unknown",
                "calls": r.calls,
                "tokens": int(r.tokens or 0),
                "cost_eur": float(r.cost or 0),
            }
            for r in result
        ],
    }
