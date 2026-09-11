from fastapi import APIRouter

from lipari_bank_ai.models.categorize_model import CategorizeRequest, CategorizeResponse
from lipari_bank_ai.services.categorize_service import categorize

router = APIRouter(prefix="/api/ai", tags=["Categorize"])


@router.post(
    "/categorize",
    response_model=CategorizeResponse,
    summary="Categorize transaction via LLM",
)
async def categorize_endpoint(req: CategorizeRequest) -> CategorizeResponse:
    return await categorize(req)
