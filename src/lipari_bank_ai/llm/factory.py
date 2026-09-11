from lipari_bank_ai.config import settings
from lipari_bank_ai.llm.opencode_provider import OpencodeProvider


def get_llm_provider(model: str | None = None) -> OpencodeProvider:
    selected = model or settings.default_model
    return OpencodeProvider(settings.opencode_api_key, selected)
