from pathlib import Path

import pytest

from lipari_bank_ai.eval.runners import eval_categorize, eval_rag

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
async def test_categorize_accuracy_threshold():
    result = await eval_categorize(Path("src/lipari_bank_ai/eval/categorize_golden.jsonl"))
    assert result["accuracy"] >= 0.80, f"Accuracy: {result['accuracy']:.1%}"


@pytest.mark.asyncio
async def test_rag_recall_threshold():
    result = await eval_rag(Path("src/lipari_bank_ai/eval/rag_golden.jsonl"))
    assert result["recall_at_5"] >= 0.70
    assert result["recall_at_1"] >= 0.50
