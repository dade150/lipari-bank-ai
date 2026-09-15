import asyncio
import json
import time
from pathlib import Path
from typing import Any

from lipari_bank_ai.db.session import AsyncSessionLocal
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.models.categorize_model import CategorizeRequest
from lipari_bank_ai.services.categorize_service import categorize
from lipari_bank_ai.services.retrieval_service import RetrievalService


async def eval_categorize(dataset_path: Path) -> dict[str, Any]:
    """Run categorize on dataset, return metrics."""
    raw = await asyncio.to_thread(dataset_path.read_text)
    examples = [json.loads(line) for line in raw.splitlines() if line.strip()]

    correct = 0
    failures: list[dict] = []
    total_latency = 0.0

    for ex in examples:
        start = time.time()
        req = CategorizeRequest(**ex["input"])
        result = await categorize(req)
        latency = time.time() - start

        is_correct = result.category == ex["expected"]["category"]
        if is_correct:
            correct += 1
        else:
            failures.append({
                "input": ex["input"],
                "expected": ex["expected"],
                "actual": result.model_dump(),
            })

        total_latency += latency

    n = len(examples)
    return {
        "accuracy": correct / n,
        "n_examples": n,
        "n_failures": len(failures),
        "avg_latency_s": total_latency / n,
        "failures": failures[:10],
    }


async def eval_rag(dataset_path: Path) -> dict[str, Any]:
    """Eval retrieval recall@5."""
    raw = await asyncio.to_thread(dataset_path.read_text)
    examples = [json.loads(line) for line in raw.splitlines() if line.strip()]

    async with AsyncSessionLocal() as session:
        retrieval = RetrievalService(session, EmbeddingClient())

        correct_at_5 = 0
        correct_at_1 = 0

        for ex in examples:
            chunks = await retrieval.retrieve(ex["query"], top_k=5)
            doc_ids = [c.document_id for c in chunks]

            expected_doc = ex["expected_doc_id"]
            if expected_doc in doc_ids:
                correct_at_5 += 1
            if doc_ids and doc_ids[0] == expected_doc:
                correct_at_1 += 1

        n = len(examples)
        return {
            "recall_at_5": correct_at_5 / n,
            "recall_at_1": correct_at_1 / n,
            "n_examples": n,
        }


JUDGE_PROMPT = """Valuta se la risposta dell'assistente AI e' adeguata.

DOMANDA: {question}
CONTESTO FORNITO: {context}
RISPOSTA: {answer}

Criteri:
1. Faithfulness: la risposta e' basata sul contesto (no allucinazioni)?
2. Relevance: la risposta affronta la domanda?
3. Completeness: tutti i punti rilevanti sono coperti?
4. Citation: cita correttamente il documento?

Output JSON:
{{
    "faithfulness": <0.0-1.0>,
    "relevance": <0.0-1.0>,
    "completeness": <0.0-1.0>,
    "citation_correct": <true/false>,
    "overall": <0.0-1.0>,
    "reasoning": "..."
}}"""


async def llm_judge(question: str, context: str, answer: str) -> dict[str, Any]:
    from lipari_bank_ai.config import settings
    from lipari_bank_ai.llm.opencode_provider import OpencodeProvider
    from lipari_bank_ai.llm.types import Message

    provider = OpencodeProvider(settings.opencode_api_key)
    prompt = JUDGE_PROMPT.format(
        question=question, context=context, answer=answer
    )
    response = await provider.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=500,
    )
    return json.loads(response.content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m lipari_bank_ai.eval.runners [categorize|rag]")
        sys.exit(1)

    task = sys.argv[1]

    if task == "categorize":
        result = asyncio.run(
            eval_categorize(Path("src/lipari_bank_ai/eval/categorize_golden.jsonl"))
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        THRESHOLD = 0.80
        if result["accuracy"] < THRESHOLD:
            print(f"❌ Accuracy {result['accuracy']:.1%} < threshold {THRESHOLD:.0%}")
            sys.exit(1)
        print(f"✅ Accuracy {result['accuracy']:.1%}")

    elif task == "rag":
        result = asyncio.run(
            eval_rag(Path("src/lipari_bank_ai/eval/rag_golden.jsonl"))
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        THRESHOLD = 0.70
        if result["recall_at_5"] < THRESHOLD:
            print(f"❌ Recall@5 {result['recall_at_5']:.1%} < threshold {THRESHOLD:.0%}")
            sys.exit(1)
        print(f"✅ Recall@5 {result['recall_at_5']:.1%}")
