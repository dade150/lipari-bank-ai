import asyncio

from lipari_bank_ai.db.session import AsyncSessionLocal
from lipari_bank_ai.llm.embedding_client import EmbeddingClient
from lipari_bank_ai.services.retrieval_service import RetrievalService

GROUND_TRUTH = [
    {"q": "Costo bonifico SEPA istantaneo", "expected_doc": "commissioni_bonifico"},
    {"q": "Quanto costa un bonifico online", "expected_doc": "commissioni_bonifico"},
    {"q": "Tempo bonifico estero extra-SEPA", "expected_doc": "commissioni_bonifico"},
    {"q": "Limite importo bonifico istantaneo", "expected_doc": "commissioni_bonifico"},
    {"q": "Commissioni allo sportello", "expected_doc": "commissioni_bonifico"},
    {"q": "Costo bonifico verso paesi extra-SEPA", "expected_doc": "commissioni_bonifico"},
    {"q": "Tempo esecuzione bonifico standard", "expected_doc": "commissioni_bonifico"},
    {"q": "Bonifico SEPA quanto costa online", "expected_doc": "commissioni_bonifico"},
    {"q": "Costo bonifico istantaneo limite", "expected_doc": "commissioni_bonifico"},
    {"q": "Dove trovo le commissioni dei bonifici", "expected_doc": "commissioni_bonifico"},
]


async def main():
    async with AsyncSessionLocal() as session:
        emb = EmbeddingClient()
        retrieval = RetrievalService(session, emb)

        correct = 0
        for item in GROUND_TRUTH:
            chunks = await retrieval.retrieve(item["q"], top_k=3)
            found = any(
                c.document_id == item["expected_doc"] for c in chunks
            )
            status = "OK" if found else "FAIL"
            if found:
                correct += 1
            print(f"[{status}] {item['q']}")
            for c in chunks:
                print(f"    sim={c.similarity:.3f} | {c.document_id}")

        recall = correct / len(GROUND_TRUTH)
        print(f"\nRecall@3: {recall:.1%} ({correct}/{len(GROUND_TRUTH)})")


if __name__ == "__main__":
    asyncio.run(main())
