"""Diagnostica: chiama categorize() una sola volta, fuori da pytest, con un timeout.

Eseguire dalla root del progetto (dove sta pyproject.toml) con:
    uv run python diagnose_categorize.py
"""

import asyncio
import time

from lipari_bank_ai.models.categorize_model import CategorizeRequest
from lipari_bank_ai.services.categorize_service import categorize


async def main() -> None:
    req = CategorizeRequest(
        description="Bonifico Enel Energia bolletta marzo",
        amount=85.50,
    )

    print("Invio richiesta a categorize()... (timeout 60s)")
    start = time.time()

    try:
        result = await asyncio.wait_for(categorize(req), timeout=60)
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"❌ TIMEOUT dopo {elapsed:.1f}s: categorize() non e' tornato in 60s.")
        print("   -> Il problema e' nella chiamata al server Opencode (sessione o chat),")
        print("      non in pytest o nei golden file.")
        return
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - start
        print(f"❌ ERRORE dopo {elapsed:.1f}s: {type(e).__name__}: {e}")
        return

    elapsed = time.time() - start
    print(f"✅ Risposta ricevuta in {elapsed:.1f}s")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())