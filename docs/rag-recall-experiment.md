# RAG Recall Experiment

## Setup
- Embedding model: all-MiniLM-L6-v2 (384 dim, locale)
- Vector DB: pgvector con ivfflat index
- Top-k: 3
- Documento testato: commissioni_bonifico.md

## Ground Truth (10 Q&A)

| # | Query | Expected Doc | Status |
|---|---|---|---|
| 1 | Costo bonifico SEPA istantaneo | commissioni_bonifico | OK |
| 2 | Quanto costa un bonifico online | commissioni_bonifico | OK |
| 3 | Tempo bonifico estero extra-SEPA | commissioni_bonifico | OK |
| 4 | Limite importo bonifico istantaneo | commissioni_bonifico | OK |
| 5 | Commissioni allo sportello | commissioni_bonifico | OK |
| 6 | Costo bonifico verso paesi extra-SEPA | commissioni_bonifico | OK |
| 7 | Tempo esecuzione bonifico standard | commissioni_bonifico | OK |
| 8 | Bonifico SEPA quanto costa online | commissioni_bonifico | OK |
| 9 | Costo bonifico istantaneo limite | commissioni_bonifico | OK |
| 10 | Dove trovo le commissioni dei bonifici | commissioni_bonifico | OK |

## Risultato
**Recall@3: 100.0% (10/10)**

## Conclusioni
- Il modello all-MiniLM-L6-v2 gestisce bene domande in italiano su documenti bancari
- La similarità media è ~0.55 (range 0.45-0.64), compatibile con l'uso di cosine distance
- Con un solo documento il recall è ovviamente al massimo; test futuri con multi-doc serve a validare la discriminazione
