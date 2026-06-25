# Evaluation finale - Agentic RAG (Education financiere personnelle au Maroc)

Systeme RAG agentique construit avec LangGraph (sans `create_agent`), sur le
theme de l'education financiere personnelle au Maroc : budget, epargne, credit,
investissement, marche des capitaux et assurance.

## Stack

- LLM : Ollama local `llama3.2:3b`
- Embeddings : HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vectorstore : Chroma (persiste dans `data/chroma_db/`)
- Orchestration : LangGraph (`StateGraph`)

## Installation

```bash
uv sync --group dev
```

Assurez-vous qu'Ollama est lance et que le modele est disponible :

```bash
ollama pull llama3.2:3b
```

## Base documentaire

4 guides officiels marocains sur l'education financiere (sources publiques) :

| Fichier | Institution | Lien |
|---------|-------------|------|
| `bkam_guide_grand_public.pdf` | Bank Al-Maghrib | [Telecharger](https://www.bkam.ma/pedagogique/content/download/344251/2920093/BKAM_Guide_PEDA_0318_Web_VF.pdf) |
| `ammc_guide_investisseur_circuits.pdf` | AMMC | [Telecharger](https://www.ammc.ma/sites/default/files/Guide%20de%20l%27investisseur%20-%20Nov%202020.pdf) |
| `ammc_guide_instruments_financiers.pdf` | AMMC | [Telecharger](https://www.ammc.ma/sites/default/files/Guide%20de%20l%27investisseur%20-%20Comprendre%20les%20instruments%20financiers%20et%20leurs%20m%C3%A9canismes_0.pdf) |
| `acaps_guide_assure.pdf` | ACAPS | [Telecharger](https://www.acaps.ma/sites/default/files/acaps_guide_assure_vf.pdf) |

Placer les PDF dans `data/pdfs/` puis construire le vectorstore :

```bash
uv run python ingest.py   # 132 chunks indexes dans data/chroma_db/
```

## Utilisation

```bash
uv run python main.py
```

Chat interactif avec memoire conversationnelle (un `thread_id` par session).

## Architecture du graphe

```bash
uv run python generate_graph.py
```

Genere `graph.mmd` (et `graph.png` si internet disponible). Le graphe suit le
pattern Agentic RAG retrieve -> grade -> generate/rewrite :

- `agent` : le LLM decide d'appeler `retrieve_documents`,
  `compute_savings_projection`, `compute_loan_payment`, ou de repondre.
- `tools` : execute les outils demandes.
- `grade_documents` : evalue la pertinence des documents recuperes.
- `rewrite_query` : reformule la question si les documents ne sont pas
  pertinents (jusqu'a 2 fois), puis retourne a `agent`.

La memoire conversationnelle est assuree par un `InMemorySaver` (checkpointer)
indexe par `thread_id`.

## Outils

- `retrieve_documents(query)` : recherche semantique dans la base documentaire.
- `compute_savings_projection(initial_amount, monthly_contribution, annual_rate_percent, years)` :
  projection d'epargne avec interets composes.
- `compute_loan_payment(principal, annual_rate_percent, years)` : mensualite et
  cout total d'un credit amortissable.

## Tests

```bash
uv run pytest -v
```

## Evaluation

```bash
uv run python -m evaluation.run_evaluation
```

Execute 10 questions simples + 10 questions complexes, mesure le temps de
reponse et enregistre les sources recuperees dans
`evaluation/results/results.csv`.
