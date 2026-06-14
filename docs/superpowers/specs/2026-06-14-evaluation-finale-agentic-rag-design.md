# Design — Évaluation finale : Agentic RAG avec LangGraph (Économie/Finance)

## Contexte

Évaluation individuelle du module IA Agentique (Master IIBDCC, Prof. RETAL Sara). Objectif : construire un système RAG agentique complet avec **LangGraph** (sans `create_agent`), couvrant base documentaire, outils, graphe (state + mémoire), visualisation, et évaluation sur 10 questions simples + 10 complexes.

**Domaine** : Économie/Finance — sous-thème **éducation financière personnelle** (budget, épargne, crédit, investissement, surendettement).

**Stack** : Ollama local `llama3.2:3b` + `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`), cohérent avec les Labs 1-9 du cours (gratuit, offline).

## 1. Base documentaire

4 PDF publics et gratuits, téléchargés dans `data/pdfs/` :

| Fichier | Source | Contenu |
|---|---|---|
| `guide_pedagogique_finance_pour_tous.pdf` | lafinancepourtous.com (partenaire AMF) | Budget personnel, compte bancaire, risques du crédit |
| `quest_ce_que_education_financiere.pdf` | lafinancepourtous.com | Concepts généraux d'éducation financière |
| `amf_investir_votre_epargne.pdf` | AMF (amf-france.org) | Épargne, placements, investissement, diversification |
| `guide_surendettement.pdf` | IEDOM / Banque de France | Crédit, endettement, procédure de surendettement |

**Ingestion** (`ingest.py`) :
- `PyPDFLoader` par fichier (comme Lab3)
- `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)`
- `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`
- Vectorstore **Chroma** persisté dans `data/chroma_db/` (construit une seule fois, réutilisé ensuite — évite le ré-embedding à chaque lancement)
- Métadonnées conservées : nom du fichier source + numéro de page

## 2. LLM

`src/llm.py` : une instance `ChatOllama(model="llama3.2:3b", temperature=0)`, réutilisée pour :
- le raisonnement de l'agent (avec outils liés)
- le grading de pertinence des documents récupérés
- la reformulation de requête

## 3. Outils (`src/tools.py`)

1. **`retrieve_documents(query: str)`** — recherche sémantique dans le vectorstore Chroma, retourne les top-k passages avec leur source/page.
2. **`compute_savings_projection(initial_amount, monthly_contribution, annual_rate_percent, years)`** — projection d'épargne avec intérêts composés (formule de valeur future avec versements réguliers).
3. **`compute_loan_payment(principal, annual_rate_percent, years)`** — calcul de mensualité de crédit (formule d'amortissement) + intérêts totaux.

Ces deux outils de calcul permettent à l'agent de répondre aux questions complexes nécessitant un raisonnement numérique en plus de la récupération documentaire, ce qui illustre la prise de décision "agentic" (quand récupérer vs calculer vs les deux).

## 4. Architecture du graphe (`src/graph.py`)

### State (`src/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    llm_calls: int
    rewrite_count: int
    retrieved_docs: list[str]   # contenu + métadonnées des derniers passages récupérés
```

### Nodes

- **`agent`** : LLM avec les 3 outils liés (`bind_tools`). System prompt définit le rôle (assistant en éducation financière personnelle) et invite à utiliser les outils quand pertinent.
- **`tools`** : exécute tous les tool calls demandés par le dernier message de l'agent. Si `retrieve_documents` figure parmi les outils appelés, son résultat est aussi stocké dans `state["retrieved_docs"]`.
- **`grade_documents`** : uniquement si `retrieve_documents` a été appelé lors du dernier passage par `tools`. Le LLM évalue (réponse contrainte "oui"/"non") si les passages dans `state["retrieved_docs"]` sont pertinents par rapport à la question.
- **`rewrite_query`** : si les documents ne sont pas pertinents et `rewrite_count < 2`, le LLM reformule la question (ajoutée comme nouveau message), incrémente `rewrite_count`, et on retente la récupération.

### Edges (graphe agentic RAG retrieve → grade → generate/rewrite)

```
START → agent
agent → [conditional] → tools          (si tool_calls présents)
agent → [conditional] → END            (sinon, réponse finale prête)
tools → [conditional] → grade_documents (si retrieve_documents a été appelé)
tools → [conditional] → agent           (sinon, ex: outils de calcul seuls)
grade_documents → [conditional] → agent        (docs pertinents OU rewrite_count >= 2)
grade_documents → [conditional] → rewrite_query (docs non pertinents et rewrite_count < 2)
rewrite_query → agent
```

### Mémoire

`InMemorySaver` (checkpointer) compilé avec le graphe. Chaque session de chat utilise un `thread_id` unique (`config={"configurable": {"thread_id": ...}}`), permettant une mémoire conversationnelle multi-tours.

## 5. Visualisation

`graph.get_graph().draw_mermaid_png()` → sauvegardé en `graph.png` à la racine du projet, inclus dans le rapport et le README.

## 6. Structure du projet

```
Evaluation-finale/
├── data/
│   ├── pdfs/                    # les 4 PDF sources
│   └── chroma_db/                # vectorstore persisté (généré par ingest.py, gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                 # chemins, noms de modèles, constantes
│   ├── llm.py                    # instance ChatOllama
│   ├── vectorstore.py            # chargement/construction du vectorstore Chroma
│   ├── tools.py                  # retrieve_documents, compute_savings_projection, compute_loan_payment
│   ├── state.py                  # AgentState (TypedDict)
│   └── graph.py                  # construction et compilation du graphe LangGraph
├── ingest.py                      # script d'indexation (à lancer une fois)
├── main.py                        # CLI interactif (chat avec mémoire via thread_id)
├── evaluation/
│   ├── questions.py               # 10 questions simples + 10 complexes (FR)
│   ├── run_evaluation.py           # exécute les 20 questions, mesure temps/pertinence, écrit les résultats
│   └── results/                    # results.csv + résumé markdown
├── graph.png                       # visualisation générée du graphe
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## 7. Plan d'évaluation

- **10 questions simples** : factuelles, réponse attendue dans un seul document (ex : "Qu'est-ce qu'un livret d'épargne ?", "Quel est le rôle de l'AMF ?").
- **10 questions complexes** : raisonnement multi-étapes, calculs via les outils, comparaisons entre documents, ou plusieurs cycles de récupération (déclenchant potentiellement `rewrite_query`).

`evaluation/run_evaluation.py` :
- pour chaque question : invoque le graphe (nouveau `thread_id`), mesure le temps de réponse (`time.perf_counter`), capture la réponse finale, les sources récupérées (`state["retrieved_docs"]`), `llm_calls`, `rewrite_count`.
- écrit `evaluation/results/results.csv` (colonnes : id, type, question, réponse, temps_s, sources, llm_calls, rewrites) + un résumé markdown.
- L'analyse qualitative (note de pertinence/qualité, comparaison simple vs complexe) est faite manuellement à partir de ce CSV pour le rapport.

## 8. Couverture de la grille de notation

| Critère | Couverture |
|---|---|
| Prétraitement et vectorisation (2) | `ingest.py` + Chroma + métadonnées source/page |
| Développement des outils (2) | 3 outils : retrieval + 2 calculateurs financiers |
| Qualité du graphe LangGraph (4) | Graphe multi-nœuds, routage conditionnel, boucle grade/rewrite, mémoire |
| Respect de l'approche Agentic RAG (4) | Décision d'outil par l'agent, auto-évaluation de pertinence, reformulation de requête |
| Qualité du code (2) | Package `src/` modulaire, un fichier = une responsabilité |
| Expérimentation et Simulation (3) | `evaluation/` : 20 questions, mesures temps/pertinence |
| Rapport (3) | Hors scope code — données issues de `evaluation/results/` |
