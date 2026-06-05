# LAB 5 : Visualiser, tester et déboguer des agents avec LangGraph Studio

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Utiliser **LangGraph Studio** pour visualiser le graphe d'un agent, l'exécuter étape par étape, inspecter les inputs/outputs de chaque node, et déboguer les outils et le RAG — le tout depuis une interface web interactive.

---

## Structure du projet

```
Lab5-LangGraph_Studio/
├── agent_simple.py      # Agent avec outil RAG simulé
├── langgraph.json       # Configuration LangGraph Studio
├── pyproject.toml       # Dépendances uv
├── .env.example         # Template variables d'environnement
└── .gitignore
```

---

## Prérequis

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) installé
- [Ollama](https://ollama.com/) avec `llama3.2:3b` (`ollama pull llama3.2:3b`)
- Compte [LangSmith](https://smith.langchain.com/) avec une clé API

### Installation

```bash
# Copier et remplir le fichier d'environnement
cp .env.example .env
# → Ajouter au minimum : LANGSMITH_API_KEY

# Installer les dépendances
uv sync
```

---

## Partie 1 — Créer un compte LangSmith et importer une clé

1. Aller sur **https://smith.langchain.com/**
2. Se connecter / créer un compte (Google, GitHub ou Email)
3. Aller dans **Settings → API Keys → Create API Key**
4. Donner un nom (ex: `langgraph-project`) et copier la clé
5. Ajouter dans `.env` :
   ```
   LANGSMITH_API_KEY=lsv2_pt_...
   ```

---

## Partie 2 — Créer l'agent

Le fichier `agent_simple.py` définit un agent LangChain avec un outil RAG simulé :

```python
@tool
def rag_search_opt(query: str) -> str:
    """Recherche des informations dans le texte."""
    return "Le personnage principal est un jeune homme nommé Jack..."

agent = create_agent(
    model=ChatOllama(model="llama3.2:3b", temperature=0),
    tools=[rag_search_opt],
    system_prompt="Tu es un assistant spécialisé dans l'analyse de texte..."
)
```

**Test rapide :**
```bash
uv run --active python -c "
from langchain.messages import HumanMessage
from agent_simple import agent
r = agent.invoke({'messages': [HumanMessage(content='Qui est le personnage principal ?')]})
print(r['messages'][-1].content)
"
```

---

## Partie 2 — Fichier de configuration `langgraph.json`

Ce fichier indique à LangGraph Studio où trouver l'agent, quel environnement utiliser, et comment lancer le projet :

```json
{
    "graphs": {
        "agent_simple": "./agent_simple.py:agent"
    },
    "env": "./.env",
    "source": {
        "kind": "uv",
        "root": "."
    }
}
```

| Clé | Description |
|---|---|
| `graphs` | Nom du graphe → chemin du fichier : variable |
| `env` | Fichier `.env` à charger |
| `source.kind` | Gestionnaire de paquets (`uv`) |
| `source.root` | Répertoire racine du projet |

---

## Partie 2 — Lancer LangGraph Studio

```bash
uv run --active langgraph dev
```

Le serveur démarre sur **http://127.0.0.1:2024** et ouvre automatiquement l'interface Studio dans le navigateur via **https://smith.langchain.com/studio**.

### Ce que vous pouvez faire dans Studio

| Fonctionnalité | Description |
|---|---|
| **Graph** | Visualise le graphe : `__start__ → model ⇄ tools → __end__` |
| **Chat** | Envoie des messages et voit les réponses en temps réel |
| **Interrupts** | Pause l'exécution à un node pour inspecter l'état |
| **Memory** | Visualise l'état persistant du thread |
| **Tracing** | Inspecte chaque appel LLM et tool call |

### Graphe de l'agent

```
        __start__
            │
            ▼
          model  ◄────┐
            │         │
     ┌──────┴──────┐  │
     │             │  │
   __end__       tools─┘
```

---

## Résultat attendu

Après `langgraph dev`, l'interface Studio affiche :

- Le graphe de l'agent avec les nodes `model` et `tools`
- Un panneau **Input** pour envoyer des messages
- La trace complète de chaque exécution (LLM call → tool call → réponse)

**Exemple d'interaction :**
> **User:** Qui est le personnage principal ?
> **Agent:** *(appelle `rag_search_opt`)* → Le personnage principal est Jack, un jeune homme qui découvre un ancien artefact magique.
