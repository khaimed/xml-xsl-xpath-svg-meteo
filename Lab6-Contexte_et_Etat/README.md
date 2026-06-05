# LAB 6 : L'état et le contexte d'un agent

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Comprendre et utiliser les deux mécanismes de mémoire d'un agent LangChain :
- **Contexte** : données passées à l'invocation, accessibles via `ToolRuntime`
- **État** : données persistées entre invocations via `InMemorySaver` et `AgentState`

---

## Structure du projet

```
Lab6-Contexte_et_Etat/
├── part1_context.py     # Parties 1-4 : ColourContext (contexte par invocation)
├── part2_state.py       # Parties 5-6 : CustomState  (état persisté entre invocations)
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Prérequis

- Python >= 3.10 · [uv](https://docs.astral.sh/uv/) · [Ollama](https://ollama.com/) avec `llama3.2:3b`

```bash
uv sync
```

---

## Partie 1 — `ColourContext` : classe de contexte structurée

Un **contexte** est un objet `@dataclass` passé à chaque `agent.invoke()`.  
Il est lu-only côté agent : il ne persiste pas entre les appels.

```python
@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"
```

---

## Partie 2 — Agent sans contexte

Sans tool, le LLM ne peut pas accéder aux valeurs du contexte —
il répond uniquement depuis sa connaissance générale.

```bash
uv run --active python part1_context.py
```

```
--- Partie 2 : Agent sans contexte ---
I don't have any information about your personal preferences...
```

---

## Partie 3 — Agent avec contexte

Des tools utilisant `ToolRuntime` permettent de lire le contexte depuis l'intérieur de l'exécution :

```python
@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour
```

```
--- Partie 3 : Agent avec contexte ---
Your favourite colour is blue.
```

---

## Partie 4 — Changement de contexte

Le contexte est injecté à l'invocation — changer sa valeur change immédiatement le comportement :

```python
agent.invoke(..., context=ColourContext(favourite_colour="green"))
```

```
--- Partie 4 : Changement de contexte (green) ---
Your favourite colour is green.
```

---

## Partie 5 — `CustomState` : état personnalisé

Un **état** hérite de `AgentState` et est persisté entre les invocations via un `checkpointer`.  
Contrairement au contexte, l'état **survit** d'un appel à l'autre (même `thread_id`).

```python
class CustomState(AgentState):
    favourite_colour: str
```

---

## Partie 6 — Agent qui modifie et récupère un état

```bash
uv run --active python part2_state.py
```

Un tool retourne une `Command` pour modifier l'état :

```python
@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state."""
    return Command(update={
        "favourite_colour": favourite_colour,
        "messages": [ToolMessage("Successfully updated favourite colour",
                     tool_call_id=runtime.tool_call_id)]
    })
```

Un autre tool lit l'état persisté :

```python
@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour from the state."""
    return runtime.state["favourite_colour"]
```

```
--- Stocker : "My favourite colour is green" ---
Successfully updated your favourite colour to green!

--- Récupérer : "What's my favourite colour?" (même thread_id) ---
Your favourite colour is green!
```

---

## Contexte vs État — Différences clés

| | Contexte (`@dataclass`) | État (`AgentState`) |
|---|---|---|
| **Persistance** | Non — par invocation | Oui — entre invocations |
| **Modification** | Non (read-only) | Oui (via `Command`) |
| **Passage** | `context=ColourContext(...)` | `{"configurable": {"thread_id": "1"}}` |
| **Accès tool** | `runtime.context.xxx` | `runtime.state["xxx"]` |
| **Checkpointer** | Non requis | `InMemorySaver()` requis |
| **Usage** | Préférences utilisateur par session | Mémoire persistante multi-tours |

---

## Architecture

```
Contexte (par invocation)          État (persisté)
─────────────────────────          ───────────────────────────
invoke(context=ColourContext())     invoke({}, thread_id="1")
         │                                    │
         ▼                                    ▼
   ToolRuntime.context              InMemorySaver (checkpointer)
         │                                    │
   runtime.context.xxx              runtime.state["xxx"]
                                    ← Command(update={...})
```
