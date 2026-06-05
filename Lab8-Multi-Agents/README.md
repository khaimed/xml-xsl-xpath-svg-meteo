# LAB 8 : Multi-Agents en Langchain

**Master BDCC — SMA et IAD | Prof. RETAL SARA**

## Objectif

Construire un système multi-agents hiérarchique où un **agent principal** délègue des calculs à des **sous-agents spécialisés**, chacun équipé d'un outil dédié.

---

## Structure du projet

```
Lab8-Multi-Agents/
├── multi_agents.py    # Parties 1-4 : système multi-agents complet
├── langgraph.json     # Partie 5 : configuration LangGraph Studio
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

## Partie 1 — Définition des outils

Deux outils mathématiques simples :

```python
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2
```

---

## Partie 2 — Création des sous-agents

Chaque sous-agent est spécialisé sur un seul outil :

```python
model = ChatOllama(model="llama3.2:3b", temperature=0)

subagent_1 = create_agent(model=model, tools=[square_root])
subagent_2 = create_agent(model=model, tools=[square])
```

---

## Partie 3 — Création de l'agent principal

L'agent principal expose des tools qui encapsulent les appels aux sous-agents :

```python
@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 to calculate the square root of a number"""
    response = subagent_1.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square root of {x}")]}
    )
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> str:
    """Call subagent 2 to calculate the square of a number"""
    response = subagent_2.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square of {x}")]}
    )
    return response["messages"][-1].content

main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a math assistant...",
)
```

---

## Partie 4 — Appel et résultats

```bash
uv run --active python multi_agents.py
```

```
Question : What is the square root of 456?
  [HumanMessage] What is the square root of 456?
  [AIMessage] => appelle call_subagent_1({'x': 456})
  [ToolMessage:call_subagent_1] => The square root of 456.0 is approximately 21.3542.
  [AIMessage] The square root of 456 is approximately 21.3542.
Reponse finale : The square root of 456 is approximately 21.3542.

Question : What is the square of 12?
  [HumanMessage] What is the square of 12?
  [AIMessage] => appelle call_subagent_2({'x': 12})
  [ToolMessage:call_subagent_2] => The square of 12.0 is 144.0.
  [AIMessage] The square of 12 is 144.
Reponse finale : The square of 12 is 144.
```

---

## Partie 5 — LangGraph Studio

`langgraph.json` expose les 3 agents dans Studio :

```json
{
    "graphs": {
        "main_agent":  "./multi_agents.py:main_agent",
        "subagent_1":  "./multi_agents.py:subagent_1",
        "subagent_2":  "./multi_agents.py:subagent_2"
    },
    "env": "./.env",
    "source": {"kind": "uv", "root": "."}
}
```

Lancement du serveur Studio :

```bash
uv run langgraph dev
```

Puis ouvrir [https://smith.langchain.com/studio](https://smith.langchain.com/studio) et sélectionner l'agent à tester.

---

## Architecture

```
Utilisateur
    |
    v
main_agent  (llama3.2:3b)
    |               |
    v               v
call_subagent_1   call_subagent_2
    |               |
    v               v
subagent_1        subagent_2
(llama3.2:3b)     (llama3.2:3b)
    |               |
    v               v
square_root()     square()
(calcul Python)   (calcul Python)
```

---

## Concept clé : agents comme outils

Un sous-agent est invoqué via un `@tool` ordinaire. Le `main_agent` ne sait pas que `call_subagent_1` contient un autre agent — il l'appelle comme n'importe quel outil. C'est ce qui rend l'architecture **modulaire et extensible**.

| Composant | Rôle | Outil exposé |
|---|---|---|
| `subagent_1` | Racines carrées | `square_root` |
| `subagent_2` | Carrés | `square` |
| `main_agent` | Coordinateur | `call_subagent_1`, `call_subagent_2` |
