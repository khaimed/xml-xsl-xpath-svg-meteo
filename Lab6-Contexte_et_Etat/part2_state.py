from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="llama3.2:3b", temperature=0)

# ============================================================
# PARTIE 5 : Définir un état personnalisé héritant de AgentState
# ============================================================

class CustomState(AgentState):
    favourite_colour: str


# ============================================================
# PARTIE 6 : Agent qui modifie un état
# ============================================================

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour,
        "messages": [ToolMessage(
            "Successfully updated favourite colour",
            tool_call_id=runtime.tool_call_id
        )]
    })


agent_update = create_agent(
    model=model,
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

response = agent_update.invoke(
    {"messages": [HumanMessage(content="My favourite colour is green")]},
    {"configurable": {"thread_id": "1"}}
)
print("--- Partie 6 : Agent qui modifie un état ---")
print(response['messages'][-1].content)

# ============================================================
# PARTIE 6 (suite) : Agent qui récupère un état persisté
# ============================================================

@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"


agent_read = create_agent(
    model=model,
    tools=[update_favourite_colour, read_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)

# Première invocation : stocker la couleur
response = agent_read.invoke(
    {"messages": [HumanMessage(content="My favourite colour is green")]},
    {"configurable": {"thread_id": "1"}}
)
print("\n--- Partie 6 (suite) : Stocker la couleur ---")
print(response['messages'][-1].content)

# Deuxième invocation (même thread) : récupérer la couleur persistée
response = agent_read.invoke(
    {"messages": [HumanMessage(content="What's my favourite colour?")]},
    {"configurable": {"thread_id": "1"}}
)
print("\n--- Partie 6 (suite) : Récupérer la couleur ---")
print(response['messages'][-1].content)
