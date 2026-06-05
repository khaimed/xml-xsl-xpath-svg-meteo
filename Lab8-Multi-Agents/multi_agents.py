from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

# ============================================================
# PARTIE 1 : Définition des outils
# ============================================================

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2


# ============================================================
# PARTIE 2 : Création des sous-agents
# ============================================================

model = ChatOllama(model="llama3.2:3b", temperature=0)

subagent_1 = create_agent(
    model=model,
    tools=[square_root],
)

subagent_2 = create_agent(
    model=model,
    tools=[square],
)


# ============================================================
# PARTIE 3 : Création de l'agent principal
# ============================================================

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
    system_prompt=(
        "You are a math assistant. When asked to compute a square root or square, "
        "call the appropriate subagent tool and reply with the EXACT numerical result "
        "returned by the subagent. Always include the number in your answer."
    ),
)


# ============================================================
# PARTIE 4 : Appeler les agents et afficher le résultat
# ============================================================

if __name__ == "__main__":
    def run(question: str):
        print(f"\nQuestion : {question}")
        response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
        for msg in response["messages"]:
            role = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [{role}] => appelle {tc['name']}({tc['args']})")
            elif hasattr(msg, "name") and msg.name:
                print(f"  [ToolMessage:{msg.name}] => {msg.content}")
            else:
                print(f"  [{role}] {msg.content}")
        print(f"Réponse finale : {response['messages'][-1].content}")

    run("What is the square root of 456?")
    run("What is the square of 12?")
