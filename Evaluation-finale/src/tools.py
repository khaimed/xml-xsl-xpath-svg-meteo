from pathlib import Path

from langchain.tools import tool

from src.config import RETRIEVAL_K
from src.vectorstore import get_vectorstore


@tool
def compute_savings_projection(
    initial_amount: float,
    monthly_contribution: float,
    annual_rate_percent: float,
    years: float,
) -> dict:
    """Calcule la valeur future d'une epargne avec versements mensuels et
    interets composes mensuellement.

    Args:
        initial_amount: montant initial deja epargne (en dirhams marocains, MAD).
        monthly_contribution: montant verse chaque mois (en dirhams marocains, MAD).
        annual_rate_percent: taux d'interet annuel en pourcentage (ex: 6 pour 6%).
        years: duree de l'epargne en annees.
    """
    n = years * 12
    i = annual_rate_percent / 100 / 12
    if i == 0:
        future_value = initial_amount + monthly_contribution * n
    else:
        future_value = initial_amount * (1 + i) ** n + monthly_contribution * (
            ((1 + i) ** n - 1) / i
        )
    total_contributed = initial_amount + monthly_contribution * n
    return {
        "future_value": round(future_value, 2),
        "total_contributed": round(total_contributed, 2),
        "interest_earned": round(future_value - total_contributed, 2),
    }


@tool
def compute_loan_payment(principal: float, annual_rate_percent: float, years: float) -> dict:
    """Calcule la mensualite d'un credit amortissable et le cout total des interets.

    Args:
        principal: montant emprunte (en dirhams marocains, MAD).
        annual_rate_percent: taux d'interet annuel en pourcentage (ex: 6 pour 6%).
        years: duree du credit en annees.
    """
    n = years * 12
    i = annual_rate_percent / 100 / 12
    if i == 0:
        monthly_payment = principal / n
    else:
        monthly_payment = principal * i * (1 + i) ** n / ((1 + i) ** n - 1)
    total_paid = monthly_payment * n
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_paid - principal, 2),
    }


@tool
def retrieve_documents(query: str) -> str:
    """Recherche dans la base documentaire d'education financiere personnelle
    (budget, epargne, credit, investissement, surendettement) les passages
    les plus pertinents pour une question.

    Args:
        query: la question ou les mots-cles a rechercher.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    if not docs:
        return "Aucun document pertinent trouve."
    formatted = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "inconnu")).name
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


TOOLS = [retrieve_documents, compute_savings_projection, compute_loan_payment]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
