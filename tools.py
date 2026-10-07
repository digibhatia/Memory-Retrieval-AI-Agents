"""
tools.py — Smriti's tools. lookup_policy now accepts a `mode`, so the
SAME retrieval tool can be pointed at three deliberately broken states,
not just the normal one:

    mode="normal"    -- the real vector store, as built
    mode="stale"     -- pretend the index hasn't been rebuilt: answer from
                        a small, deliberately outdated snapshot instead
    mode="empty"     -- pretend nothing was indexed at all
    mode="oversized" -- ignore the top-k limit and the context budget,
                        and return far more than context.py would ever
                        pack into a real prompt

These exist to make retrieval's failure modes something you can trigger
on demand, in a lab, rather than something you only discover by accident
in production. calculate_monthly_repayment is unchanged from Nirikshan.
"""

from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from context import pack_context

load_dotenv()

HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "lloyds_smriti"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# A small, deliberately outdated snapshot used only by mode="stale" — this
# is what the knowledge base looked like before the Premier account's
# overdraft rate was last repriced, kept here on purpose as a fixture.
_STALE_SNAPSHOT = [
    {"text": "Premier account arranged overdraft rate: 24.9% EAR (variable). Standard account: 39.9% EAR.",
     "source": "02_premier_account_terms.docx (STALE SNAPSHOT)", "unit_id": "Overdraft Rates"},
]

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. See README.md, Step 2.")
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        if _collection.count() == 0:
            raise RuntimeError("The vector store is empty. Run 'python build_vectorstore.py' first.")
    return _collection


def lookup_policy(query: str, mode: str = "normal") -> str:
    """
    Retrieval tool. `mode` lets a lab exercise deliberately misbehave:
      normal    -- real retrieval against the built vector store
      stale     -- answer from _STALE_SNAPSHOT instead of the real index
      empty     -- return no hits, as if the index were empty
      oversized -- return far more raw context than should ever be packed
    """
    if mode == "empty":
        return pack_context([])

    if mode == "stale":
        return pack_context(_STALE_SNAPSHOT)

    collection = _get_collection()

    if mode == "oversized":
        # Deliberately ask for way more than a real system would, and skip
        # context.py's budget entirely — this is what "stuffing the
        # context window" looks like from the inside.
        result = collection.query(query_texts=[query], n_results=min(50, collection.count()))
        hits = [
            {"text": doc, "source": meta["source"], "unit_id": meta.get("unit_id", "")}
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]
        return "\n\n".join(f"[Source: {h['source']} | {h['unit_id']}]\n{h['text']}" for h in hits)

    # normal
    result = collection.query(query_texts=[query], n_results=4)
    hits = [
        {"text": doc, "source": meta["source"], "unit_id": meta.get("unit_id", "")}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
    return pack_context(hits)


def calculate_monthly_repayment(principal: float, annual_rate_percent: float, term_years: int) -> str:
    """Calculator tool: standard repayment-mortgage monthly payment formula. Unchanged from Nirikshan."""
    principal = float(principal)
    annual_rate_percent = float(annual_rate_percent)
    term_years = int(term_years)

    if principal <= 0 or term_years <= 0:
        return "Error: principal and term_years must both be greater than zero."
    if annual_rate_percent < 0:
        return "Error: annual_rate_percent cannot be negative."

    monthly_rate = (annual_rate_percent / 100) / 12
    n_payments = term_years * 12
    if monthly_rate == 0:
        payment = principal / n_payments
    else:
        payment = (
            principal * monthly_rate * (1 + monthly_rate) ** n_payments
            / ((1 + monthly_rate) ** n_payments - 1)
        )
    total_paid = payment * n_payments
    return (
        f"Estimated monthly repayment: GBP {payment:,.2f}\n"
        f"Over {term_years} years ({n_payments} payments) at {annual_rate_percent}% per year.\n"
        f"Total repaid: GBP {total_paid:,.2f}\n"
        "(Illustrative only — ignores fees, overpayments, and rate changes after any fixed period.)"
    )


TOOL_SPECS = [
    {
        "name": "lookup_policy",
        "description": (
            "Search the Lloyds Financial knowledge base for the passage most relevant to a question. "
            "Always use this before answering any question about rates, limits, eligibility, or "
            "procedure. Only pass `mode` if explicitly asked to test a failure scenario; otherwise omit it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question, or the specific fact to check."},
                "mode": {"type": "string", "enum": ["normal", "stale", "empty", "oversized"],
                          "description": "Retrieval mode. Omit, or use 'normal', unless deliberately testing a failure scenario."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_monthly_repayment",
        "description": "Calculate an illustrative monthly repayment for a loan, given the amount, rate, and term.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in GBP."},
                "annual_rate_percent": {"type": "number", "description": "Annual interest rate, e.g. 5.25."},
                "term_years": {"type": "integer", "description": "Loan term in whole years."},
            },
            "required": ["principal", "annual_rate_percent", "term_years"],
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_monthly_repayment": calculate_monthly_repayment,
}


if __name__ == "__main__":
    print("Empty mode:", lookup_policy("overdraft rate", mode="empty"))
    print("\nStale mode:", lookup_policy("overdraft rate", mode="stale"))
