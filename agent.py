"""
agent.py — run this in Terminal 2, while app.py keeps running in Terminal 1.

Sends 5 domain questions through ONE shared session (so you can watch
memory accumulate across them), then a 6th follow-up question that only
makes sense WITH memory, then three bonus sections on today's actual
topic: inspecting and resetting memory, stress-testing retrieval's
failure modes on demand, and measuring non-determinism directly.

RUN IT:
    python agent.py
"""

from __future__ import annotations

import requests

BASE_URL = "http://localhost:8000"
SESSION_ID = "lab-session-1"

QUESTIONS = [
    ("What is the maximum loan-to-value for a first-time buyer under the assisted scheme?", "PDF (mortgage SOP)"),
    ("What is the overdraft interest rate on a Premier account?", "DOCX (account terms)"),
    ("Who do I contact if I suspect fraud outside business hours?", "HTML (FAQ — also a PII redaction test)"),
    ("Which complaints are still open?", "CSV (complaints register)"),
    ("Which ISA product has the highest interest rate, and what is the early-access penalty on it?", "JSON (product rates)"),
]
MEMORY_FOLLOWUP = "And what would the monthly repayment be on that first one, for GBP 250,000 at 5.2% over 30 years?"


def chat(session_id: str, message: str) -> dict:
    resp = requests.post(f"{BASE_URL}/chat", params={"session_id": session_id, "message": message}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_memory(session_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/memory/{session_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def reset_memory(session_id: str) -> dict:
    resp = requests.post(f"{BASE_URL}/memory/{session_id}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def retrieval_stress(q: str, mode: str) -> dict:
    resp = requests.get(f"{BASE_URL}/retrieval-stress", params={"q": q, "mode": mode}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def stability(q: str, n_runs: int = 5, temperature=None) -> dict:
    params = {"q": q, "n_runs": n_runs}
    if temperature is not None:
        params["temperature"] = temperature
    resp = requests.get(f"{BASE_URL}/stability", params=params, timeout=90)
    resp.raise_for_status()
    return resp.json()


def run_domain_questions() -> None:
    print("=" * 70)
    print(f"5 DOMAIN QUESTIONS, one shared session ({SESSION_ID!r}) — watch memory grow")
    print("=" * 70)
    for question, tag in QUESTIONS:
        print(f"\n[{tag}]\nQ: {question}")
        try:
            result = chat(SESSION_ID, question)
        except requests.exceptions.ConnectionError:
            print("Could not reach Smriti — is app.py running in another terminal?")
            return
        print(f"A: {result['answer']}")
        print(f"   memory turn_count: {result['memory']['turn_count']}")

    print(f"\n[Follow-up that only works WITH memory]\nQ: {MEMORY_FOLLOWUP}")
    result = chat(SESSION_ID, MEMORY_FOLLOWUP)
    print(f"A: {result['answer']}")
    print(
        "\nNotice the follow-up said 'that first one' and 'the premier account' — it only resolves "
        "correctly because every earlier turn in this session was resent as context. Without memory, "
        "this question is unanswerable."
    )


def bonus_1_memory_inspection() -> None:
    print("\n" + "=" * 70)
    print("BONUS 1 — inspecting and resetting memory")
    print("=" * 70)
    mem = get_memory(SESSION_ID)
    print(f"Session {SESSION_ID!r}: {mem['turn_count']} turns stored, "
          f"{mem['truncated_count_lifetime']} dropped lifetime.")
    for t in mem["turns"]:
        print(f"    {t['role']:<10} {t['content'][:60]}")

    print("\nResetting this session...")
    reset_memory(SESSION_ID)
    mem_after = get_memory(SESSION_ID)
    print(f"Found after reset: {mem_after['found']}  (memory is just a dict — reset really does empty it)")

    print("\nNow overflowing a fresh session on purpose, 6 short turns at a time...")
    overflow_session = "overflow-demo"
    for i in range(6):
        chat(overflow_session, f"Quick question number {i + 1} about ISA rates.")
    mem_overflow = get_memory(overflow_session)
    print(f"After 6 exchanges: turn_count={mem_overflow['turn_count']}, "
          f"truncated_count_lifetime={mem_overflow['truncated_count_lifetime']}")
    print(
        "If turn_count capped below what 6 full exchanges would produce, truncation already fired — "
        "the oldest turns were silently dropped to keep the conversation within budget."
    )


def bonus_2_retrieval_stress() -> None:
    print("\n" + "=" * 70)
    print("BONUS 2 — stress-testing retrieval's failure modes on demand")
    print("=" * 70)
    query = "What is the overdraft interest rate on a Premier account?"
    for mode in ["normal", "stale", "empty", "oversized"]:
        r = retrieval_stress(query, mode)
        print(f"\n[mode={mode}]  {r['context_chars']} chars retrieved in {r['duration_ms']} ms")
        print(f"  {r['retrieved_context'][:140]}...")
    print(
        "\nNotice 'stale' confidently returns a plausible-looking but outdated rate, with nothing "
        "marking it as wrong — that is exactly why a monitoring system needs to know when an index "
        "was last rebuilt, not just whether retrieval 'succeeded.'"
    )


def bonus_3_nondeterminism() -> None:
    print("\n" + "=" * 70)
    print("BONUS 3 — measuring non-determinism directly")
    print("=" * 70)
    question = "In one sentence, what is a mortgage?"

    print(f"Q: {question!r}, run 5 times at default temperature\n")
    result = stability(question, n_runs=5)
    for i, a in enumerate(result["answers"], 1):
        print(f"  run {i}: {a}")
    print(f"\n  unique_answers: {result['unique_answers']} / {result['n_runs']}")
    print(f"  avg_pairwise_similarity: {result['avg_pairwise_similarity']}  (1.0 = identical every time)")
    print(
        "\nThe retrieval step and the calculator tool are deterministic by construction — ask Bonus 2's "
        "'normal' query five times and the chunks returned never change. The model's own wording is "
        "the one genuinely stochastic part of this whole system."
    )


if __name__ == "__main__":
    run_domain_questions()
    bonus_1_memory_inspection()
    bonus_2_retrieval_stress()
    bonus_3_nondeterminism()
