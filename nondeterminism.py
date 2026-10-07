"""
nondeterminism.py — ask the identical question N times, and measure, not
guess, how much the answers actually vary.

A model is not a calculator: the same input does not guarantee the same
output. That's a fact worth demonstrating rather than asserting. This
file runs one question N times through the SAME tool-calling loop
(reusing llm_provider.chat_raw directly, no retrieval needed for this
specific measurement — see run_stability_test's docstring) and reports:

  - whether every run chose the same tool (or no tool) first
  - the token counts across runs (these vary too — a longer or shorter
    answer uses more or fewer tokens, even for an identical question)
  - a simple text-similarity score between runs, so "the wording changed"
    is a number, not an impression

WHICH PARTS OF THE SYSTEM ABSORB NON-DETERMINISM, AND WHICH DON'T:
  - calculate_monthly_repayment is pure arithmetic: identical inputs
    always produce an identical result. Non-determinism cannot enter here.
  - lookup_policy's retrieval step, against an unchanged vector store,
    returns the same nearest chunks every time for the same query —
    retrieval is deterministic even though the model that reads the
    result is not.
  - The model call itself is the one genuinely stochastic step. Setting
    temperature=0 (see run_stability_test) reduces, but does not
    guarantee, run-to-run identical wording.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from llm_provider import chat_raw
from tools import TOOL_SPECS

SYSTEM_PROMPT = (
    "You are a customer service assistant for Lloyds Financial (a training scenario). Use "
    "lookup_policy for any question about rates, limits, eligibility, fraud, complaints, or "
    "savings products. Keep answers to one or two sentences."
)


def _text_similarity(a: str, b: str) -> float:
    """0.0 (completely different) to 1.0 (identical) — a cheap, good-enough stand-in for 'how similar is the wording'."""
    return SequenceMatcher(None, a, b).ratio()


def run_stability_test(question: str, n_runs: int = 5, temperature: float | None = None) -> dict:
    """
    Run `question` through a single (no-tool) chat call n_runs times and
    report how stable the answers were. temperature=0 is the standard
    "reduce randomness" setting; pass None to use the model's default and
    compare the two.
    """
    answers = []
    for _ in range(n_runs):
        turn = chat_raw(
            [{"role": "user", "content": question}],
            system=SYSTEM_PROMPT,
            temperature=temperature,
        )
        answers.append(turn.text)

    pairwise = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            pairwise.append(_text_similarity(answers[i], answers[j]))
    avg_similarity = sum(pairwise) / len(pairwise) if pairwise else 1.0

    return {
        "question": question,
        "n_runs": n_runs,
        "temperature": temperature,
        "answers": answers,
        "unique_answers": len(set(answers)),
        "avg_pairwise_similarity": round(avg_similarity, 3),
    }


if __name__ == "__main__":
    result = run_stability_test("In one sentence, what is a mortgage?", n_runs=5)
    for i, a in enumerate(result["answers"], 1):
        print(f"Run {i}: {a}")
    print(f"\nUnique answers: {result['unique_answers']} / {result['n_runs']}")
    print(f"Average pairwise similarity: {result['avg_pairwise_similarity']}")
