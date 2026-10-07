"""
app.py — Smriti's FastAPI backend. Keep this running in Terminal 1.

Five endpoints:

  POST /chat {session_id, message}
      The same tool-calling agent loop as Nirikshan's /ask, now with
      conversation MEMORY: every turn is appended to memory_store, keyed
      by session_id, and resent on the next call in the same session —
      so the model can answer "and what about the second one?" correctly.
      Memory is truncated automatically beyond MAX_TURNS_DEFAULT turns.

  GET /memory/{session_id}
      Read-only view of exactly what's stored for a session right now —
      every turn, and how many have been truncated away over its life.

  POST /memory/{session_id}/reset
      Clears a session's memory. An operational control, not magic.

  GET /retrieval-stress?q=...&mode=normal|stale|empty|oversized
      Calls lookup_policy directly, in whichever failure mode you ask
      for, with no model call involved — see retrieval misbehave on its
      own, isolated from everything downstream of it.

  GET /stability?q=...&n_runs=5&temperature=0
      Runs nondeterminism.run_stability_test — the same question asked
      n_runs times, with a measured similarity score across the answers.

RUN IT:
    uvicorn app:app --reload --port 8000
Then visit http://localhost:8000/docs, or use agent.py from Terminal 2.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Query

from llm_provider import chat_raw
from memory_store import MAX_TURNS_DEFAULT, store
from nondeterminism import run_stability_test
from pricing import estimate_cost_usd
from tools import TOOL_FUNCTIONS, TOOL_SPECS, lookup_policy
from tracer import Trace

load_dotenv()

app = FastAPI(title="Smriti", description="Retrieval and memory as operational components, and where non-determinism enters.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a customer service assistant for Lloyds Financial (a training scenario, not a real "
    "bank). Use lookup_policy for any question about rates, limits, eligibility, fraud handling, "
    "complaints, or savings products — never answer those from memory. Use "
    "calculate_monthly_repayment for repayment arithmetic. If lookup_policy finds no good match, "
    "say so plainly. You may refer back to earlier turns in this conversation. Keep answers short."
)


def _run_traced_agent_turn(messages: list[dict]) -> tuple[str, Trace]:
    """Shared agent loop: runs until a final answer, returns (answer, trace). `messages` grows in place."""
    trace = Trace()
    answer = None

    for _ in range(MAX_AGENT_STEPS):
        turn = chat_raw(messages, tools=TOOL_SPECS, system=SYSTEM_PROMPT)
        usage = turn.raw.usage
        cost = estimate_cost_usd(MODEL, usage.prompt_tokens, usage.completion_tokens) or 0.0
        trace.record(
            kind="model_call", name=MODEL, duration_ms=turn.duration_ms,
            input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, cost_usd=cost,
        )

        if not turn.tool_calls:
            answer = turn.text
            messages.append({"role": "assistant", "content": turn.text})
            break

        messages.append({
            "role": "assistant", "content": turn.text,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls],
        })
        for tc in turn.tool_calls:
            fn = TOOL_FUNCTIONS.get(tc.name)
            t0 = time.perf_counter()
            try:
                result = fn(**tc.arguments) if fn else f"Error: no such tool '{tc.name}'."
            except Exception as exc:
                result = f"Error running {tc.name}: {exc}"
            tool_ms = (time.perf_counter() - t0) * 1000
            trace.record(kind="tool_call", name=tc.name, duration_ms=tool_ms)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})

    if answer is None:
        answer = "(Stopped after repeated tool calls with no final answer — a runaway-loop pattern.)"
    return answer, trace


@app.post("/chat")
def chat(session_id: str = Query(...), message: str = Query(...)) -> dict:
    session = store.get_or_create(session_id)
    session.append("user", message)

    # Rebuild the message list the agent loop needs from this session's
    # remembered turns — memory IS this list, nothing more.
    messages = [{"role": t["role"], "content": t["content"]} for t in session.turns]

    answer, trace = _run_traced_agent_turn(messages)

    session.append("assistant", answer)
    dropped = session.truncate(MAX_TURNS_DEFAULT)

    return {
        "session_id": session_id,
        "answer": answer,
        "trace": trace.as_dict(),
        "memory": {
            "turn_count": len(session.turns),
            "turns_dropped_this_call": dropped,
            "turns_dropped_lifetime": session.truncated_count,
        },
    }


@app.get("/memory/{session_id}")
def get_memory(session_id: str) -> dict:
    result = store.inspect(session_id)
    if result is None:
        return {"session_id": session_id, "found": False}
    return {"found": True, **result}


@app.post("/memory/{session_id}/reset")
def reset_memory(session_id: str) -> dict:
    existed = store.reset(session_id)
    return {"session_id": session_id, "existed": existed, "reset": True}


@app.get("/retrieval-stress")
def retrieval_stress(
    q: str = Query(...),
    mode: str = Query("normal", description="normal | stale | empty | oversized"),
) -> dict:
    start = time.perf_counter()
    result = lookup_policy(q, mode=mode)
    duration_ms = (time.perf_counter() - start) * 1000
    return {
        "query": q,
        "mode": mode,
        "retrieved_context": result,
        "context_chars": len(result),
        "duration_ms": round(duration_ms, 1),
    }


@app.get("/stability")
def stability(
    q: str = Query(...),
    n_runs: int = Query(5, ge=2, le=20),
    temperature: float | None = Query(None),
) -> dict:
    return run_stability_test(q, n_runs=n_runs, temperature=temperature)


@app.get("/")
def root() -> dict:
    return {
        "service": "Smriti",
        "endpoints": [
            "/chat (POST, session_id + message)",
            "/memory/{session_id}",
            "/memory/{session_id}/reset (POST)",
            "/retrieval-stress?q=...&mode=normal|stale|empty|oversized",
            "/stability?q=...&n_runs=5&temperature=0",
            "/docs",
        ],
    }
