# Smriti — Retrieval and Memory as Operational Components, and Where Non-Determinism Enters

*Smriti (स्मृति) means "memory." This lab treats memory the same way
Nirikshan treated a model call: as something to open up and inspect, not
something to take on faith — alongside a close look at retrieval's own
failure modes, and a direct measurement of how non-deterministic a model
really is.*

## 1. What Changed Since Nirikshan (the Previous Session)

| | Nirikshan | Smriti |
|---|---|---|
| Core question | What's really in a model call? | What's really in retrieval and memory? |
| State across requests | None — every `/ask` call was independent | `/chat` remembers, per `session_id`, via `memory_store.py` |
| Retrieval | One mode: query the real index | Four modes: `normal`, `stale`, `empty`, `oversized` — on demand |
| New failure class explored | Truncated answers (`finish_reason`) | Stale/empty/oversized retrieval, memory overflow, non-determinism |
| New endpoints | `/ask`, `/inspect` | `/chat`, `/memory/{id}`, `/memory/{id}/reset`, `/retrieval-stress`, `/stability` |
| Reused unchanged | — | `loaders.py`, `pii.py`, `chunking.py`, `context.py`, `pricing.py`, `anatomy.py`, `tracer.py`, `build_vectorstore.py`, `inspect_vectorstore.py` |

Same 5 documents, same domain, same underlying agent — Smriti is Nirikshan
with memory added and retrieval given a set of failure switches to flip
on demand.

## 2. The Five Documents, Five Formats

Unchanged from Nirikshan — see that lab's documentation for the full
table. All five live in `documents/` exactly as before.

## 3. Architecture — the Full Flow

```
STEP 1 -- build_vectorstore.py  (identical to Nirikshan; run once)

    documents/*.{pdf,docx,html,csv,json} -> loaders -> pii -> chunking -> OpenAI embeddings -> ./chroma_store

STEP 2 -- app.py  (FastAPI backend, keep running in Terminal 1)

    POST /chat {session_id, message}
        1. memory_store.py        -- append the new message to this
                                      session's remembered turns
        2. the same tool-calling agent loop as Nirikshan's /ask, now
           fed the WHOLE remembered conversation, not just one message
        3. tracer.py               -- every model call / tool call timed
        4. memory_store.py again   -- append the answer, then truncate
                                       if the session exceeds 8 turns
        5. return JSON              -- {answer, trace, memory}

    GET /memory/{session_id}        -- read-only view of a session's turns
    POST /memory/{session_id}/reset -- clear a session, no side effects elsewhere

    GET /retrieval-stress?q=...&mode=normal|stale|empty|oversized
        -- calls tools.py's lookup_policy directly, no model involved,
           in whichever failure mode you ask for

    GET /stability?q=...&n_runs=5&temperature=...
        -- nondeterminism.py asks the IDENTICAL question n_runs times
           and reports how much the wording actually varied

STEP 3 -- agent.py  (Terminal 2, while app.py keeps running)

    5 domain questions, ONE shared session -- watch memory grow turn by
    turn -- then a 6th follow-up question ("that first one...") that
    ONLY resolves correctly because memory carried the earlier answers
    forward, THEN:
      BONUS 1: inspect memory, reset it, then overflow a fresh session
               on purpose and watch truncation fire
      BONUS 2: the same query through all four retrieval modes side by
               side -- normal, stale, empty, oversized
      BONUS 3: the identical question run 5 times, with a measured
               similarity score across the answers

STEP 4 (optional) -- inspect_vectorstore.py  (identical to Nirikshan)
```

## 4. Setup — Step by Step

Identical to Nirikshan's Steps 1–3 (clone, `.env`, virtual environment,
`pip install -r requirements.txt`). Then:

### Step 4 — Build the vector store (skip if reusing Nirikshan's)

```
python build_vectorstore.py
```

### Step 5 — Start Smriti (Terminal 1)

```
uvicorn app:app --reload --port 8000
```

### Step 6 — Run the lab (Terminal 2)

```
python agent.py
```

## 5. What to Notice While Running This

- **Memory is just a list that gets resent.** Read `app.py`'s `/chat`
  handler: there is no special "remembering" mechanism — every call
  rebuilds the full message list from `session.turns` and sends the
  whole thing again. "Memory" is entirely this resend, nothing more.
- **Unbounded memory is a silent cost leak, not an error.** Before
  `truncate()` runs, every additional turn makes every future call in
  that session bigger, and therefore slower and more expensive — with no
  warning, right up until a context window limit is hit. Watch
  `agent.py`'s overflow demo and check `truncated_count_lifetime`.
- **Stale retrieval is the most dangerous failure, not the most obvious
  one.** `empty` mode visibly fails — the assistant has to say it
  doesn't know. `stale` mode returns a confident, plausible, WRONG rate
  with nothing marking it as outdated. Compare the two in Bonus 2.
- **Non-determinism is measurable, not just a vibe.** Bonus 3's
  `avg_pairwise_similarity` turns "the wording changes sometimes" into
  an actual number you can track run to run, or compare before and after
  a prompt change.
- **Which parts of the system are actually deterministic.**
  `calculate_monthly_repayment` and `lookup_policy`'s retrieval step
  (against an unchanged index) both return identical output for
  identical input, every time. Only the model's own generation varies —
  confirm this yourself by running `/retrieval-stress` with the same
  query twice and diffing the output.

## 6. A Note on Session Storage

`memory_store.py` is a plain in-process Python dictionary. Restarting
`app.py` erases every session instantly — there is no database, no
disk persistence, nothing durable. That's deliberate for a lab, and also
exactly the gap a real deployment has to close: a genuine production
memory layer needs its own persistence, its own eviction policy, and its
own access control, none of which a dictionary gives you for free.

## 7. Troubleshooting

| Problem | Likely cause |
|---|---|
| `/chat` fails with "OPENAI_API_KEY is not set" | Copy `.env.example` to `.env` and fill in your key |
| `/chat` fails with "vector store is empty" | Run `build_vectorstore.py` first |
| A follow-up question doesn't use earlier context | Confirm you're reusing the SAME `session_id` across calls — a new session has no memory of a different one |
| `/memory/{id}` returns `"found": false"` right after a `/chat` call | Check you used the exact same `session_id` string in both calls — it's case-sensitive |
| `/stability`'s answers all look identical every single run | Try a longer, more open-ended question — short factual answers naturally vary less |
| `truncated_count_lifetime` never increases | You haven't yet sent enough turns in one session to exceed `MAX_TURNS_DEFAULT` (8) — see `agent.py`'s overflow demo for how to trigger it |

## 8. Sample Assignment Outline (for Participants)

**Assignment: Build Anukriti, a Consistency-Grading Pipeline**

1. Starting from a copy of this repo, add a `GET /stability-compare` endpoint
   that runs `/stability`'s test twice on the same question — once at
   `temperature=0` and once at the model's default — and returns both
   results side by side.
2. Add a simple grading rule to `nondeterminism.py`: classify a question's
   stability as `"stable"` if `avg_pairwise_similarity >= 0.85`,
   `"moderate"` if between 0.6 and 0.85, otherwise `"unstable"`.
3. Run the grading rule across at least 4 different questions, including
   at least one factual lookup-style question and one open-ended
   explanatory question. Record each question's grade.
4. Extend `/memory/{session_id}` with a new field, `estimated_tokens_if_resent`,
   estimating (roughly — word count times a small constant is fine) how
   many tokens the NEXT `/chat` call in that session would send, purely
   from accumulated memory, before any tool results are added.
5. **Written reflection (150–200 words):** Did factual or open-ended
   questions turn out more stable? Does that match your intuition about
   where non-determinism "enters" the system? What would you change
   about the system prompt to make an unstable question more stable,
   without simply setting `temperature=0` everywhere?

**Definition of done:**

- A working `/stability-compare` endpoint, tested on at least one question
- The stable / moderate / unstable grading rule, applied to at least 4 questions with results recorded
- `estimated_tokens_if_resent` present and reasonable in `/memory/{session_id}`
- A written reflection answering the question in step 5

## 9. Where This Goes Next

Today's memory is a dictionary; today's non-determinism measurement is a
script you run by hand, five times, and eyeball. Module 4 (Observability
Platforms and Pipelines) and Module 5 (Production Evaluation and Drift)
replace both: memory becomes a session store a real platform can inspect
across every live conversation at once, and non-determinism stops being
something you check occasionally and becomes a quality signal — drift —
that a dashboard watches continuously and alerts on when it moves outside
an agreed band.
