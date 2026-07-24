# AI Interactions & Auditability Log

**Project:** CodePath AI110 · Project 4 — "Show What You Know: Applied AI System"
**System:** PawPal+ Applied AI Advice Agent ([main.py](main.py))

This log documents (1) how AI was prompted during development, (2) the
intermediate reasoning/design decisions, and (3) **guardrail decision traces**
emitted by the running system — so a reviewer can audit *why* each response was
produced.

---

## 1. Prompt structure used with the AI assistant

The build followed a **context → constraint → verify** prompting pattern:

1. **Context first.** Point the assistant at the existing Module 2 repo
   (`pawpal_system.py`, `app.py`, `uml.mmd`) so the extension stays consistent
   with the base project rather than inventing a new domain.
2. **Explicit constraints.** State the required contract up front: a
   `pawpal_agent(query)` returning JSON with `status`, `confidence_score`,
   `retrieved_context`, `advice`; guardrail must **fail closed**; stdlib only
   (reproducible, no API keys); two runnable demo cases.
3. **Verify every step.** After each change, run `python main.py` and read the
   structured log + reliability metrics instead of trusting code by inspection.

Representative prompt skeleton:

```text
CONTEXT:  Extend the Module 2 "PawPal+" pet-care scheduler. Base classes live
          in pawpal_system.py. Keep it thematically consistent (pet care).
TASK:     Add a RAG advice layer + a toxicity safety guardrail.
CONTRACT: pawpal_agent(query) -> dict{status, confidence_score,
          retrieved_context, advice}; pipeline order Query -> RAG -> Guardrail
          -> Output; guardrail overrides advice on a toxin (fail closed);
          structured logging; stdlib only.
VERIFY:   Show me the exact stdout/stderr of `python main.py`.
```

---

## 2. Intermediate reasoning steps (design decisions & trade-offs)

| # | Decision point | Reasoning / trade-off | Outcome |
|---|---|---|---|
| 1 | Pipeline order | Follow the rubric sequence `Query → RAG → Guardrail → Output`. Safety is preserved by a **fail-closed override** (not by ordering): the guardrail replaces the advice when a toxin is found. | RAG retrieval is **stage 1**; the Guardrail Evaluator (**stage 2**) overrides the result on a hazard, keeping the RAG candidate in `metadata`. |
| 2 | Retrieval scoring | Needed an explainable, deterministic similarity in `[0,1]` without embeddings/deps. | **Otsuka–Ochiai cosine** over keyword sets: `|A∩B| / √(|A|·|B|)`. |
| 3 | Low-similarity handling | A weak RAG match should not masquerade as confident advice. | Below `RETRIEVAL_THRESHOLD = 0.15` → safe "consult a vet" fallback (still `SUCCESS`). |
| 4 | Matching strategy | Substring matching false-positives (`garlic` in `garlicky`); over-broad aliases (`energy`) block safe queries. | **Whole-word token** matching + a *precise* alias list. |
| 5 | Reproducibility | Grading needs identical, inspectable evidence. | **Stdlib only**, no randomness, no network → deterministic output. |
| 6 | Preserve base project | Overwriting `main.py` would destroy the Module 2 CLI demo. | Copied it to `schedule_demo.py`; scheduling logic in `pawpal_system.py` untouched. |
| 7 | Confidence semantics | One field, two branches. | `confidence_score` = guardrail detection confidence when BLOCKED; retrieval similarity when SUCCESS. |

**Errors caught during self-review** (see [model_card.md](model_card.md) §AI
Collaboration): over-broad xylitol/caffeine aliases were trimmed, and a
copy-pasted wrong `name` on the avocado toxin record was corrected.

---

## 3. Guardrail decision traces (auditability)

Every pipeline run emits structured `event=… key=value` log records to *stderr*.
These are the actual traces from `python main.py`.

### Trace A — SAFE query (RAG → guardrail passes → SUCCESS)

Query: `"What's a good daily exercise routine for a Labrador?"`

```text
event=query_received    query="What's a good daily exercise routine for a Labrador?"
event=rag_retrieved     doc_id="KB-EXERCISE-01" title="Daily exercise routines for dogs" similarity=0.471
event=guardrail_passed  similarity=0.471
event=pipeline_complete status="SUCCESS" confidence=0.471
```

**Decision path:** `retrieve()` scored `KB-EXERCISE-01` at `0.471` →
`screen_for_toxins()` returned `[]` (guardrail passed) → `0.471 ≥ 0.15` →
grounded advice returned.

### Trace B — TOXIC query (RAG runs first, then guardrail overrides → BLOCKED)

Query: `"Can I give my dog chocolate as a treat?"`

```text
event=query_received    query="Can I give my dog chocolate as a treat?"
event=rag_retrieved     doc_id="KB-HYDRATION-01" title="Hydration and water intake" similarity=0.167
event=guardrail_blocked detected=["Chocolate"] matched_terms=["chocolate"] confidence=0.99
event=pipeline_complete status="BLOCKED_BY_GUARDRAIL" confidence=0.99
```

**Decision path:** RAG retrieved a weak candidate (`KB-HYDRATION-01`, `0.167`)
first → the Guardrail Evaluator matched token `chocolate` to the `Chocolate`
record (confidence `0.99`) → the guardrail **overrode** the retrieved advice
with an emergency medical warning + ASPCA poison-control number. The RAG
candidate is preserved in `metadata.rag_candidate` for audit. Note that
`rag_retrieved` fires **before** `guardrail_blocked`, confirming the rubric
order `Query → RAG → Guardrail → Output`.

### Trace C — Reliability self-check (audit summary)

```text
event=reliability_check total_cases=14 overall_accuracy=1.0 toxin_block_rate=1.0
                        toxins_blocked="9/9" false_positive_rate=0.0
                        safe_queries_passed="5/5" knowledge_base_size=7 toxin_index_size=8
```

**Interpretation:** on the 14-case labeled set, all 9 toxic queries blocked and
all 5 safe queries passed. Scope caveats are documented in
[model_card.md](model_card.md).

---

## 4. How to reproduce this trace

```bash
python main.py            # JSON responses -> stdout; decision traces -> stderr
python main.py 2>trace.log   # capture only the auditable log trace
```

Because the system is deterministic (stdlib only, no randomness), the traces
above reproduce exactly on every run (timestamps aside).
