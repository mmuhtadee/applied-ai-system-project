# Model Card — PawPal+ Applied AI Advice Agent

**Project:** CodePath AI110 · Project 4 — "Show What You Know: Applied AI System"
**Base project:** PawPal+ (Module 2)
**System type:** Retrieval-Augmented Generation (RAG) advice agent with a
fail-closed safety/toxicity guardrail.
**Implementation:** [main.py](main.py) — Python standard library only,
deterministic, no external model or API in the current build.

> **Terminology.** The current build ships a *mock* retriever and a
> deterministic, rule-based "generator" (template formatting) so results are
> fully reproducible for grading. This card describes both the system as built
> **and** the risks that apply once the mock generator is replaced by a real LLM
> (e.g. Claude) in production.

---

## Limitations & Biases

**Knowledge & coverage limits**

- **Tiny knowledge base.** The RAG index is **7 hand-written documents**
  covering exercise, diet, hydration, grooming, dental, vaccination, and
  enrichment. Anything outside these topics returns a low-confidence
  "consult a vet" fallback — it does not fabricate an answer, but it also
  cannot help.
- **Species/breed bias.** Content skews toward **dogs and cats**, with a dog
  lean in several documents (e.g. the exercise doc references Labradors).
  Birds, reptiles, and small mammals are underserved; toxicity for those
  species is only partially represented (e.g. avocado is flagged mainly via
  the bird risk).
- **Keyword retrieval is shallow.** Retrieval uses an Otsuka–Ochiai cosine
  score over **bag-of-keyword sets**, not semantic embeddings. It has no
  synonym or paraphrase understanding: a query using vocabulary absent from a
  document's keyword set scores low even when semantically relevant. Long
  queries are penalized because the score normalizes by query length.
- **Region/units bias.** Guidance uses metric units and US emergency contacts
  (ASPCA Animal Poison Control, 888-426-4435). Non-US users get a less useful
  emergency referral.

**Guardrail limits**

- **Whole-word token matching only.** The guardrail matches single tokens on
  word boundaries. This creates a genuine **precision/recall trade-off** that
  adversarial probing during development confirmed:
  - **False negatives (missed toxins):** inflected or adjectival forms are
    distinct tokens, so `"garlicky bread"` and `"chocolatey treat"` slip
    through as `SUCCESS` even though `garlic`/`chocolate` are listed. It also
    misses misspellings (`chocolat`) and hidden ingredients (`sugar-free gum`
    containing **xylitol**).
  - **False positives (benign over-blocks):** compound nouns that merely
    contain a toxin token get blocked — `"coffee table"` → Caffeine,
    `"root beer"` → Alcohol, `"wine rack"` → Alcohol.
  We deliberately tuned the alias list toward *precision* (removing broad
  tokens like `energy` and `sweetener`) because for a safety gate an
  occasional over-block is safer than a miss — but this is a documented,
  measured trade-off, not a solved problem. A production build would replace
  keyword matching with an embedding classifier plus context disambiguation.
- **Reliability numbers are scoped to canonical phrasing.** The 100% / 0%
  figures below are measured on a curated 14-case set that uses canonical
  toxin words and clearly-benign queries; it does **not** include the inflected
  false-negatives or compound-noun false-positives above, so it overstates
  real-world robustness (see *Reliability & Testing Observations*).
- **Not a substitute for a vet.** A `SUCCESS` result is **not** a safety
  guarantee; it only means no *known listed* toxin was detected.

---

## Misuse & Risk Mitigation

| Risk | Mitigation in this system |
|---|---|
| User acts on unsafe "advice" (e.g. feeds a toxic food) | **Fail-closed guardrail override.** The pipeline runs RAG retrieval then the guardrail (rubric sequence `Query → RAG → Guardrail → Output`); on a detected toxin the guardrail **overrides** the retrieved advice with an emergency medical warning + poison-control number, so casual advice is never emitted for a hazardous query. *(Design note: because the guardrail runs after retrieval, RAG executes even for blocked queries; a production LLM build would gate before the expensive generation step — see below.)* |
| Over-trust / automation bias | Every `SUCCESS` response appends an explicit "general guidance; consult a licensed veterinarian" disclaimer, and the README carries a prominent disclaimer. |
| Hallucinated advice | Advice is **grounded**: `retrieved_context` always cites the source document ID, and low-similarity queries fall back to "consult a vet" instead of inventing content. |
| Silent failure / no audit trail | **Structured logging** records every stage in order (`query_received`, `rag_retrieved`, `guardrail_passed`/`guardrail_blocked`, `pipeline_complete`) with confidence metrics for post-hoc audit — see [ai_interactions.md](ai_interactions.md). |
| Prompt-injection / jailbreak (future LLM build) | The guardrail is a **deterministic gate independent of any model prompt**, so it cannot be talked out of blocking by the query text. A production build should keep the guardrail outside the LLM and add an output-side check. |
| Emergency mishandling | Blocked responses give an actionable next step (contact vet / ASPCA APCC) rather than only saying "no". |

**Explicitly out of scope / not mitigated:** dosing or treatment instructions,
diagnosis, non-listed toxins, and non-US emergency routing. These require a
licensed veterinarian.

---

## Reliability & Testing Observations

Measured by `run_reliability_check()` in [main.py](main.py) over a 14-case
labeled test set (reproduce with `python main.py`):

| Metric | Result |
|---|---|
| Overall accuracy | **100% (14/14)** |
| Toxin block rate | **100% (9/9)** — chocolate, grapes, garlic, xylitol, onions, macadamia, coffee, alcohol, avocado |
| False-positive rate | **0% (0/5)** — exercise, grooming, hydration, vaccination, enrichment all pass |
| RAG retrieval (safe demo) | Correct document `KB-EXERCISE-01` at similarity `0.471` |
| Determinism | 100% reproducible across runs (pure stdlib, no randomness) |

**Honest reading of these numbers.** 100% is measured on a **small, curated,
in-repo** test set that was written alongside the toxin list — so it confirms
the guardrail behaves correctly on its *known* targets and does not over-block
the *known* safe queries. It is **not** evidence of exhaustive real-world
coverage. A realistic evaluation would need adversarial/misspelled inputs,
hidden-ingredient cases, and a much larger, independently sourced test set —
where recall would fall (see Limitations). This build also complements the
Module 2 scheduling suite ([tests/test_pawpal.py](tests/test_pawpal.py), 21
tests).

---

## AI Collaboration Reflection

This project was built collaboratively with an AI coding assistant (Claude).
Two concrete moments illustrate the value **and** the necessity of human
verification.

### ✅ One helpful AI suggestion

**"Make the guardrail a *fail-closed override*, and match toxins on word
boundaries."** An early sketch treated the guardrail as advisory (append a
warning to the RAG advice) and matched toxin names with naive substring search.
The AI recommended (a) a **fail-closed override**: when a toxin is detected, the
guardrail *replaces* the retrieved advice with an emergency warning and flips
`status` to `BLOCKED_BY_GUARDRAIL`, so casual advice can never reach the user
for a hazardous query — this holds regardless of pipeline order (the final
pipeline follows the rubric sequence `Query → RAG → Guardrail → Output`); and
(b) **word-boundary token matching** rather than substring search, so
`chocolate` matches as a whole word instead of firing on unrelated substrings.
This shaped the final Guardrail Evaluator stage (see
[diagrams/architecture.mmd](diagrams/architecture.mmd)). **Verified** by the
reliability self-check: 9/9 toxins blocked, 0/5 false positives.

### ❌ One flawed / incorrect AI suggestion

**Over-broad toxin aliases that would have caused false positives — plus a
copy-paste data error.** The AI's first toxin table included aliases like
`"energy"` (for caffeine/energy drinks) and `"sweetener"`/`"sugarfree"`/`"birch"`
(for xylitol). These are **too broad**: a perfectly safe query like *"my dog
has lots of energy, how much exercise does he need?"* contains the token
`energy` and would have been **wrongly BLOCKED**, and `"sugarfree"` never even
matches the hyphenated `sugar-free` after tokenization — so it was both unsafe
*and* ineffective. Separately, the avocado entry was initially mislabeled with a
copy-pasted `name` (`"Macadamia / other toxic foods…"`), which would have
printed the wrong hazard name to users. **Both were caught on review and
corrected**: aliases were trimmed to precise, high-signal tokens and the name
was fixed. Lesson: AI-generated *data tables* need the same scrutiny as
AI-generated *code* — a plausible-looking list can quietly encode both
false-positive and mislabeling bugs.

**How suggestions were verified overall:** by running `python main.py` after
every change and reading the structured log trace + reliability metrics, rather
than trusting the code by inspection alone.
