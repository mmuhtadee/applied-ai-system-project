"""
PawPal+ — Applied AI Advice Agent
=================================

Project 4: "Show What You Know: Applied AI System" (CodePath AI110)

This module EXTENDS the Module 2 project "PawPal+" (a Streamlit pet-care task
scheduler built around the ``Owner`` / ``Pet`` / ``Task`` / ``Scheduler``
classes in ``pawpal_system.py``) with an Applied-AI advice layer.

Where Module 2 answered *"when should I do my pet-care tasks?"*, this layer
answers *"what should I do, and is it safe?"* using two AI-system building
blocks required by Project 4:

  1. Retrieval-Augmented Generation (RAG)
     A mock Pet-Care Knowledge Base is indexed and queried so that every piece
     of advice is *grounded* in a retrieved source document rather than
     free-floating model text.

  2. Safety / Toxicity Guardrail
     A deterministic guardrail inspects each query for substances that are
     toxic to pets (chocolate, grapes, onions, garlic, xylitol, ...). If a
     hazard is detected the pipeline *fails closed*: the guardrail overrides the
     retrieved advice and instead returns an emergency medical warning.

The pipeline follows the rubric sequence
``Query -> RAG Retrieval -> Guardrail Evaluator -> Response Output``.

The public entry point is ``pawpal_agent(query) -> dict``, which returns a
structured, JSON-serialisable result with the keys ``status``,
``confidence_score``, ``retrieved_context`` and ``advice``.

Run ``python main.py`` to execute the two required demo cases (one safe query,
one toxic query) plus a guardrail reliability self-check.

Design note
-----------
The system is intentionally self-contained (Python standard library only) so
that execution evidence is fully reproducible without API keys or network
access. In production the mock retriever/generator would be swapped for a real
vector store and a hosted LLM (e.g. Claude), but the guardrail would remain a
deterministic gate independent of the generation model. See ``model_card.md``
for the full discussion.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #
#: Minimum retrieval similarity (Otsuka-Ochiai cosine over keyword sets) below
#: which we treat the RAG hit as "no confident match" and fall back to a safe,
#: generic "consult a vet" response instead of asserting grounded advice.
RETRIEVAL_THRESHOLD: float = 0.15

#: US emergency resources surfaced by the safety guardrail.
ASPCA_POISON_CONTROL = "ASPCA Animal Poison Control Center (888-426-4435)"


# --------------------------------------------------------------------------- #
# Structured logging  (requirement #4)
# --------------------------------------------------------------------------- #
logger = logging.getLogger("pawpal.agent")


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single structured stream handler to the agent logger.

    Idempotent: safe to call from every entry point without duplicating
    handlers. Log records carry an ISO-8601 timestamp, a level, the logger
    name, and a machine-parseable ``event=... key=value`` message body so the
    pipeline execution trace can be audited (see ``ai_interactions.md``).
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Emit one structured log record: an event name plus key=value metrics."""
    payload = " ".join(f"{k}={json.dumps(v, default=str, ensure_ascii=False)}" for k, v in fields.items())
    logger.log(level, "event=%s %s", event, payload)


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
#: Common function words dropped before scoring so retrieval focuses on the
#: content-bearing terms of a query.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "am", "be", "can", "could", "should",
    "would", "will", "do", "does", "did", "have", "has", "my", "our", "your",
    "i", "we", "you", "to", "for", "of", "and", "or", "in", "on", "at", "it",
    "its", "with", "as", "so", "if", "that", "this", "what", "whats", "how",
    "when", "why", "give", "get", "got", "good", "best", "some", "any", "much",
    "many", "there", "here",
})


def _tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> set:
    """Content-bearing tokens (stopwords and single characters removed)."""
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1}


# --------------------------------------------------------------------------- #
# 1. Mock Pet-Care Knowledge Base  (RAG index — requirement #1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KBDocument:
    """A single factual pet-care document in the retrieval index."""

    doc_id: str
    title: str
    keywords: frozenset
    content: str


KNOWLEDGE_BASE: List[KBDocument] = [
    KBDocument(
        doc_id="KB-EXERCISE-01",
        title="Daily exercise routines for dogs",
        keywords=frozenset({
            "exercise", "walk", "walking", "walks", "routine", "daily",
            "activity", "active", "energy", "play", "playtime", "fetch",
            "dog", "dogs", "puppy", "labrador", "retriever", "physical",
        }),
        content=(
            "Adult dogs generally need 30-120 minutes of exercise per day "
            "depending on breed and age. High-energy working breeds such as "
            "Labrador Retrievers do best with two walks per day plus active "
            "play (fetch, tug, or swimming). Split activity into a morning and "
            "an evening session, provide water breaks, and avoid strenuous "
            "exercise during peak afternoon heat."
        ),
    ),
    KBDocument(
        doc_id="KB-DIET-01",
        title="Feeding and portion basics",
        keywords=frozenset({
            "feed", "feeding", "food", "diet", "portion", "portions", "meal",
            "meals", "nutrition", "kibble", "treats", "overfeeding", "weight",
            "dog", "cat", "puppy", "kitten",
        }),
        content=(
            "Feed a complete, life-stage-appropriate diet and measure portions "
            "with a cup or scale rather than free-feeding. Most adult dogs and "
            "cats do well on two measured meals per day. Keep treats under "
            "about 10% of daily calories, and transition between foods over "
            "5-7 days to avoid stomach upset."
        ),
    ),
    KBDocument(
        doc_id="KB-HYDRATION-01",
        title="Hydration and water intake",
        keywords=frozenset({
            "water", "hydration", "hydrated", "drink", "drinking", "thirst",
            "thirsty", "bowl", "fluids", "dehydration", "dog", "cat",
        }),
        content=(
            "As a rough guide, dogs and cats need roughly 50-60 ml of water per "
            "kilogram of body weight per day, more in hot weather or after "
            "exercise. Always keep fresh, clean water available. Reduced "
            "drinking, dry gums, or skin that is slow to spring back can signal "
            "dehydration and warrant a call to your veterinarian."
        ),
    ),
    KBDocument(
        doc_id="KB-GROOMING-01",
        title="Grooming and coat care",
        keywords=frozenset({
            "groom", "grooming", "brush", "brushing", "coat", "fur", "shed",
            "shedding", "bath", "bathe", "nails", "mats", "matting",
            "longhaired", "long", "haired", "cat", "dog",
        }),
        content=(
            "Brushing frequency depends on coat type: short coats need weekly "
            "brushing, while long-haired cats and dogs benefit from daily "
            "brushing to prevent mats. Bathe only when needed with a pet-safe "
            "shampoo, and trim nails every 3-4 weeks so they do not click on "
            "the floor."
        ),
    ),
    KBDocument(
        doc_id="KB-DENTAL-01",
        title="Dental care",
        keywords=frozenset({
            "dental", "teeth", "tooth", "brush", "brushing", "gums", "breath",
            "tartar", "plaque", "chew", "chews", "mouth", "dog", "cat",
        }),
        content=(
            "Daily tooth brushing with a pet-formulated toothpaste (never human "
            "toothpaste) is the gold standard for dental health. Dental chews "
            "and a yearly veterinary cleaning help control plaque and tartar. "
            "Persistent bad breath, red gums, or difficulty eating should be "
            "checked by a veterinarian."
        ),
    ),
    KBDocument(
        doc_id="KB-VET-01",
        title="Vaccination and preventive care schedule",
        keywords=frozenset({
            "vaccine", "vaccines", "vaccinated", "vaccination", "shots",
            "puppy", "kitten", "vet", "veterinarian", "checkup", "deworming",
            "preventive", "boosters", "schedule", "dog", "cat",
        }),
        content=(
            "Puppies and kittens typically begin a vaccine series around 6-8 "
            "weeks of age with boosters every 3-4 weeks until roughly 16 weeks, "
            "followed by adult boosters on a schedule set by your veterinarian. "
            "Pair vaccinations with parasite prevention and an annual wellness "
            "exam. Your veterinarian tailors the exact schedule to your pet."
        ),
    ),
    KBDocument(
        doc_id="KB-ENRICHMENT-01",
        title="Mental enrichment and behaviour",
        keywords=frozenset({
            "enrichment", "bored", "boredom", "mental", "stimulation", "toys",
            "puzzle", "training", "behaviour", "behavior", "chewing", "anxiety",
            "scratching", "dog", "cat",
        }),
        content=(
            "Mental enrichment reduces boredom-driven behaviours like "
            "destructive chewing or excessive scratching. Rotate puzzle "
            "feeders and toys, run short daily training sessions, and give cats "
            "vertical space and scratching posts. Enrichment complements, but "
            "does not replace, physical exercise."
        ),
    ),
]


def retrieve(query: str, top_k: int = 1) -> List[Tuple[float, KBDocument]]:
    """Return the ``top_k`` best-matching documents with similarity scores.

    Similarity is the Otsuka-Ochiai coefficient (cosine similarity over binary
    bag-of-keyword vectors): ``|A ∩ B| / sqrt(|A| * |B|)`` where ``A`` is the
    set of content tokens in the query and ``B`` is a document's keyword set.
    The score lies in ``[0, 1]`` and is easy to explain in a reliability report.
    """
    q = _content_tokens(query)
    scored: List[Tuple[float, KBDocument]] = []
    for doc in KNOWLEDGE_BASE:
        overlap = len(q & doc.keywords)
        score = overlap / sqrt(len(q) * len(doc.keywords)) if overlap and q else 0.0
        scored.append((score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


# --------------------------------------------------------------------------- #
# 2. Safety & Toxicity Guardrail Evaluator  (requirement #2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Toxin:
    """A substance known to be hazardous to pets, plus how to detect it."""

    name: str
    aliases: frozenset
    severity: str          # "mild" | "moderate" | "severe"
    species: str
    mechanism: str
    symptoms: str
    detection_confidence: float


#: Curated from widely published veterinary toxicology guidance (ASPCA / Pet
#: Poison Helpline). Aliases are single tokens matched on word boundaries to
#: avoid substring false positives.
TOXIN_INDEX: List[Toxin] = [
    Toxin(
        name="Chocolate",
        aliases=frozenset({"chocolate", "cocoa", "cacao"}),
        severity="severe",
        species="dogs and cats",
        mechanism="theobromine and caffeine (methylxanthine) toxicity",
        symptoms="vomiting, diarrhea, restlessness, rapid heart rate, tremors, seizures",
        detection_confidence=0.99,
    ),
    Toxin(
        name="Grapes / raisins",
        aliases=frozenset({"grape", "grapes", "raisin", "raisins", "sultana", "sultanas"}),
        severity="severe",
        species="dogs",
        mechanism="idiosyncratic acute kidney injury",
        symptoms="vomiting, lethargy, reduced urination, loss of appetite",
        detection_confidence=0.98,
    ),
    Toxin(
        name="Onion / garlic (allium)",
        aliases=frozenset({
            "onion", "onions", "garlic", "chive", "chives", "leek", "leeks",
            "shallot", "shallots", "allium",
        }),
        severity="moderate",
        species="dogs and cats",
        mechanism="oxidative damage to red blood cells (hemolytic anemia)",
        symptoms="weakness, pale gums, rapid breathing, reddish urine",
        detection_confidence=0.97,
    ),
    Toxin(
        name="Xylitol",
        aliases=frozenset({"xylitol"}),
        severity="severe",
        species="dogs",
        mechanism="rapid insulin release causing hypoglycemia and liver injury",
        symptoms="weakness, collapse, tremors, seizures, vomiting",
        detection_confidence=0.98,
    ),
    Toxin(
        name="Macadamia nuts",
        aliases=frozenset({"macadamia", "macadamias"}),
        severity="moderate",
        species="dogs",
        mechanism="unknown mechanism causing transient weakness and tremors",
        symptoms="hind-limb weakness, tremors, vomiting, fever",
        detection_confidence=0.95,
    ),
    Toxin(
        name="Caffeine",
        aliases=frozenset({"caffeine", "coffee", "espresso"}),
        severity="severe",
        species="dogs and cats",
        mechanism="methylxanthine stimulant toxicity",
        symptoms="hyperactivity, rapid heart rate, tremors, seizures",
        detection_confidence=0.96,
    ),
    Toxin(
        name="Alcohol",
        aliases=frozenset({"alcohol", "beer", "wine", "liquor", "ethanol"}),
        severity="severe",
        species="dogs and cats",
        mechanism="ethanol central-nervous-system depression",
        symptoms="disorientation, vomiting, difficulty breathing, collapse",
        detection_confidence=0.97,
    ),
    Toxin(
        name="Avocado",
        aliases=frozenset({"avocado", "persin"}),
        severity="mild",
        species="dogs, cats, and especially birds",
        mechanism="persin irritation; the large pit is also a choking/obstruction risk",
        symptoms="mild vomiting or diarrhea; obstruction if the pit is swallowed",
        detection_confidence=0.90,
    ),
]


def screen_for_toxins(query: str) -> List[Dict]:
    """Scan a query for any known pet toxin.

    Returns a list of hazard dictionaries (empty if the query is clear). Each
    hazard records which alias matched so the decision is fully auditable.
    """
    tokens = set(_tokenize(query))
    hits: List[Dict] = []
    for toxin in TOXIN_INDEX:
        matched = sorted(tokens & toxin.aliases)
        if matched:
            hits.append({
                "name": toxin.name,
                "matched_terms": matched,
                "severity": toxin.severity,
                "species": toxin.species,
                "mechanism": toxin.mechanism,
                "symptoms": toxin.symptoms,
                "detection_confidence": toxin.detection_confidence,
            })
    return hits


def _format_hazard_context(hazards: List[Dict]) -> str:
    """Render the retrieved toxicology facts backing a guardrail block."""
    lines = []
    for h in hazards:
        lines.append(
            f"[TOX-{h['name']}] severity={h['severity']}; affects {h['species']}; "
            f"{h['mechanism']}; watch for {h['symptoms']}."
        )
    return " ".join(lines)


def _emergency_warning(hazards: List[Dict]) -> str:
    """Compose the emergency medical warning returned on a guardrail block."""
    names = ", ".join(h["name"] for h in hazards)
    verb = "is" if len(hazards) == 1 else "are"
    symptoms = "; ".join(sorted({h["symptoms"] for h in hazards}))
    return (
        f"⚠️ SAFETY ALERT — {names} {verb} toxic to pets and must NOT be given to "
        f"your animal. If your pet may have already ingested it, treat this as "
        f"an emergency: contact your veterinarian or the {ASPCA_POISON_CONTROL} "
        f"immediately, and do not wait for symptoms to appear. Watch for: "
        f"{symptoms}."
    )


# --------------------------------------------------------------------------- #
# 3. Main execution pipeline  (requirement #3)
# --------------------------------------------------------------------------- #
def _format_advice(doc: KBDocument) -> str:
    """Ground a helpful answer in a retrieved knowledge-base document."""
    return (
        f"{doc.content} (Source: {doc.doc_id} — {doc.title}.) "
        f"This is general guidance; for concerns specific to your pet, consult "
        f"a licensed veterinarian."
    )


def pawpal_agent(query: str) -> Dict:
    """Run the full PawPal+ advice pipeline for a single query.

    Pipeline order (matches the project rubric sequence
    ``Query -> RAG Retrieval -> Guardrail Evaluator -> Response Output``):

      1. RAG retrieval  — retrieve the best-matching knowledge-base document
         for the query.
      2. Guardrail eval — evaluate the query for pet toxins. If a hazard is
         detected the guardrail *overrides* the retrieved advice and returns an
         emergency medical warning (``BLOCKED_BY_GUARDRAIL``); the RAG candidate
         is preserved in ``metadata`` for auditability.
      3. Response output — emit grounded advice, a safe low-confidence fallback,
         or the guardrail's emergency warning.

    Returns a JSON-serialisable dict with (at least) the required keys:
    ``status``, ``confidence_score``, ``retrieved_context`` and ``advice``.
    """
    configure_logging()
    log_event("query_received", query=query)

    # --- Stage 1: RAG retrieval --------------------------------------------- #
    score, doc = retrieve(query, top_k=1)[0]
    score = round(score, 3)
    log_event("rag_retrieved", doc_id=doc.doc_id, title=doc.title, similarity=score)

    # --- Stage 2: Safety & Toxicity Guardrail Evaluation -------------------- #
    hazards = screen_for_toxins(query)
    if hazards:
        # The guardrail overrides the RAG result and fails closed.
        confidence = max(h["detection_confidence"] for h in hazards)
        log_event(
            "guardrail_blocked",
            level=logging.WARNING,
            detected=[h["name"] for h in hazards],
            matched_terms=[t for h in hazards for t in h["matched_terms"]],
            confidence=confidence,
        )
        result = {
            "query": query,
            "status": "BLOCKED_BY_GUARDRAIL",
            "confidence_score": round(confidence, 3),
            "retrieved_context": _format_hazard_context(hazards),
            "advice": _emergency_warning(hazards),
            "metadata": {
                "stage": "safety_guardrail",
                "detected_hazards": [h["name"] for h in hazards],
                "severity": sorted({h["severity"] for h in hazards}),
                # RAG ran first; its candidate is recorded even though the
                # guardrail overrode it, so the trace is fully auditable.
                "rag_candidate": {"doc_id": doc.doc_id, "similarity": score},
            },
        }
        log_event("pipeline_complete", status=result["status"], confidence=result["confidence_score"])
        return result

    log_event("guardrail_passed", similarity=score)

    # --- Stage 3: Response output (guardrail cleared) ----------------------- #
    if score < RETRIEVAL_THRESHOLD:
        log_event("rag_low_confidence", level=logging.WARNING, similarity=score,
                  threshold=RETRIEVAL_THRESHOLD)
        result = {
            "query": query,
            "status": "SUCCESS",
            "confidence_score": score,
            "retrieved_context": "(no knowledge-base document matched above the retrieval threshold)",
            "advice": (
                "I don't have a strong match for that in the Pet-Care Knowledge "
                "Base, so I can't give grounded advice here. For anything "
                f"health-related, please consult a licensed veterinarian or the "
                f"{ASPCA_POISON_CONTROL} in an emergency."
            ),
            "metadata": {"stage": "rag_fallback", "matched_document": doc.doc_id},
        }
    else:
        result = {
            "query": query,
            "status": "SUCCESS",
            "confidence_score": score,
            "retrieved_context": f"[{doc.doc_id}] {doc.title}: {doc.content}",
            "advice": _format_advice(doc),
            "metadata": {"stage": "rag_grounded", "matched_document": doc.doc_id},
        }

    log_event("pipeline_complete", status=result["status"], confidence=result["confidence_score"])
    return result


# --------------------------------------------------------------------------- #
# Guardrail reliability self-check  (backs the README reliability table)
# --------------------------------------------------------------------------- #
#: Labeled evaluation set: (query, expected_blocked). Toxic queries MUST block;
#: safe queries MUST pass. Used to compute guardrail accuracy on demand.
GUARDRAIL_TEST_SET: List[Tuple[str, bool]] = [
    # --- Toxic queries that MUST be blocked --------------------------------- #
    ("Can I give my dog chocolate as a treat?", True),
    ("Are grapes safe for my puppy?", True),
    ("My cat licked some garlic butter, is that ok?", True),
    ("Is xylitol gum dangerous for dogs?", True),
    ("Can dogs eat onions?", True),
    ("Are macadamia nuts toxic to dogs?", True),
    ("Is coffee bad for my cat?", True),
    ("Can my dog have a sip of beer?", True),
    ("Is avocado safe for my parrot?", True),
    # --- Safe queries that MUST pass ---------------------------------------- #
    ("What's a good daily exercise routine for a Labrador?", False),
    ("How often should I groom a long-haired cat?", False),
    ("How much water should my dog drink each day?", False),
    ("When should my puppy get vaccinated?", False),
    ("How can I keep my bored cat mentally stimulated?", False),
]


def run_reliability_check() -> Dict:
    """Evaluate the guardrail against the labeled test set and return metrics."""
    configure_logging()
    total = len(GUARDRAIL_TEST_SET)
    correct = 0
    toxic_total = toxic_blocked = 0
    safe_total = false_positives = 0

    for query, expected_blocked in GUARDRAIL_TEST_SET:
        blocked = bool(screen_for_toxins(query))
        if blocked == expected_blocked:
            correct += 1
        if expected_blocked:
            toxic_total += 1
            toxic_blocked += int(blocked)
        else:
            safe_total += 1
            false_positives += int(blocked)

    metrics = {
        "total_cases": total,
        "overall_accuracy": round(correct / total, 3),
        "toxin_block_rate": round(toxic_blocked / toxic_total, 3),
        "toxins_blocked": f"{toxic_blocked}/{toxic_total}",
        "false_positive_rate": round(false_positives / safe_total, 3),
        "safe_queries_passed": f"{safe_total - false_positives}/{safe_total}",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "toxin_index_size": len(TOXIN_INDEX),
    }
    log_event("reliability_check", **metrics)
    return metrics


# --------------------------------------------------------------------------- #
# 5. Runnable demo  (requirement #5)
# --------------------------------------------------------------------------- #
def _demo(title: str, query: str) -> None:
    print(f"\n{'-' * 70}\n{title}\nquery: {query!r}\n{'-' * 70}")
    print(json.dumps(pawpal_agent(query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    configure_logging()
    print("=" * 70)
    print("  PawPal+ — Applied AI Advice Agent (RAG + Safety Guardrail)")
    print("=" * 70)

    # Required test case 1: a SAFE query -> RAG SUCCESS.
    _demo("TEST CASE 1 — SAFE QUERY (exercise routine)",
          "What's a good daily exercise routine for a Labrador?")

    # Required test case 2: a TOXIC query -> BLOCKED_BY_GUARDRAIL.
    _demo("TEST CASE 2 — TOXIC QUERY (chocolate)",
          "Can I give my dog chocolate as a treat?")

    # Reliability evidence for the README / model card.
    print(f"\n{'-' * 70}\nGUARDRAIL RELIABILITY SELF-CHECK\n{'-' * 70}")
    print(json.dumps(run_reliability_check(), indent=2))
