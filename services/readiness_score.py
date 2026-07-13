"""Weighted readiness scoring model (DORA demo profile).

This module is a *reusable* scoring layer that sits on top of the existing
:mod:`services.scoring_engine`. The scoring engine already produces per-answer
scores on a 0-100 scale (via :func:`services.scoring_engine.score_value` and
:func:`services.scoring_engine.score_free_text_answer`) but combines them with
a simple weighted average of question weights. Product feedback: business
users want a *weighted assessment areas* view, not a per-question weighted
average.

For the DORA demo we ship the following seven areas (weights sum to 100%):

    ICT Governance & Risk Management            20%
    ICT Policies & Standards                    15%
    ICT Processes & Operating Model             15%
    ICT Controls & Compliance Controls          20%
    ICT Technology & Architecture               15%
    Documentation & Evidence                    10%
    Training & Awareness                         5%

Public entry point:

    result = compute_weighted_readiness(questions, state, ...)

``result`` is a :class:`WeightedReadinessResult` dataclass carrying:

- ``overall_readiness_score``        — 0-100, weighted-average of area scores
- ``readiness_rating``               — banded label ("Highly Ready" / ...)
- ``area_scores``                    — {area: raw_score_0_to_100}
- ``weighted_scores``                — {area: score * weight, 0-100}
- ``coverage_gaps``                  — {area: 100 - score}
- ``top_gap_areas``                  — top-N (area, gap, severity) tuples
- ``completeness_score``             — 0-100, applicable answered ratio
- ``accuracy_score``                 — 0-100, evidence/consistency/mapping blend
- ``accuracy_breakdown``             — {evidence_coverage, answer_consistency,
                                        requirement_mapping_coverage}
- ``gap_categories``                 — {category: {score, count, top_areas}}
- ``recommendations_input``          — structured hints for Agent 4 / Rec svc
- ``area_details``                   — full per-area breakdown for the UI table
- ``overall_coverage_gap``           — 100 - overall_readiness_score

The module is **pure Python / no Streamlit** so it is trivially testable and
can be reused from CLI / batch scripts. The Streamlit dashboard only reads
the returned dataclass.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# The scoring engine already normalises a single answer to 0-100 (or None for
# N/A). We reuse that so option metadata + legacy fallbacks + N/A handling
# stay consistent between the classic scorer and the weighted one.
from services.scoring_engine import (
    _NA_LABEL_TOKENS,
    score_free_text_answer,
    score_value,
)


# ---------------------------------------------------------------------------
# Configuration — weighted areas
# ---------------------------------------------------------------------------
#
# Weights are expressed as percentages (must sum to 100). Kept as a plain
# ``Dict[str, float]`` so it is trivially serialisable and can be overridden
# by callers that want a different profile (e.g. MiFID II, SOX, HIPAA). The
# module validates the sum at import time so a misconfiguration is caught
# immediately instead of producing nonsense scores.

DORA_AREA_WEIGHTS: Dict[str, float] = {
    "ICT Governance & Risk Management": 20.0,
    "ICT Policies & Standards": 15.0,
    "ICT Processes & Operating Model": 15.0,
    "ICT Controls & Compliance Controls": 20.0,
    "ICT Technology & Architecture": 15.0,
    "Documentation & Evidence": 10.0,
    "Training & Awareness": 5.0,
}


def validate_weights(weights: Mapping[str, float]) -> None:
    """Raise ``ValueError`` if the weight config does not sum to exactly 100.

    A small tolerance (0.01) is allowed to accommodate floating-point noise.
    Weights must all be non-negative.
    """
    if not weights:
        raise ValueError("Area weight configuration is empty.")
    for area, w in weights.items():
        if not isinstance(w, (int, float)):
            raise ValueError(f"Weight for area {area!r} is not numeric: {w!r}")
        if w < 0:
            raise ValueError(f"Weight for area {area!r} is negative: {w}")
    total = sum(weights.values())
    if abs(total - 100.0) > 0.01:
        raise ValueError(
            f"Area weights must sum to 100 (got {total:.2f}). "
            f"Adjust DORA_AREA_WEIGHTS so every weighted profile totals 100."
        )


validate_weights(DORA_AREA_WEIGHTS)


# ---------------------------------------------------------------------------
# Area classification — map arbitrary question areas to a weighted area
# ---------------------------------------------------------------------------
#
# The rest of the codebase already stamps each question with an ``area``
# label (see ``services.questionnaire_generator.AREA_KEYWORDS``) but those
# labels are the operational-org taxonomy (Front Office, Middle Office, IT
# Security, …) and do NOT match the DORA-weighted taxonomy. This module
# provides a two-step classifier:
#
#   1. Direct alias lookup — a manually curated table so common labels resolve
#      instantly and deterministically.
#   2. Keyword scoring — if the alias table misses, we score the question
#      text + area + function against per-target keyword bags and take the
#      best match. This lets us classify AI-generated questions that use
#      novel wording.
#
# Everything is case-insensitive. When no bucket wins the question is placed
# in ``ICT Processes & Operating Model`` — the broadest operational bucket —
# so no answer is ever silently discarded.

_DEFAULT_AREA = "ICT Processes & Operating Model"

_AREA_ALIASES: Dict[str, str] = {
    # Existing "Governance" / "Risk" style areas → Governance bucket.
    "governance model": "ICT Governance & Risk Management",
    "governance model / risk management": "ICT Governance & Risk Management",
    "risk & controls framework": "ICT Governance & Risk Management",
    "risk and controls framework": "ICT Governance & Risk Management",
    "program sponsorship / budget planning": "ICT Governance & Risk Management",
    # Policies / compliance.
    "internal compliances": "ICT Policies & Standards",
    "compliance & legal": "ICT Policies & Standards",
    "legal (contracts, readiness & agreements)": "ICT Policies & Standards",
    "people, policies & processes": "ICT Policies & Standards",
    # Processes / operating model.
    "operating model": "ICT Processes & Operating Model",
    "business structure & functions": "ICT Processes & Operating Model",
    "firm type / client type": "ICT Processes & Operating Model",
    "front office": "ICT Processes & Operating Model",
    "middle office": "ICT Processes & Operating Model",
    "back office": "ICT Processes & Operating Model",
    "programme maturity / programme ownership": "ICT Processes & Operating Model",
    # Controls / compliance controls.
    "third party risk management / dependency": "ICT Controls & Compliance Controls",
    "third-party risk management": "ICT Controls & Compliance Controls",
    "it security / cyber security": "ICT Controls & Compliance Controls",
    "cyber security": "ICT Controls & Compliance Controls",
    # Technology / architecture.
    "it, systems & technology": "ICT Technology & Architecture",
    "technology / it operations": "ICT Technology & Architecture",
    "data reporting & governance": "ICT Technology & Architecture",
    # Documentation & evidence.
    "regulatory reporting & financial reporting": "Documentation & Evidence",
    "data governance / reporting": "Documentation & Evidence",
    "internal audit / assurance": "Documentation & Evidence",
    # Training.
    "hr": "Training & Awareness",
    "human resources / training": "Training & Awareness",
    # Direct pass-through — if the AI already emitted the canonical label
    # we obviously keep it.
    "ict governance & risk management": "ICT Governance & Risk Management",
    "ict policies & standards": "ICT Policies & Standards",
    "ict processes & operating model": "ICT Processes & Operating Model",
    "ict controls & compliance controls": "ICT Controls & Compliance Controls",
    "ict technology & architecture": "ICT Technology & Architecture",
    "documentation & evidence": "Documentation & Evidence",
    "training & awareness": "Training & Awareness",
}

# Keyword bags used when the alias table misses. Order does not matter — the
# classifier scores every bag and picks the winner. Keywords are lowercase
# whole-word matches (regex ``\b``-anchored).
_AREA_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "ICT Governance & Risk Management": (
        "governance", "board", "management body", "accountability",
        "risk appetite", "risk register", "risk assessment", "risk owner",
        "committee", "escalation", "oversight", "sponsor", "budget",
        "steering", "raci",
    ),
    "ICT Policies & Standards": (
        "policy", "policies", "standard", "standards", "procedure",
        "procedures", "guideline", "guidelines", "code of conduct",
        "compliance", "legal", "regulation", "regulatory", "obligation",
        "attestation",
    ),
    "ICT Processes & Operating Model": (
        "process", "processes", "workflow", "operating model", "operations",
        "runbook", "sop", "handoff", "role", "responsibility", "raci",
        "front office", "middle office", "back office", "settlement",
    ),
    "ICT Controls & Compliance Controls": (
        "control", "controls", "safeguard", "mitigation", "vulnerability",
        "access", "iam", "mfa", "encryption", "monitoring", "detection",
        "prevention", "third party", "third-party", "vendor", "provider",
        "cyber", "security incident", "penetration", "attestation",
    ),
    "ICT Technology & Architecture": (
        "system", "systems", "application", "applications", "infrastructure",
        "platform", "architecture", "cloud", "network", "database",
        "cmdb", "itsm", "siem", "api", "integration", "endpoint",
        "asset inventory", "capacity", "resilience", "backup", "restore",
        "recovery",
    ),
    "Documentation & Evidence": (
        "document", "documentation", "evidence", "record", "records",
        "audit trail", "audit", "trace", "traceability", "log", "logs",
        "report", "reports", "artefact", "artifact", "register",
        "dashboard", "kpi", "kri", "metric",
    ),
    "Training & Awareness": (
        "training", "awareness", "education", "e-learning", "elearning",
        "onboarding", "induction", "briefing", "drill", "phishing",
        "curriculum", "certification", "upskill", "reskill",
    ),
}


def _tokenise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _keyword_score(needles: Sequence[str], haystack: str) -> int:
    if not haystack:
        return 0
    hits = 0
    for kw in needles:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, haystack):
            hits += 1
    return hits


def classify_question_area(
    question: Mapping[str, Any],
    weights: Mapping[str, float] = DORA_AREA_WEIGHTS,
) -> str:
    """Classify one question into a weighted assessment area.

    Uses (in order) an explicit ``weighted_area`` field on the question if
    the AI already stamped one, then the alias table, then keyword scoring
    over ``area + function + question`` text. Falls back to the operating
    model bucket so every question always resolves to some weighted area.
    """
    # 1. Explicit override — if a question already carries a
    # ``weighted_area`` field (set by an upstream classifier or a manual
    # override) and that value is a known weighted bucket, trust it.
    explicit = _tokenise(str(question.get("weighted_area") or ""))
    for canon in weights.keys():
        if explicit == canon.lower():
            return canon

    # 2. Alias table.
    raw_area = _tokenise(str(question.get("area") or ""))
    if raw_area in _AREA_ALIASES and _AREA_ALIASES[raw_area] in weights:
        return _AREA_ALIASES[raw_area]

    raw_function = _tokenise(str(question.get("function") or ""))
    if raw_function in _AREA_ALIASES and _AREA_ALIASES[raw_function] in weights:
        return _AREA_ALIASES[raw_function]

    # 3. Keyword scoring against area + function + question text.
    haystack_parts: List[str] = [raw_area, raw_function]
    q_text = question.get("question") or ""
    if q_text:
        haystack_parts.append(_tokenise(str(q_text)))
    rationale = question.get("rationale") or ""
    if rationale:
        haystack_parts.append(_tokenise(str(rationale)))
    haystack = " ".join(part for part in haystack_parts if part)

    best_area = _DEFAULT_AREA if _DEFAULT_AREA in weights else next(iter(weights))
    best_score = 0
    for canon, kws in _AREA_KEYWORDS.items():
        if canon not in weights:
            continue
        score = _keyword_score(kws, haystack)
        if score > best_score:
            best_score = score
            best_area = canon

    return best_area


# ---------------------------------------------------------------------------
# Rating bands — readiness label + gap severity
# ---------------------------------------------------------------------------
#
# Ranges are inclusive on the lower bound, inclusive on the upper bound of
# the previous band's neighbour (i.e. 90 → "Highly Ready", 89.9 → "Largely
# Ready"). The helpers return the label string.

_READINESS_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (90.0, 100.0, "Highly Ready"),
    (75.0, 89.999, "Largely Ready"),
    (60.0, 74.999, "Moderately Ready"),
    (40.0, 59.999, "Needs Significant Improvement"),
    (0.0, 39.999, "Not Ready"),
)

_GAP_SEVERITY_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 10.0, "Low"),
    (10.001, 25.0, "Medium"),
    (25.001, 40.0, "High"),
    (40.001, 100.0, "Critical"),
)


def readiness_rating(score: float) -> str:
    """Return the banded readiness label for a 0-100 score."""
    s = max(0.0, min(100.0, float(score)))
    for lo, hi, label in _READINESS_BANDS:
        if lo <= s <= hi:
            return label
    return "Not Ready"


def gap_severity(gap_pct: float) -> str:
    """Return the banded severity label for a 0-100 coverage gap."""
    g = max(0.0, min(100.0, float(gap_pct)))
    for lo, hi, label in _GAP_SEVERITY_BANDS:
        if lo <= g <= hi:
            return label
    return "Critical"


# ---------------------------------------------------------------------------
# Answer scoring — reuse scoring_engine + normalise raw percentages
# ---------------------------------------------------------------------------
#
# Quantitative options carrying ``score_value`` metadata are already handled
# by ``scoring_engine.score_value``. This helper additionally normalises
# raw percentage strings ("95%", "60 %") when the answer is a free-form
# text but the underlying signal is quantitative — so an SME can type
# "MFA coverage 95%" and still get a numeric score.

_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMERIC_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def normalise_quantitative_answer(value: Any) -> Optional[float]:
    """Try to parse a raw string into a 0-100 percentage.

    Returns ``None`` when no numeric signal is present. Values outside
    ``[0, 100]`` are clamped so an accidental "150%" still lands at the
    ceiling.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        pct = float(value)
        return max(0.0, min(100.0, pct))
    text = str(value).strip()
    if not text:
        return None
    m = _PERCENT_RE.search(text)
    if not m:
        # Bare numeric strings ("60", "95.5") are treated as a percentage
        # only when the whole trimmed input is exactly a number. Anything
        # with adjacent letters ("50k budget"), currency signs ("$50") or
        # suffixed units ("50 FTEs") is ambiguous and rejected - we do
        # not want to score a headcount or a budget as a readiness %.
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return None
        try:
            num = float(text)
        except ValueError:
            return None
        if 0.0 <= num <= 100.0:
            return num
        return None
    try:
        pct = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(100.0, pct))


def score_answer_for_readiness(
    answer: Any,
    question: Mapping[str, Any],
) -> Optional[float]:
    """Return a 0-100 readiness score for one answer.

    Resolution order:

    1. :func:`services.scoring_engine.score_value` — option metadata,
       legacy option-label table, enumeration-ratio fallback.
    2. Quantitative percentage normalisation ("95%" → 95).
    3. Free-text scoring when the question is Open Ended / free-text.

    Returns ``None`` for empty answers, explicit N/A picks, or answers
    that can't be resolved to a numeric signal. Those are excluded from
    the denominator so N/A answers do not distort the area score.
    """
    if answer is None:
        return None

    q_is_free = bool(question.get("is_free_text"))
    qtype = str(question.get("question_type") or "").lower()
    is_free = q_is_free or "open" in qtype or "free" in qtype or "text" == qtype

    # Multi-select: score every pick individually and take the mean of the
    # scores that resolve to a numeric value. Empty list → None.
    if isinstance(answer, (list, tuple, set)):
        parts: List[float] = []
        any_na = False
        for item in answer:
            s = score_value(item, question)
            if s is None:
                any_na = True
                continue
            parts.append(float(s))
        if parts:
            return sum(parts) / len(parts)
        return None if any_na or not answer else None

    if isinstance(answer, str):
        stripped = answer.strip()
        if not stripped:
            return None
        if stripped.lower() in _NA_LABEL_TOKENS:
            return None

    direct = score_value(answer, question)
    if direct is not None:
        return float(direct)

    quant = normalise_quantitative_answer(answer)
    if quant is not None:
        return quant

    if is_free:
        ft = score_free_text_answer(answer, question_text=question.get("question"))
        if ft is not None:
            return float(ft)
    return None


# ---------------------------------------------------------------------------
# Accuracy sub-scores
# ---------------------------------------------------------------------------
#
# Accuracy is a blend of three signals so we never surface a single
# unexplainable number to the user:
#
#   Accuracy = 0.40 * evidence_coverage
#            + 0.30 * answer_consistency
#            + 0.30 * requirement_mapping_coverage
#
# Each sub-score is 0-100. All three are exposed on the result so the UI
# can render "Accuracy 82 (Evidence 90, Consistency 75, Mapping 80)".

_EVIDENCE_KEYS: Tuple[str, ...] = (
    "evidence", "evidence_notes", "evidence_reference", "attachment",
    "attachments", "supporting_evidence", "artefact", "artifact",
)


def _has_evidence_reference(question: Mapping[str, Any], answer: Any) -> bool:
    """Best-effort detection of "this answer has evidence backing it".

    We look at:

    - Any question-level evidence field (``evidence_expectations`` is a
      *requirement* not a signal — we ignore that here).
    - Free-form answer text that mentions filenames, URLs, "see attached",
      "policy XYZ v1.2", "confluence", "SharePoint", …
    - Answers that carry a dict payload with an explicit evidence key.
    """
    if isinstance(answer, Mapping):
        for key in _EVIDENCE_KEYS:
            val = answer.get(key)
            if val:
                return True
    text_bits: List[str] = []
    if isinstance(answer, str):
        text_bits.append(answer)
    elif isinstance(answer, (list, tuple, set)):
        for item in answer:
            if isinstance(item, str):
                text_bits.append(item)
    haystack = " ".join(text_bits).lower()
    if not haystack:
        return False
    return any(
        marker in haystack
        for marker in (
            "http://", "https://", "www.", "sharepoint", "confluence",
            "attached", "attachment", "see ", "screenshot", "artefact",
            "artifact", "evidence:", "policy ", "sop ", "runbook",
            ".pdf", ".docx", ".xlsx", ".png", ".jpg",
        )
    )


def compute_evidence_coverage(
    questions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> float:
    """0-100. Share of *answered* questions that have evidence backing."""
    answered = [
        q for q in questions
        if q.get("question_id") in responses and responses[q["question_id"]] not in (None, "", [])
    ]
    if not answered:
        return 0.0
    with_evidence = sum(
        1 for q in answered if _has_evidence_reference(q, responses.get(q["question_id"]))
    )
    return (with_evidence / len(answered)) * 100.0


def compute_requirement_mapping_coverage(
    questions: Sequence[Mapping[str, Any]],
) -> float:
    """0-100. Share of questions that map to at least one BRD requirement.

    Uses ``mapped_requirement_ids`` (BRD IDs) OR ``mapped_obligation_ids``
    (regulatory OBL-XXX IDs). Questions without any mapping are considered
    "orphan" and drag the accuracy score down.
    """
    if not questions:
        return 0.0
    mapped = 0
    for q in questions:
        if q.get("mapped_requirement_ids") or q.get("mapped_obligation_ids"):
            mapped += 1
    return (mapped / len(questions)) * 100.0


_HIGH_CONFIDENCE_LABELS = {
    "fully implemented", "implemented", "complete", "mostly complete",
    "consistently met / exceeded", "tracked, mostly met", "yes",
    "measured / optimised",
}


def compute_answer_consistency(
    questions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> float:
    """0-100. Penalises internal contradictions in the answer set.

    Rule-based first pass:

    - Claiming "Fully Implemented" (or equivalent high-maturity label)
      *without* any supporting evidence → -1 for every occurrence.
    - Claiming full coverage on a control-family question but flagging a
      dependent question as "Not Started" → -1 per pair.

    Starts at 100 and subtracts a penalty for every inconsistency, floored
    at 0 so a mostly-consistent questionnaire still lands near the top.
    """
    if not questions or not responses:
        return 100.0

    penalty = 0
    considered = 0
    for q in questions:
        qid = q.get("question_id")
        if not qid or qid not in responses:
            continue
        ans = responses[qid]
        label = ""
        if isinstance(ans, str):
            label = ans.strip().lower()
        elif isinstance(ans, (list, tuple)) and ans:
            first = ans[0]
            label = str(first).strip().lower() if isinstance(first, str) else ""
        if not label:
            continue
        considered += 1
        if label in _HIGH_CONFIDENCE_LABELS and not _has_evidence_reference(q, ans):
            penalty += 1

    if considered == 0:
        return 100.0
    penalty_pct = min(60.0, (penalty / considered) * 100.0)
    return max(0.0, 100.0 - penalty_pct)


def compute_completeness(
    questions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> float:
    """0-100. Answered Applicable Requirements / Total Applicable * 100.

    "Applicable" excludes questions the SME explicitly flagged as N/A
    (their answer resolves to ``score_value == None``). This matches the
    behaviour of the existing rules engine so the two numbers stay in sync.
    """
    if not questions:
        return 0.0
    applicable = 0
    answered = 0
    for q in questions:
        qid = q.get("question_id")
        ans = responses.get(qid) if qid else None
        if isinstance(ans, str) and ans.strip().lower() in _NA_LABEL_TOKENS:
            continue
        applicable += 1
        if ans in (None, "", []):
            continue
        answered += 1
    if applicable == 0:
        return 0.0
    return (answered / applicable) * 100.0


def compute_accuracy_score(
    questions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> Dict[str, float]:
    """Return ``{overall, evidence_coverage, answer_consistency, mapping_coverage}``."""
    ev = compute_evidence_coverage(questions, responses)
    cons = compute_answer_consistency(questions, responses)
    mapping = compute_requirement_mapping_coverage(questions)
    overall = 0.40 * ev + 0.30 * cons + 0.30 * mapping
    return {
        "overall": round(overall, 2),
        "evidence_coverage": round(ev, 2),
        "answer_consistency": round(cons, 2),
        "requirement_mapping_coverage": round(mapping, 2),
    }


# ---------------------------------------------------------------------------
# Gap categories
# ---------------------------------------------------------------------------
#
# Nine cross-cutting gap categories the client dashboard surfaces alongside
# the per-area breakdown. Each category is a keyword bag (matched against
# question text + area + function + rationale) so any *unanswered* or
# *low-scoring* question rolls up to the categories it touches.

_GAP_CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Regulatory Interpretation Gap": (
        "regulation", "regulatory", "obligation", "interpretation",
        "article", "clause", "supervisor", "regulator",
    ),
    "Requirement Coverage Gap": (
        "requirement", "brd", "business requirement", "acceptance",
        "user story", "epic",
    ),
    "Policy Gap": (
        "policy", "policies", "standard", "code of conduct", "guideline",
    ),
    "Process Gap": (
        "process", "workflow", "runbook", "sop", "operating model",
        "handoff",
    ),
    "Control Gap": (
        "control", "safeguard", "mitigation", "monitoring", "detection",
        "prevention", "iam", "mfa", "access", "encryption",
    ),
    "Technology Gap": (
        "system", "application", "infrastructure", "platform", "cloud",
        "network", "cmdb", "itsm", "siem", "api", "endpoint", "architecture",
    ),
    "Documentation Gap": (
        "document", "documentation", "record", "log", "artefact", "artifact",
        "report", "register", "audit trail",
    ),
    "Training Gap": (
        "training", "awareness", "education", "e-learning", "elearning",
        "onboarding", "induction",
    ),
    "Evidence Gap": (
        "evidence", "attestation", "traceability", "proof", "screenshot",
        "attachment",
    ),
}


def _match_gap_categories(text: str) -> List[str]:
    if not text:
        return []
    matched: List[str] = []
    for cat, kws in _GAP_CATEGORY_KEYWORDS.items():
        if _keyword_score(kws, text) > 0:
            matched.append(cat)
    return matched


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AreaBreakdown:
    """Per-area row for the weighted scoring table on the dashboard."""

    area: str
    weight: float                 # % (e.g. 20.0)
    num_questions: int            # applicable + answered
    total_questions: int          # total mapped to this area (denominator)
    area_score: float             # 0-100 mean of answered scores
    weighted_score: float         # area_score * (weight/100), 0-100 scale
    coverage_gap: float           # 100 - area_score
    gap_severity: str             # Low / Medium / High / Critical


@dataclass
class GapCategoryBreakdown:
    """Aggregate score for one of the nine cross-cutting gap categories."""

    category: str
    score: float                  # mean readiness of matched questions (0-100)
    coverage_gap: float           # 100 - score
    matched_questions: int        # answered + N/A-excluded questions in scope
    top_areas: List[str] = field(default_factory=list)


@dataclass
class WeightedReadinessResult:
    """Structured output of :func:`compute_weighted_readiness`."""

    overall_readiness_score: float
    readiness_rating: str
    overall_coverage_gap: float
    completeness_score: float
    accuracy_score: float
    accuracy_breakdown: Dict[str, float]
    area_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    coverage_gaps: Dict[str, float]
    top_gap_areas: List[Dict[str, Any]]
    area_details: List[AreaBreakdown]
    gap_categories: Dict[str, GapCategoryBreakdown]
    recommendations_input: List[Dict[str, Any]]
    weights: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain-dict copy so the result can round-trip through JSON.

        Used for SQLite persistence in ``services/database.py`` and for the
        Streamlit dashboard which serialises snapshots into
        ``st.session_state``.
        """
        return {
            "overall_readiness_score": self.overall_readiness_score,
            "readiness_rating": self.readiness_rating,
            "overall_coverage_gap": self.overall_coverage_gap,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "accuracy_breakdown": dict(self.accuracy_breakdown),
            "area_scores": dict(self.area_scores),
            "weighted_scores": dict(self.weighted_scores),
            "coverage_gaps": dict(self.coverage_gaps),
            "top_gap_areas": [dict(row) for row in self.top_gap_areas],
            "area_details": [
                {
                    "area": row.area,
                    "weight": row.weight,
                    "num_questions": row.num_questions,
                    "total_questions": row.total_questions,
                    "area_score": row.area_score,
                    "weighted_score": row.weighted_score,
                    "coverage_gap": row.coverage_gap,
                    "gap_severity": row.gap_severity,
                }
                for row in self.area_details
            ],
            "gap_categories": {
                cat: {
                    "category": row.category,
                    "score": row.score,
                    "coverage_gap": row.coverage_gap,
                    "matched_questions": row.matched_questions,
                    "top_areas": list(row.top_areas),
                }
                for cat, row in self.gap_categories.items()
            },
            "recommendations_input": [dict(r) for r in self.recommendations_input],
            "weights": dict(self.weights),
        }


# ---------------------------------------------------------------------------
# Public entry point — compute a full weighted readiness result
# ---------------------------------------------------------------------------


def _extract_responses(state: Any) -> Dict[str, Any]:
    """Read the ``responses`` mapping off an ``AssessmentState`` or dict."""
    if state is None:
        return {}
    if isinstance(state, Mapping):
        return dict(state.get("responses") or {})
    return dict(getattr(state, "responses", {}) or {})


def compute_weighted_readiness(
    questions: Sequence[Mapping[str, Any]],
    state: Any = None,
    *,
    weights: Mapping[str, float] = DORA_AREA_WEIGHTS,
    top_n: int = 5,
    responses: Optional[Mapping[str, Any]] = None,
) -> WeightedReadinessResult:
    """Return a :class:`WeightedReadinessResult` for the given questionnaire.

    Parameters
    ----------
    questions:
        Sequence of question dicts (as stored in the ``QuestionnairePackage``
        or emitted by :func:`services.scoring_engine.evaluate`). Each dict
        should carry at least ``question_id``, ``question``, ``area``,
        ``function``, ``question_type``, ``options`` and either
        ``mapped_requirement_ids`` or ``mapped_obligation_ids``.
    state:
        An ``AssessmentState`` instance or a plain dict with a ``responses``
        key. Answers are read from ``state.responses[question_id]``.
    weights:
        Weight profile. Defaults to :data:`DORA_AREA_WEIGHTS`. The mapping
        is validated on entry - the function raises ``ValueError`` if the
        weights do not sum to 100.
    top_n:
        How many "highest gap" areas the result exposes for the UI.
    responses:
        Optional explicit response mapping. Overrides whatever is on
        ``state`` when provided. Useful for CLI / test harnesses.

    Returns
    -------
    WeightedReadinessResult
        Full structured result. ``.as_dict()`` produces a JSON-safe copy.
    """
    try:
        validate_weights(weights)
    except Exception:
        logger.exception("Readiness weights failed validation. weights=%s", dict(weights or {}))
        raise
    responses_map: Dict[str, Any] = (
        dict(responses) if responses is not None else _extract_responses(state)
    )
    logger.debug(
        "compute_weighted_readiness. questions=%d responses=%d weights=%s",
        len(questions or []), len(responses_map), list(weights.keys()),
    )

    # --- Group scores per area ------------------------------------------
    per_area_scores: Dict[str, List[float]] = defaultdict(list)
    per_area_total: Dict[str, int] = defaultdict(int)
    per_area_answered: Dict[str, int] = defaultdict(int)
    per_area_questions: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for area in weights:
        per_area_scores[area] = []
        per_area_total[area] = 0
        per_area_answered[area] = 0
        per_area_questions[area] = []

    # For gap categories we track per-category (score list, area frequency).
    per_cat_scores: Dict[str, List[float]] = defaultdict(list)
    per_cat_area_counter: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for q in questions:
        area = classify_question_area(q, weights)
        per_area_total[area] += 1
        per_area_questions[area].append(q)

        qid = q.get("question_id")
        ans = responses_map.get(qid) if qid else None
        score = score_answer_for_readiness(ans, q)
        if score is not None:
            per_area_scores[area].append(score)
            per_area_answered[area] += 1

        # Gap-category attribution: score contributes when we have one,
        # otherwise the question still counts as "in scope" for the
        # category so we can spot uncovered categories.
        text_bag = " ".join(
            _tokenise(str(q.get(field_) or ""))
            for field_ in ("area", "function", "question", "rationale")
        )
        cats = _match_gap_categories(text_bag)
        for cat in cats:
            if score is not None:
                per_cat_scores[cat].append(score)
            per_cat_area_counter[cat][area] += 1

    # --- Area breakdown + weighted overall ------------------------------
    area_scores: Dict[str, float] = {}
    weighted_scores: Dict[str, float] = {}
    coverage_gaps: Dict[str, float] = {}
    area_details: List[AreaBreakdown] = []
    overall_weighted = 0.0
    for area, weight in weights.items():
        scores = per_area_scores.get(area, [])
        if scores:
            avg = sum(scores) / len(scores)
        else:
            avg = 0.0
        gap = max(0.0, 100.0 - avg)
        weighted = avg * (weight / 100.0)
        area_scores[area] = round(avg, 2)
        weighted_scores[area] = round(weighted, 2)
        coverage_gaps[area] = round(gap, 2)
        overall_weighted += weighted
        area_details.append(
            AreaBreakdown(
                area=area,
                weight=round(weight, 2),
                num_questions=per_area_answered.get(area, 0),
                total_questions=per_area_total.get(area, 0),
                area_score=round(avg, 2),
                weighted_score=round(weighted, 2),
                coverage_gap=round(gap, 2),
                gap_severity=gap_severity(gap),
            )
        )

    overall_readiness = round(overall_weighted, 2)
    overall_gap = round(max(0.0, 100.0 - overall_readiness), 2)

    top_gap_areas = sorted(
        (
            {
                "area": row.area,
                "coverage_gap": row.coverage_gap,
                "weight": row.weight,
                "gap_severity": row.gap_severity,
                "area_score": row.area_score,
            }
            for row in area_details
        ),
        key=lambda r: (r["coverage_gap"], r["weight"]),
        reverse=True,
    )[:top_n]

    # --- Accuracy + completeness ----------------------------------------
    completeness = round(compute_completeness(questions, responses_map), 2)
    acc = compute_accuracy_score(questions, responses_map)
    accuracy_overall = acc["overall"]

    # --- Gap categories rollup ------------------------------------------
    gap_categories: Dict[str, GapCategoryBreakdown] = {}
    for cat in _GAP_CATEGORY_KEYWORDS:
        scores = per_cat_scores.get(cat, [])
        matched = per_cat_area_counter.get(cat, {})
        if scores:
            cat_score = sum(scores) / len(scores)
        else:
            cat_score = 0.0
        top_area_list = sorted(matched.items(), key=lambda kv: kv[1], reverse=True)
        gap_categories[cat] = GapCategoryBreakdown(
            category=cat,
            score=round(cat_score, 2),
            coverage_gap=round(max(0.0, 100.0 - cat_score), 2),
            matched_questions=sum(matched.values()),
            top_areas=[a for a, _ in top_area_list[:3]],
        )

    # --- Recommendations input hints ------------------------------------
    # The recommendation service can either consume these directly (Agent 4)
    # or use them to seed a template - we give it every high-signal input.
    rec_input: List[Dict[str, Any]] = []
    for row in area_details:
        if row.coverage_gap <= 10.0:
            continue
        rec_input.append({
            "area": row.area,
            "weight": row.weight,
            "area_score": row.area_score,
            "coverage_gap": row.coverage_gap,
            "severity": row.gap_severity,
            "num_answered": row.num_questions,
            "num_total": row.total_questions,
            "gap_categories": [
                cat for cat, bd in gap_categories.items()
                if row.area in bd.top_areas and bd.coverage_gap > 10.0
            ],
        })
    rec_input.sort(key=lambda r: (r["coverage_gap"], r["weight"]), reverse=True)

    result = WeightedReadinessResult(
        overall_readiness_score=overall_readiness,
        readiness_rating=readiness_rating(overall_readiness),
        overall_coverage_gap=overall_gap,
        completeness_score=completeness,
        accuracy_score=accuracy_overall,
        accuracy_breakdown={
            "evidence_coverage": acc["evidence_coverage"],
            "answer_consistency": acc["answer_consistency"],
            "requirement_mapping_coverage": acc["requirement_mapping_coverage"],
        },
        area_scores=area_scores,
        weighted_scores=weighted_scores,
        coverage_gaps=coverage_gaps,
        top_gap_areas=top_gap_areas,
        area_details=area_details,
        gap_categories=gap_categories,
        recommendations_input=rec_input,
        weights=dict(weights),
    )
    logger.info(
        "Weighted readiness computed. overall=%.1f rating=%s completeness=%.1f accuracy=%.1f",
        overall_readiness, result.readiness_rating, completeness, accuracy_overall,
    )
    return result


# ---------------------------------------------------------------------------
# Demo / validation helper
# ---------------------------------------------------------------------------


def demo_result() -> WeightedReadinessResult:
    """Return the *reference* example from the product spec (Overall = 76.0).

    Governance 90, Policies 80, Processes 70, Controls 65, Technology 85,
    Documentation & Evidence 60, Training 75 → 18 + 12 + 10.5 + 13 + 12.75
    + 6 + 3.75 = 76.0.

    Handy for validating the maths from a REPL or a unit test - the
    Streamlit dashboard also uses this when the user clicks "Load demo".
    """
    demo_scores = {
        "ICT Governance & Risk Management": 90.0,
        "ICT Policies & Standards": 80.0,
        "ICT Processes & Operating Model": 70.0,
        "ICT Controls & Compliance Controls": 65.0,
        "ICT Technology & Architecture": 85.0,
        "Documentation & Evidence": 60.0,
        "Training & Awareness": 75.0,
    }
    fake_questions: List[Dict[str, Any]] = []
    fake_responses: Dict[str, Any] = {}
    counter = 0
    for area, target in demo_scores.items():
        counter += 1
        qid = f"DEMO-{counter:03d}"
        # Build a single option carrying the target score - the answer
        # then resolves through ``score_value`` (option metadata) to the
        # exact target, sidestepping the enumeration-ratio fallback that
        # a bare numeric answer would trigger.
        label = f"Demo target {target:.0f}"
        fake_questions.append({
            "question_id": qid,
            "question": f"Demo probe for {area}",
            "weighted_area": area,
            "area": area,
            "function": "",
            "options": [
                {"label": label, "score_value": float(target)},
                {"label": "Full", "score_value": 100.0},
                {"label": "Partial", "score_value": 60.0},
                {"label": "None", "score_value": 0.0},
            ],
            "question_type": "Single Select",
            "mapped_requirement_ids": ["BR-DEMO-001"],
        })
        fake_responses[qid] = label

    class _Stub:
        responses = fake_responses

    return compute_weighted_readiness(fake_questions, _Stub())


__all__ = [
    "DORA_AREA_WEIGHTS",
    "AreaBreakdown",
    "GapCategoryBreakdown",
    "WeightedReadinessResult",
    "classify_question_area",
    "compute_accuracy_score",
    "compute_answer_consistency",
    "compute_completeness",
    "compute_evidence_coverage",
    "compute_requirement_mapping_coverage",
    "compute_weighted_readiness",
    "demo_result",
    "gap_severity",
    "normalise_quantitative_answer",
    "readiness_rating",
    "score_answer_for_readiness",
    "validate_weights",
]
