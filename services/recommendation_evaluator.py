"""Recommendation evaluation loop.

Mentor guidance #6 called for the recommendation-quality check to use
**semantic similarity + domain-expert benchmarks** rather than the
implicit "trust the LLM" pattern that previously governed the
consulting-grade generator.

This module provides an offline, deterministic evaluator that scores
each recommendation across four dimensions and returns per-dimension
scores + a composite score + warning bullets. The evaluator is
intentionally lightweight (token-overlap Jaccard similarity, keyword
heuristics, structural checks) so it stays fast and reproducible; a
future upgrade can swap the Jaccard similarity for embeddings without
changing the public contract.

Dimensions
----------
* **Coverage**       — does the recommendation address the underlying
  gap? (Jaccard overlap between the gap statement and the ``what/why/how``
  narrative.)
* **Specificity**    — is the language area-specific rather than
  generic? (Compared against a small generic-language reference set.)
* **Actionability**  — are owner, timeline, KPIs named?
* **Grounding**      — does the recommendation cite the source
  obligation / regulatory basis?

Each dimension yields a 0..1 score; the composite is a weighted mean
(Coverage 0.35 / Actionability 0.25 / Grounding 0.20 / Specificity 0.20)
and is stored on ``RichRecommendation.metadata["_eval"]`` when
:func:`attach_evaluations` is invoked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------


#: Weights applied to the four dimensions when computing the composite score.
DIMENSION_WEIGHTS: Dict[str, float] = {
    "coverage":      0.35,
    "actionability": 0.25,
    "grounding":     0.20,
    "specificity":   0.20,
}


#: Domain-expert benchmark corpus — a small set of phrases that mark
#: well-formed consulting recommendations. Coverage of these phrases in
#: the ``how`` / ``implementation_steps`` sections contributes to
#: specificity. Kept short on purpose so drift stays cheap to review.
_DOMAIN_BENCHMARK_PHRASES: Sequence[str] = (
    "board approval", "risk appetite", "kpi", "kri", "runbook",
    "stress test", "tabletop", "resilience testing",
    "recovery time objective", "recovery point objective",
    "vendor due diligence", "exit strategy",
    "log retention", "incident classification",
    "policy attestation", "control testing",
    "gap remediation", "governance charter",
    "management body", "second line", "third line",
    "regulatory reporting", "supervisory dialogue",
    "raci", "operating model", "target operating model",
    "compensating control",
)


#: Phrases we treat as **generic filler**. Frequent occurrences pull the
#: specificity score down. Keep patterns lowercase; word boundaries are
#: applied at scoring time.
_GENERIC_LANGUAGE_PHRASES: Sequence[str] = (
    "best practice",
    "industry standard",
    "as appropriate",
    "as needed",
    "consider",
    "review",
    "leverage synergies",
    "leverage best practices",
    "utilize",
    "utilise",
    "consult with stakeholders",
    "align with the strategy",
    "ensure alignment",
    "monitor progress",
)


#: Cheap Jaccard tokeniser. Non-alphanumeric characters split tokens,
#: tokens are lowercased, and English stop-words are dropped so the
#: overlap reflects meaningful content, not filler.
_STOP_WORDS = frozenset({
    "the", "and", "or", "of", "to", "a", "an", "in", "on", "for", "at",
    "by", "with", "from", "is", "are", "be", "will", "shall", "should",
    "must", "may", "can", "as", "that", "this", "which", "we", "our",
    "it", "its", "you", "your", "any", "all",
})


def _tokenise(text: Optional[str]) -> set:
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOP_WORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    if not union:
        return 0.0
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Dataclass returned to callers
# ---------------------------------------------------------------------------


@dataclass
class RecommendationEvaluation:
    """Result of :func:`evaluate_recommendation`."""

    coverage: float
    specificity: float
    actionability: float
    grounding: float
    composite: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage":      round(self.coverage, 3),
            "specificity":   round(self.specificity, 3),
            "actionability": round(self.actionability, 3),
            "grounding":     round(self.grounding, 3),
            "composite":     round(self.composite, 3),
            "warnings":      list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Per-dimension scorers
# ---------------------------------------------------------------------------


def _score_coverage(
    rec_text: str,
    gap_text: str,
    obligation_text: str,
) -> float:
    """How well the recommendation semantics overlap with the gap.

    Uses token Jaccard against both the gap description and the
    underlying obligation text. Whichever is higher wins (a
    recommendation that quotes the obligation directly should still
    count as covering the gap).
    """
    rec_tokens = _tokenise(rec_text)
    gap_tokens = _tokenise(gap_text)
    ob_tokens = _tokenise(obligation_text)
    return max(_jaccard(rec_tokens, gap_tokens), _jaccard(rec_tokens, ob_tokens))


def _score_specificity(rec_text: str) -> float:
    """Reward area-specific consulting phrases; penalise generic filler."""
    text = str(rec_text or "").lower()
    if not text:
        return 0.0
    positive_hits = sum(1 for phrase in _DOMAIN_BENCHMARK_PHRASES if phrase in text)
    negative_hits = sum(1 for phrase in _GENERIC_LANGUAGE_PHRASES if phrase in text)
    positive_score = min(1.0, positive_hits / 3.0)  # 3+ phrases -> full credit
    penalty = min(0.5, negative_hits * 0.1)
    return max(0.0, positive_score - penalty)


def _score_actionability(rec: Any) -> float:
    """Does the recommendation name owner, horizon, KPIs, steps?"""
    checks = []
    checks.append(bool(str(getattr(rec, "suggested_owner", "") or "").strip()))
    checks.append(bool(str(getattr(rec, "horizon", "") or "").strip()))
    steps = list(getattr(rec, "implementation_steps", []) or [])
    metrics = list(getattr(rec, "success_metrics", []) or [])
    checks.append(len(steps) >= 3)
    checks.append(len(metrics) >= 1)
    # Bonus if there's a "how" narrative that names a concrete action.
    how = str(getattr(rec, "how", "") or "").lower()
    checks.append(bool(re.search(r"\b(implement|deploy|establish|publish|rebuild|roll out|codify|charter)\b", how)))
    return sum(1 for c in checks if c) / len(checks)


def _score_grounding(
    rec: Any,
    obligation_ids: Iterable[str],
    regulation: str,
) -> float:
    """Does the recommendation cite an obligation / regulation / clause?"""
    text_bits = " ".join([
        str(getattr(rec, "why", "") or ""),
        str(getattr(rec, "what", "") or ""),
        str(getattr(rec, "how", "") or ""),
        " ".join(str(x) for x in (getattr(rec, "requirement_ids", []) or [])),
    ]).lower()
    if not text_bits:
        return 0.0
    checks = []
    if regulation:
        checks.append(regulation.lower() in text_bits)
    ob_ids = [o for o in obligation_ids if o]
    if ob_ids:
        checks.append(any(o.lower() in text_bits for o in ob_ids))
    # Article / clause reference detected.
    checks.append(bool(re.search(r"\bart(?:icle|\.)\s*\d+", text_bits)))
    # Explicit regulatory basis on the rec.
    checks.append(bool(str(getattr(rec, "regulatory_basis", "") or "").strip()))
    if not checks:
        return 0.0
    return sum(1 for c in checks if c) / len(checks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_recommendation(
    rec: Any,
    *,
    gap_text: str = "",
    obligation_text: str = "",
    obligation_ids: Optional[Sequence[str]] = None,
    regulation: str = "",
) -> RecommendationEvaluation:
    """Score one recommendation across the four evaluation dimensions.

    Returns a :class:`RecommendationEvaluation` with per-dimension
    scores, composite, and warning bullets. See module docstring for
    the weighting scheme.
    """
    rec_text = " ".join([
        str(getattr(rec, "title", "") or ""),
        str(getattr(rec, "what", "") or ""),
        str(getattr(rec, "why", "") or ""),
        str(getattr(rec, "how", "") or ""),
        " ".join(str(s) for s in getattr(rec, "implementation_steps", []) or []),
    ])

    coverage = _score_coverage(rec_text, gap_text, obligation_text)
    specificity = _score_specificity(rec_text)
    actionability = _score_actionability(rec)
    grounding = _score_grounding(rec, obligation_ids or [], regulation)

    composite = (
        coverage      * DIMENSION_WEIGHTS["coverage"]
        + specificity   * DIMENSION_WEIGHTS["specificity"]
        + actionability * DIMENSION_WEIGHTS["actionability"]
        + grounding     * DIMENSION_WEIGHTS["grounding"]
    )

    warnings: List[str] = []
    if coverage < 0.10:
        warnings.append(
            "Coverage < 10% — the recommendation may not address the underlying gap."
        )
    if specificity < 0.20:
        warnings.append(
            "Specificity < 20% — language reads as generic; add area-specific detail."
        )
    if actionability < 0.60:
        warnings.append(
            "Actionability < 60% — name owner, horizon, at least 3 steps, and 1 KPI."
        )
    if grounding < 0.50:
        warnings.append(
            "Grounding < 50% — cite the source obligation / regulation article."
        )

    return RecommendationEvaluation(
        coverage=coverage,
        specificity=specificity,
        actionability=actionability,
        grounding=grounding,
        composite=composite,
        warnings=warnings,
    )


def attach_evaluations(
    recommendations: Iterable[Any],
    *,
    obligations: Optional[Sequence[Any]] = None,
    scoring_evaluation: Optional[Mapping[str, Any]] = None,
    regulation: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every recommendation and attach the report to its metadata.

    Returns a ``{recommendation_id -> eval_dict}`` map for convenience.
    Mutates each recommendation's ``metadata["_eval"]`` in place. The
    call is safe on both :class:`RichRecommendation` and the legacy
    compact :class:`Recommendation` shape as long as either exposes a
    dict-like ``metadata``.
    """
    ob_by_id: Dict[str, Any] = {
        str(getattr(o, "obligation_id", "")): o for o in (obligations or [])
    }
    area_gaps: Dict[str, str] = {}
    if scoring_evaluation:
        for area, summary in (scoring_evaluation.get("area_summary") or {}).items():
            bits: List[str] = []
            for key in ("Gap", "Gaps", "gaps"):
                gap = summary.get(key)
                if isinstance(gap, list):
                    bits.extend(str(x) for x in gap)
                elif gap:
                    bits.append(str(gap))
            if bits:
                area_gaps[str(area).lower()] = " ".join(bits)

    out: Dict[str, Dict[str, Any]] = {}
    for rec in recommendations or []:
        area = str(getattr(rec, "area", "") or "").lower()
        gap_text = area_gaps.get(area, "")

        ob_ids: List[str] = list(getattr(rec, "obligation_ids", []) or [])
        if not ob_ids:
            # Fall back to the legacy ``mapped_requirement_ids`` slot used
            # by :class:`services.recommendation_service.Recommendation`
            # and to any generic ``requirement_ids`` field seen elsewhere.
            for name in (
                "mapped_requirement_ids",
                "requirement_ids",
                "requirement_id",
            ):
                val = getattr(rec, name, None)
                if isinstance(val, (list, tuple)):
                    ob_ids.extend(str(x) for x in val)
                elif val:
                    ob_ids.append(str(val))

        obligation_texts: List[str] = []
        for oid in ob_ids:
            ob = ob_by_id.get(oid)
            if ob is not None:
                obligation_texts.append(str(getattr(ob, "compliance_requirement", "") or ""))
                obligation_texts.append(str(getattr(ob, "title", "") or ""))

        report = evaluate_recommendation(
            rec,
            gap_text=gap_text,
            obligation_text=" ".join(obligation_texts),
            obligation_ids=ob_ids,
            regulation=regulation,
        )
        report_dict = report.to_dict()

        try:
            metadata = getattr(rec, "metadata", None)
            if metadata is None:
                setattr(rec, "metadata", {"_eval": report_dict})
            elif isinstance(metadata, dict):
                metadata["_eval"] = report_dict
        except Exception:  # pragma: no cover - defensive; recs are usually mutable
            pass

        rid = str(getattr(rec, "id", "") or getattr(rec, "area", "") or id(rec))
        out[rid] = report_dict
    return out


__all__ = [
    "DIMENSION_WEIGHTS",
    "RecommendationEvaluation",
    "attach_evaluations",
    "evaluate_recommendation",
]
