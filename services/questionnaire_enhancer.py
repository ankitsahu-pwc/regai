"""Questionnaire post-processing (impact weighting + ordering).

Historical role
---------------
This module used to inject a small set of hardcoded quantitative question
templates (percentage-implementation, business-unit count, monitoring
frequency, ...) into every questionnaire and bump the scoring weight of
existing questions based on impact severity.

Current role
------------
The AI questionnaire agent
(:mod:`services.ai_questionnaire_generator`) now generates ALL question
content — including quantitative options with per-option scoring — from
the live regulatory / BRD / RTM context. There are no hardcoded question
or option templates left in this module.

What remains here is a lightweight, side-effect-free enhancer that:

1. Computes an ``impact_severity`` for every area referenced by the
   package (from the AI-generated ImpactAssessment when available, else
   from readiness inversion of the scored area summary).
2. Bumps each question's ``scoring_weight`` / ``impact_weight`` /
   ``impact_severity`` / ``priority_rank`` so high-impact areas
   contribute proportionally more to the readiness score.
3. Re-orders the ``questions`` list so the highest-impact questions
   appear first in the UI.

The enhancer never adds, removes or rewrites questions. It only decorates
severity / weight metadata and reorders the list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from models.workflow_models import ImpactAssessment

from .severity import (
    band_rank,
    from_label,
    weight_from_band,
)


def _build_impact_severity_index(
    impact: Optional[ImpactAssessment],
) -> Dict[str, str]:
    """Return a ``{lowercase_item -> severity}`` index built from ``impact``.

    Called once per :func:`enhance_questionnaire_package` invocation so
    the per-area lookup in :func:`_area_severity` is O(1) for exact
    matches instead of O(dimensions × items). The substring fallback
    still runs when the exact match misses.
    """
    if impact is None:
        return {}
    index: Dict[str, str] = {}
    for dim in impact.dimensions():
        severity = dim.severity or "Medium"
        for item in dim.items or []:
            key = str(item).lower().strip()
            if key and key not in index:
                index[key] = severity
    return index


def _area_severity(
    area: str,
    impact: Optional[ImpactAssessment],
    area_summary: Mapping[str, Mapping[str, Any]],
    *,
    impact_index: Optional[Dict[str, str]] = None,
) -> str:
    """Return 'Critical' / 'High' / 'Medium' / 'Low' for the area.

    Prefers the AI-generated impact assessment; falls back to readiness
    inversion when the assessment doesn't cover the area.

    ``impact_index`` is an optional pre-built
    ``{lowercase_item -> severity}`` map (see
    :func:`_build_impact_severity_index`). When provided, the exact-match
    lookup is O(1). When ``None`` we fall back to iterating
    ``impact.dimensions()`` to preserve the historical call signature.
    """
    area_lower = area.lower()

    if impact_index:
        # O(1) exact-match hit for the common case where an impact-dimension
        # item exactly matches the questionnaire area label.
        hit = impact_index.get(area_lower)
        if hit:
            return hit
        # Substring fallback: an impact item may be a fragment of the area
        # label or vice versa (e.g. "Third Party Risk" vs
        # "Third-Party Risk Management").
        for key, severity in impact_index.items():
            if key in area_lower or area_lower in key:
                return severity

    elif impact is not None:
        for dim in impact.dimensions():
            for item in dim.items or []:
                item_lower = str(item).lower()
                if area_lower == item_lower or item_lower in area_lower:
                    return dim.severity or "Medium"

    summary = area_summary.get(area) or {}
    try:
        pct = float(summary.get("Compliance %") or summary.get("compliance_pct") or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    if pct == 0.0:
        return "Medium"
    impact_pct = 100.0 - pct
    if impact_pct >= 75:
        return "Critical"
    if impact_pct >= 50:
        return "High"
    if impact_pct >= 25:
        return "Medium"
    return "Low"


def _weight_from_severity(severity: str) -> int:
    """Map severity label -> numeric weight (1..5).

    Thin wrapper over :func:`services.severity.weight_from_band`. Accepts
    either the readiness ladder (``Critical / At risk / Watch / Ready``)
    or the impact ladder (``Critical / High / Medium / Low``).
    """
    return weight_from_band(from_label(severity))


def _severity_rank(severity: str) -> int:
    """Numeric urgency rank for sorting (higher = more urgent).

    Wraps :func:`services.severity.band_rank`, but preserves the historical
    quirk that ``Low`` (impact-ladder) sorts one step ahead of ``Ready``
    (readiness-ladder). Both bands collapse to :class:`SeverityBand.READY`
    in the shared model, so we split them here to avoid changing the
    ordering of area lists on the dashboard.
    """
    sev = (severity or "").strip().lower()
    if sev == "low":
        return 1
    return band_rank(from_label(severity))


def _unique_areas(package: Mapping[str, Any]) -> List[str]:
    seen: List[str] = []
    unique: set = set()
    for q in package.get("questions") or []:
        area = str(q.get("area") or "").strip()
        if area and area not in unique:
            unique.add(area)
            seen.append(area)
    return seen


def enhance_questionnaire_package(
    package: Dict[str, Any],
    *,
    impact: Optional[ImpactAssessment] = None,
    scoring_evaluation: Optional[Mapping[str, Any]] = None,
    regulation: str = "DORA",
) -> Dict[str, Any]:
    """Decorate questions with impact severity + weights and reorder.

    Steps (no content is added or removed):

    1. Compute the severity per area (AI impact assessment first,
       readiness inversion fallback).
    2. For each question set ``impact_severity`` / ``impact_weight`` /
       ``impact_level`` / ``priority_rank`` and bump ``scoring_weight``
       to the impact-derived value (never lower — an AI-set weight wins
       when it is higher).
    3. Sort questions by priority: highest severity first, then closed
       before free-text, then by scoring weight, then by area.

    The input dict is mutated in place and also returned.
    """
    questions: List[Dict[str, Any]] = list(package.get("questions") or [])
    area_summary = (scoring_evaluation or {}).get("area_summary") or {}
    areas = _unique_areas(package)

    # Build the inverted impact-severity index ONCE, then reuse it for every
    # area / question in the package. Prior implementation walked
    # ``impact.dimensions()`` on every area — O(A x D x I) — which added up
    # on large regulations with many impacted areas and dimensions.
    impact_index = _build_impact_severity_index(impact)
    severity_by_area: Dict[str, str] = {
        area: _area_severity(area, impact, area_summary, impact_index=impact_index)
        for area in areas
    }

    for q in questions:
        area = str(q.get("area") or "").strip()
        if not area:
            continue
        severity = severity_by_area.get(area, "Medium")
        target_weight = _weight_from_severity(severity)
        # Preserve any per-question weight the AI generator has already set
        # when it is higher than the severity-derived floor.
        try:
            current = int(q.get("scoring_weight") or 1)
        except (TypeError, ValueError):
            current = 1
        if target_weight > current:
            q["scoring_weight"] = target_weight
        # Never overwrite an AI-provided impact_level; only fill it in when missing.
        if not q.get("impact_level"):
            q["impact_level"] = severity
        q["impact_severity"] = severity
        # impact_weight is always resynced to the severity-derived weight so
        # the composite scoring stays consistent even if the AI generator
        # emitted a different weight up front.
        q["impact_weight"] = target_weight
        q["priority_rank"] = _severity_rank(severity)

    def _sort_key(q: Mapping[str, Any]):
        return (
            -int(q.get("priority_rank", 2)),
            1 if q.get("is_free_text") else 0,
            -int(q.get("scoring_weight", 1)),
            str(q.get("area", "")),
            str(q.get("question_id", "")),
        )

    questions.sort(key=_sort_key)
    package["questions"] = questions

    meta = dict(package.get("metadata") or {})
    meta["impact_enhanced"] = True
    meta["area_severity_map"] = severity_by_area
    # ``quantitative_questions_added`` kept in the metadata for backward
    # compatibility with dashboards that read it — always 0 now because the
    # enhancer no longer injects any templates.
    meta["quantitative_questions_added"] = 0
    package["metadata"] = meta

    return package


def prioritize_areas_by_impact(
    area_summary: Mapping[str, Mapping[str, Any]],
    *,
    impact: Optional[ImpactAssessment] = None,
) -> List[str]:
    """Return a list of impacted areas sorted by descending impact severity."""
    impact_index = _build_impact_severity_index(impact)
    order: List[Any] = []
    for area, summary in area_summary.items():
        severity = _area_severity(
            area, impact, area_summary, impact_index=impact_index,
        )
        readiness = float(summary.get("Compliance %") or 0.0)
        order.append((-_severity_rank(severity), readiness, area))
    order.sort()
    return [item[2] for item in order]


__all__ = [
    "enhance_questionnaire_package",
    "prioritize_areas_by_impact",
]
