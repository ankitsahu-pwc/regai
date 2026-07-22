"""Weighted regulatory Impact scoring model (DORA demo profile).

Impact answers a different question from Readiness:

    Readiness = "How prepared is the organisation to comply?"
    Impact    = "How strongly does this regulation affect the
                 organisation, business area, process, system,
                 control and operating model?"

They are calculated **separately** and combined into a Priority Score
so the dashboard can surface *high-impact / low-readiness* areas.

Weighted Impact Formula
------------------------
    Overall Impact = Sigma(Factor Score * Factor Weight)

DORA impact factors (weights must sum to 100):

    1. Regulatory Obligation Criticality  25%
    2. Business Capability Impact         20%
    3. Process Change Impact              15%
    4. Technology / System Impact         15%
    5. Control & Compliance Impact        10%
    6. Data / Reporting Impact            10%
    7. Third-Party / Vendor Impact         5%

Public entry point:

    result = compute_weighted_impact(
        analysis=..., brd_artifact=..., rtm_artifact=...,
        questionnaire=..., readiness_result=..., area_readiness=...,
    )

``result`` is a :class:`WeightedImpactResult` dataclass carrying:

- ``overall_impact_score``           - 0-100, weighted-average of factor scores
- ``impact_rating``                  - banded label ("High Impact" / "Very High Impact" / ...)
- ``factor_scores``                  - {factor: raw_score_0_to_100}
- ``weighted_scores``                - {factor: score * weight, 0-100}
- ``factor_details``                 - per-factor breakdown for the UI table
- ``top_impacted_business_capabilities`` - list of (area, hit_count)
- ``top_impacted_processes``         - list of (process_requirement_summary, priority)
- ``top_impacted_systems``           - list of (system_signal, hit_count)
- ``top_impacted_controls``          - list of (control_expectation, hit_count)
- ``top_impacted_third_parties``     - list of (third_party_signal, hit_count)
- ``heatmap_rows``                   - per (area, function) impact + readiness + priority
- ``priority_areas``                 - sorted list of {area, impact, readiness, priority}
- ``overall_priority_score``         - Impact * (100 - Readiness) / 100
- ``recommendations_input``          - structured hints for Agent 4 / Rec svc

The module is **pure Python / no Streamlit** so it is trivially testable and
can be reused from CLI / batch scripts.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
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


# ---------------------------------------------------------------------------
# Configuration - weighted impact factors
# ---------------------------------------------------------------------------
#
# Weights are expressed as percentages (must sum to 100). Kept as a plain
# ``Dict[str, float]`` so it is trivially serialisable and can be overridden
# by callers that want a different profile.

DORA_IMPACT_FACTOR_WEIGHTS: Dict[str, float] = {
    "Regulatory Obligation Criticality": 25.0,
    "Business Capability Impact": 20.0,
    "Process Change Impact": 15.0,
    "Technology / System Impact": 15.0,
    "Control & Compliance Impact": 10.0,
    "Data / Reporting Impact": 10.0,
    "Third-Party / Vendor Impact": 5.0,
}


def validate_weights(weights: Mapping[str, float]) -> None:
    """Raise ``ValueError`` if the weight config does not sum to exactly 100."""
    if not weights:
        raise ValueError("Impact factor weight configuration is empty.")
    for factor, w in weights.items():
        if not isinstance(w, (int, float)):
            raise ValueError(f"Weight for {factor!r} is not numeric: {w!r}")
        if w < 0:
            raise ValueError(f"Weight for {factor!r} is negative: {w}")
    total = sum(weights.values())
    if abs(total - 100.0) > 0.01:
        raise ValueError(
            f"Impact factor weights must sum to 100 (got {total:.2f})."
        )


validate_weights(DORA_IMPACT_FACTOR_WEIGHTS)


# ---------------------------------------------------------------------------
# Rating + priority bands
# ---------------------------------------------------------------------------

_IMPACT_RATING_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (90.0, 100.0, "Very High Impact"),
    (75.0, 89.999, "High Impact"),
    (60.0, 74.999, "Medium Impact"),
    (40.0, 59.999, "Low Impact"),
    (0.0, 39.999, "Minimal Impact"),
)


def impact_rating(score: float) -> str:
    """Return the banded impact rating for a 0-100 score."""
    s = max(0.0, min(100.0, float(score)))
    for lo, hi, label in _IMPACT_RATING_BANDS:
        if lo <= s <= hi:
            return label
    return "Minimal Impact"


def priority_score(impact: float, readiness: float) -> float:
    """Priority = Impact * (100 - Readiness) / 100 (clamped to 0-100)."""
    imp = max(0.0, min(100.0, float(impact)))
    rdy = max(0.0, min(100.0, float(readiness)))
    return round((imp * (100.0 - rdy)) / 100.0, 2)


# ---------------------------------------------------------------------------
# Factor scoring - rubric-based
# ---------------------------------------------------------------------------
#
# Each factor exposes ``_score_<factor>()`` that returns a 0-100 score plus
# a short rationale string and a list of the concrete signals it counted.
# The rubrics track the product spec bands so the numbers explain
# themselves in the UI.


# --- Priority weights for MoSCoW / verb -------------------------------
# Used to weight per-obligation criticality.

_PRIORITY_WEIGHTS: Dict[str, float] = {
    "must": 1.0,
    "shall": 1.0,
    "should": 0.7,
    "could": 0.4,
    "may": 0.3,
    "can": 0.3,
    "won't": 0.1,
    "wont": 0.1,
}


_VERB_WEIGHTS: Dict[str, float] = {
    "must": 1.0,
    "shall": 1.0,
    "should": 0.7,
    "may": 0.35,
    "can": 0.3,
    "": 0.5,
}


def _obligation_criticality(obl: Mapping[str, Any]) -> float:
    """Return a 0-100 criticality score for a single obligation.

    Blends MoSCoW priority (Must/Should/Could/Won't) with the verbalised
    obligation verb (Must/Shall/Should/May) so an obligation phrased in
    hard-mandatory language (``"Firms shall ..."``) with a MoSCoW rank of
    ``Must`` maxes the score, while an informational ``May`` note with a
    ``Could`` rank lands at the bottom of the band.
    """
    prio = str(obl.get("priority") or "").strip().lower()
    verb = str(obl.get("obligation_verb") or "").strip().lower()
    prio_w = _PRIORITY_WEIGHTS.get(prio, 0.6)
    verb_w = _VERB_WEIGHTS.get(verb, 0.5)
    blend = (prio_w * 0.65) + (verb_w * 0.35)
    return round(max(0.0, min(1.0, blend)) * 100.0, 2)


def _count_saturation(count: float, half_life: float = 3.5) -> float:
    """Log-saturation of a signal count onto 0-100.

    ``half_life`` controls how quickly the score approaches 100 as the
    count grows. With ``half_life=3.5`` the mapping is:

        count = 0   -> 0
        count = 1   -> 24.9
        count = 3   -> 57.3
        count = 5   -> 76.2
        count = 10  -> 94.4
        count = 20+ -> 99.7+

    Deterministic and monotonic - suitable for the "how much material is
    affected" style factors (process / tech / data / control / third-party).
    """
    if count <= 0:
        return 0.0
    return round(min(100.0, 100.0 * (1.0 - math.exp(-count / half_life))), 2)


# --- Factor 1 : Regulatory Obligation Criticality ---------------------

def _score_regulatory_criticality(
    obligations: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    if not obligations:
        return 20.0, "No obligations extracted yet - informational baseline.", []
    scores: List[float] = [_obligation_criticality(o) for o in obligations]
    mean = sum(scores) / len(scores)
    # Signals: keep the top 5 highest-criticality obligations so the UI
    # can show a concrete "why this scored so high" list.
    ranked = sorted(
        (
            (_obligation_criticality(o), str(o.get("obligation_id") or ""), str(o.get("title") or ""))
            for o in obligations
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    top_signals = [
        f"{oid or '-'}: {title[:70]} (crit {score:.0f})"
        for score, oid, title in ranked[:5] if title
    ]
    verdict = (
        f"Averaged {len(obligations)} obligations weighted by MoSCoW "
        f"priority + obligation verb (`shall / must / should / may`)."
    )
    return round(mean, 2), verdict, top_signals


# --- Factor 2 : Business Capability Impact ----------------------------

def _score_business_capability(
    obligations: Sequence[Mapping[str, Any]],
    impact_pairs: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    # Count *canonical* area buckets so ``"Risk & Controls framework"``
    # and ``"Risk & Controls framework / Cyber Security"`` land in the
    # same top-level capability instead of double-counting. Bucket key
    # is case-insensitive; display label is the first-seen casing so
    # the UI reads naturally.
    counts: Counter[str] = Counter()
    display: Dict[str, str] = {}
    for o in obligations:
        raw = o.get("impacted_area")
        key = _area_bucket_key(raw)
        if key:
            counts[key] += 1
            display.setdefault(key, canonicalise_area(raw))
    for p in impact_pairs:
        raw = p.get("area")
        key = _area_bucket_key(raw)
        if key:
            counts[key] += 1
            display.setdefault(key, canonicalise_area(raw))
    areas: Counter[str] = Counter(
        {display.get(k, k): v for k, v in counts.items()}
    )

    distinct = len(areas)
    # Spec bands.
    if distinct >= 5:
        score = min(100.0, 90.0 + (distinct - 5) * 2.0)
    elif distinct >= 3:
        score = 70.0 + (distinct - 3) * 8.0    # 3 -> 70, 4 -> 78
    elif distinct >= 1:
        score = 40.0 + (distinct - 1) * 12.5   # 1 -> 40, 2 -> 52.5
    else:
        score = 15.0

    top_signals = [f"{a} ({n} hits)" for a, n in areas.most_common(5)]
    verdict = (
        f"{distinct} distinct business capabilities touched across obligations "
        f"+ impact pairs (spec: 5+ = 90+, 3-4 = 70-89, 1-2 = 40-69, 0 < 40)."
    )
    return round(score, 2), verdict, top_signals


# --- Factor 3 : Process Change Impact ---------------------------------

def _priority_weight(priority: str) -> float:
    return _PRIORITY_WEIGHTS.get(str(priority or "").strip().lower(), 0.5)


def _score_process_impact(
    requirements: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    process_reqs = [
        r for r in requirements
        if _norm(r.get("source_section")) and (
            "process" in _norm(r.get("source_section"))
            or _norm(r.get("normalized_id")).startswith("br-pro")
        )
    ]
    # Priority-weighted count so a Must-priority process rewrite lifts the
    # score more than a Won't-priority "Nice to have" bullet.
    weighted = sum(_priority_weight(r.get("priority")) for r in process_reqs)
    score = _band_from_weighted_count(weighted)

    top_signals = _top_requirement_signals(process_reqs, limit=5)
    verdict = (
        f"{len(process_reqs)} process requirements weighted by MoSCoW "
        f"priority (weighted count = {weighted:.1f})."
    )
    return score, verdict, top_signals


# --- Factor 4 : Technology / System Impact ----------------------------

_TECH_TERMS: Tuple[str, ...] = (
    "system", "systems", "application", "applications", "infrastructure",
    "platform", "cloud", "api", "integration", "database", "cmdb", "itsm",
    "siem", "monitoring", "encryption", "endpoint", "architecture",
    "network", "workflow tool", "software", "portal", "middleware",
)


def _score_technology_impact(
    requirements: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
    rtm_entries: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    tech_reqs: List[Mapping[str, Any]] = []
    for r in requirements:
        sect = _norm(r.get("source_section"))
        rid = _norm(r.get("normalized_id"))
        text = _norm(_req_text(r))
        if (
            "functional" in sect
            or "non-functional" in sect
            or rid.startswith("nfr")
            or rid.startswith("fr")
            or _has_any(text, _TECH_TERMS)
        ):
            tech_reqs.append(r)

    tech_obligations = [
        o for o in obligations
        if _has_any(_norm(o.get("theme")) + " " + _norm(o.get("compliance_requirement")), _TECH_TERMS)
    ]

    tech_rtm = [
        r for r in rtm_entries
        if _has_any(_norm(r.get("system_process_impact")), _TECH_TERMS)
    ]

    weighted = (
        sum(_priority_weight(r.get("priority")) for r in tech_reqs)
        + 0.7 * len(tech_obligations)
        + 0.5 * len(tech_rtm)
    )
    score = _band_from_weighted_count(weighted)

    top_signals = _top_requirement_signals(tech_reqs, limit=5)
    verdict = (
        f"{len(tech_reqs)} tech-related requirements + {len(tech_obligations)} "
        f"tech-themed obligations + {len(tech_rtm)} RTM tech rows "
        f"(weighted count = {weighted:.1f})."
    )
    return score, verdict, top_signals


# --- Factor 5 : Control & Compliance Impact ---------------------------

def _score_control_impact(
    obligations: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    control_items: Counter[str] = Counter()
    for o in obligations:
        for ce in o.get("control_expectations") or []:
            key = str(ce).strip()
            if key:
                control_items[key] += 1

    ctrl_questions = [
        q for q in questions
        if str(q.get("targets_impact_dimension") or "").strip().lower() == "controls"
        or "control" in _norm(q.get("question"))
    ]

    weighted = len(control_items) + 0.4 * len(ctrl_questions)
    score = _band_from_weighted_count(weighted)

    top_signals = [f"{name} ({n} obligations)" for name, n in control_items.most_common(5)]
    verdict = (
        f"{len(control_items)} distinct control expectations across obligations "
        f"+ {len(ctrl_questions)} control-targeted questions "
        f"(weighted count = {weighted:.1f})."
    )
    return score, verdict, top_signals


# --- Factor 6 : Data / Reporting Impact -------------------------------

_DATA_TERMS: Tuple[str, ...] = (
    "data", "reporting", "report", "metric", "kpi", "kri", "lineage",
    "dictionary", "quality", "dashboard", "register", "log", "audit trail",
    "traceability", "regulatory reporting", "submission",
)


def _score_data_reporting_impact(
    requirements: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    data_reqs: List[Mapping[str, Any]] = []
    for r in requirements:
        sect = _norm(r.get("source_section"))
        rid = _norm(r.get("normalized_id"))
        text = _norm(_req_text(r))
        if (
            "data" in sect or "reporting" in sect
            or rid.startswith("br-dat") or rid.startswith("br-rep")
            or _has_any(text, _DATA_TERMS)
        ):
            data_reqs.append(r)

    data_obligations = [
        o for o in obligations
        if _has_any(
            _norm(o.get("theme")) + " " + _norm(o.get("compliance_requirement")),
            _DATA_TERMS,
        )
    ]

    weighted = (
        sum(_priority_weight(r.get("priority")) for r in data_reqs)
        + 0.6 * len(data_obligations)
    )
    score = _band_from_weighted_count(weighted)

    top_signals = _top_requirement_signals(data_reqs, limit=5)
    verdict = (
        f"{len(data_reqs)} data / reporting requirements + "
        f"{len(data_obligations)} data-themed obligations "
        f"(weighted count = {weighted:.1f})."
    )
    return score, verdict, top_signals


# --- Factor 7 : Third-Party / Vendor Impact ---------------------------

_THIRD_PARTY_TERMS: Tuple[str, ...] = (
    "third party", "third-party", "vendor", "provider", "supplier",
    "outsourc", "sub-contractor", "subcontractor", "supply chain",
    "ict provider", "cloud provider", "exit plan", "critical provider",
)


def _score_third_party_impact(
    obligations: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
) -> Tuple[float, str, List[str]]:
    third_party_obligations: List[Mapping[str, Any]] = []
    for o in obligations:
        blob = " ".join(
            _norm(o.get(field_))
            for field_ in ("theme", "compliance_requirement", "impacted_area", "impacted_function")
        )
        if _has_any(blob, _THIRD_PARTY_TERMS):
            third_party_obligations.append(o)

    third_party_reqs: List[Mapping[str, Any]] = []
    for r in requirements:
        text = _norm(_req_text(r))
        if _has_any(text, _THIRD_PARTY_TERMS):
            third_party_reqs.append(r)

    weighted = (
        len(third_party_obligations)
        + 0.5 * sum(_priority_weight(r.get("priority")) for r in third_party_reqs)
    )
    score = _band_from_weighted_count(weighted, half_life=2.5)

    top_signals: List[str] = []
    for o in third_party_obligations[:5]:
        oid = str(o.get("obligation_id") or "-")
        title = str(o.get("title") or "").strip()
        top_signals.append(f"{oid}: {title[:70]}")
    verdict = (
        f"{len(third_party_obligations)} third-party themed obligations + "
        f"{len(third_party_reqs)} third-party requirement mentions "
        f"(weighted count = {weighted:.1f})."
    )
    return score, verdict, top_signals


# --- Small helpers used by all factors --------------------------------

def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


# ---------------------------------------------------------------------------
# Impacted-area canonicalisation
# ---------------------------------------------------------------------------
#
# The BRD classifier (LLM) emits ``impacted_area`` on each obligation as a
# free-form string. It is often inconsistent — the same underlying business
# capability sometimes comes back as::
#
#     "Risk & Controls framework"
#     "Risk & Controls framework / Cyber Security"
#     "Risk & Controls Framework / Third-Party"
#     "Risk and Controls framework"      # ampersand vs "and"
#
# In the downstream dashboard those distinct strings used to become
# **independent** peer cards, which created the confusing scenario of the
# same-looking parent showing "Critical / 100% impact" while the compound
# child showed "Ready / 0% impact" (real user bug report, 2026-07).
#
# ``canonicalise_area`` collapses every compound label onto its top-level
# parent segment so the impact model, readiness rollup and dashboard cards
# all share one canonical bucket per business capability. The sub-
# classification is preserved separately by callers that want to expose
# a drill-down (see :func:`sub_classifications_for`).
#
# Rules:
#     * Anything after the first ``/`` (or the fancy Unicode variants
#       ``|`` / ``›`` / ``›`` / ``>>`` / ``:``) is a sub-classification
#       and is dropped from the canonical bucket.
#     * The remaining head is normalised: extra whitespace collapsed,
#       leading/trailing separators trimmed, ``and`` / ``&`` unified so
#       ``"risk and controls framework"`` and ``"risk & controls
#       framework"`` land in the same bucket.
#     * Case is preserved for the *first* seen spelling of a bucket so
#       the dashboard label reads naturally; comparison is
#       case-insensitive under the hood.

_AREA_SPLIT_RE = re.compile(r"\s*(?:/|\||>|>>|›|:)\s*", flags=re.UNICODE)
_AREA_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def area_bucket_key(name: Any) -> str:
    """Return the case-insensitive bucket key for ``name``.

    Lowercase + whitespace-collapsed + ``and``/``&`` unified so
    ``"Risk & Controls Framework"``, ``"risk and controls framework"``
    and ``"Risk & Controls framework / Cyber Security"`` all collapse
    to the same bucket. Used only for equality / dict keying — the
    display form is preserved separately by :func:`canonicalise_area`.

    Callers outside :mod:`services.impact_score` that need to aggregate
    LLM-emitted area labels (e.g. :mod:`services.scoring_engine`) should
    use this helper for the counter key and record the first-seen
    :func:`canonicalise_area` output as the human-readable display.
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    head = _AREA_SPLIT_RE.split(raw, maxsplit=1)[0]
    head = _AREA_WS_RE.sub(" ", head).strip().lower()
    # Unify " and " ↔ " & " on both sides of the boundary. Do it after
    # lowercase so ``"AND"`` in ALL CAPS also folds in.
    head = head.replace(" and ", " & ")
    return head


# Deprecated private alias kept so any pre-refactor call sites in tests
# / notebooks stay wired. Do not use for new code.
_area_bucket_key = area_bucket_key


def canonicalise_area(name: Any) -> str:
    """Return the canonical top-level area label for ``name``, preserving
    the original casing of the first segment.

    Empty / non-string inputs yield an empty string. Callers should treat
    that as "no area assigned" and skip counting.

    Note: two inputs that only differ in case (``"Risk & Controls
    Framework"`` vs ``"Risk & Controls framework"``) return two
    superficially-different strings here, but callers keying by
    :func:`_area_bucket_key` will collapse them to a single bucket. When
    you need bucket-equality use :func:`area_matches` instead of ``==``.
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    head = _AREA_SPLIT_RE.split(raw, maxsplit=1)[0].strip()
    head = _AREA_WS_RE.sub(" ", head)
    return head


def area_matches(name: Any, other: Any) -> bool:
    """Case-insensitive equality after canonicalisation + ampersand-unification.

    Used when we need to look up an area name inside a dict whose keys came
    from a *different* code path than the counter we are populating (e.g.
    ``area_readiness`` was built with the raw label but we now key by
    canonical head).
    """
    a = _area_bucket_key(name)
    b = _area_bucket_key(other)
    return bool(a) and a == b


def sub_classifications_for(names: Iterable[Any]) -> Dict[str, List[str]]:
    """Return ``{canonical_head: [sub_classification, ...]}`` for callers
    that want to render a tooltip / expander of the raw child labels that
    were folded into each parent bucket.

    Duplicate sub-classifications collapse to one entry; the order
    preserves first-seen occurrence which reads well in the UI.
    """
    out: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for raw in names:
        raw_str = str(raw or "").strip()
        if not raw_str:
            continue
        head = canonicalise_area(raw_str)
        if not head:
            continue
        # Everything after the first separator, if any.
        parts = _AREA_SPLIT_RE.split(raw_str, maxsplit=1)
        if len(parts) > 1:
            child = _AREA_WS_RE.sub(" ", parts[1].strip())
            if child:
                bucket = out.setdefault(head, [])
                seen_set = seen.setdefault(head, set())
                if child.lower() not in seen_set:
                    bucket.append(child)
                    seen_set.add(child.lower())
    return out


def _req_text(r: Mapping[str, Any]) -> str:
    return " ".join(
        str(r.get(field_) or "")
        for field_ in ("requirement", "detail", "acceptance", "themes")
    )


def _has_any(haystack: str, needles: Sequence[str]) -> bool:
    if not haystack:
        return False
    return any(needle in haystack for needle in needles)


def _band_from_weighted_count(count: float, half_life: float = 3.5) -> float:
    """Map a weighted signal count to the spec bands via log-saturation.

    Rough band alignment for ``half_life=3.5``:

        count 0    -> 0-19  (No change)
        count 1    -> 20-49 (Minor)
        count 3-4  -> 50-74 (Moderate)
        count 6-8  -> 75-89 (Major)
        count 12+  -> 90-100 (New process / full rewrite)
    """
    return _count_saturation(count, half_life=half_life)


def _top_requirement_signals(
    requirements: Sequence[Mapping[str, Any]],
    limit: int = 5,
) -> List[str]:
    """Format the top N requirements by priority weight for the UI."""
    ranked = sorted(
        requirements,
        key=lambda r: _priority_weight(r.get("priority")),
        reverse=True,
    )
    out: List[str] = []
    for r in ranked[:limit]:
        rid = str(r.get("normalized_id") or r.get("source_id") or "-").upper()
        text = str(r.get("requirement") or r.get("detail") or "").strip()
        prio = str(r.get("priority") or "").title()
        out.append(f"{rid} ({prio}): {text[:80]}")
    return out


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImpactFactorBreakdown:
    """One row of the weighted impact table on the dashboard."""

    factor: str
    weight: float
    factor_score: float
    weighted_score: float
    rating: str
    rationale: str
    signals: List[str] = field(default_factory=list)


@dataclass
class PriorityAreaBreakdown:
    """One row of the "High-Impact / Low-Readiness" priority table."""

    area: str
    function: str = ""
    impact_score: float = 0.0
    readiness_score: float = 0.0
    priority_score: float = 0.0
    signal_count: int = 0


@dataclass
class WeightedImpactResult:
    """Structured output of :func:`compute_weighted_impact`."""

    overall_impact_score: float
    impact_rating: str
    factor_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    weights: Dict[str, float]
    factor_details: List[ImpactFactorBreakdown]

    # Top impacted lists.
    top_impacted_business_capabilities: List[Dict[str, Any]]
    top_impacted_processes: List[str]
    top_impacted_systems: List[str]
    top_impacted_controls: List[str]
    top_impacted_third_parties: List[str]

    # Heatmap + priority.
    heatmap_rows: List[Dict[str, Any]]
    priority_areas: List[PriorityAreaBreakdown]
    overall_priority_score: float

    # Recommendation seeds.
    recommendations_input: List[Dict[str, Any]]

    # Sub-classifications that were folded into each canonical area
    # bucket during roll-up (see ``canonicalise_area``). Keyed by the
    # canonical head, e.g. ``{"Risk & Controls framework": ["Cyber
    # Security", "Third-Party"]}``. Empty when the LLM never emitted
    # any ``Parent / Child`` labels. The dashboard exposes this as a
    # tooltip / expander on each rolled-up impact card. Defaulted so
    # older persisted results (loaded before this field existed)
    # deserialise cleanly.
    area_sub_classifications: Dict[str, List[str]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe copy for persistence + session-state round-trip."""
        return {
            "overall_impact_score": self.overall_impact_score,
            "impact_rating": self.impact_rating,
            "factor_scores": dict(self.factor_scores),
            "weighted_scores": dict(self.weighted_scores),
            "weights": dict(self.weights),
            "factor_details": [
                {
                    "factor": row.factor,
                    "weight": row.weight,
                    "factor_score": row.factor_score,
                    "weighted_score": row.weighted_score,
                    "rating": row.rating,
                    "rationale": row.rationale,
                    "signals": list(row.signals),
                }
                for row in self.factor_details
            ],
            "top_impacted_business_capabilities":
                [dict(row) for row in self.top_impacted_business_capabilities],
            "top_impacted_processes": list(self.top_impacted_processes),
            "top_impacted_systems": list(self.top_impacted_systems),
            "top_impacted_controls": list(self.top_impacted_controls),
            "top_impacted_third_parties": list(self.top_impacted_third_parties),
            "heatmap_rows": [dict(row) for row in self.heatmap_rows],
            "priority_areas": [
                {
                    "area": row.area,
                    "function": row.function,
                    "impact_score": row.impact_score,
                    "readiness_score": row.readiness_score,
                    "priority_score": row.priority_score,
                    "signal_count": row.signal_count,
                }
                for row in self.priority_areas
            ],
            "overall_priority_score": self.overall_priority_score,
            "recommendations_input": [dict(r) for r in self.recommendations_input],
            "area_sub_classifications": {
                k: list(v) for k, v in (self.area_sub_classifications or {}).items()
            },
        }


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------
#
# The public entry point accepts a mix of dataclass / Pydantic / dict inputs
# so the caller can hand in whatever shape session state currently holds
# without needing to convert first. Every extractor is defensive - missing
# fields degrade gracefully to empty lists / zeroed scores.


def _to_dict_list(items: Any) -> List[Mapping[str, Any]]:
    if not items:
        return []
    out: List[Mapping[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            out.append(item)
            continue
        try:
            from dataclasses import asdict, is_dataclass
            if is_dataclass(item):
                out.append(asdict(item))
                continue
        except Exception:
            pass
        try:
            out.append(dict(getattr(item, "__dict__", {})))
        except Exception:
            continue
    return out


def _extract_obligations(analysis: Any) -> List[Mapping[str, Any]]:
    if analysis is None:
        return []
    if isinstance(analysis, Mapping):
        return _to_dict_list(analysis.get("obligations") or [])
    return _to_dict_list(getattr(analysis, "obligations", []) or [])


def _extract_requirements_from_report(brd_artifact: Any) -> List[Mapping[str, Any]]:
    """Flatten every requirement item across the BRD report into dicts."""
    if brd_artifact is None:
        return []
    report = None
    if isinstance(brd_artifact, Mapping):
        report = brd_artifact.get("report")
    else:
        report = getattr(brd_artifact, "report", None)
    if report is None:
        return []

    sections = (
        "process_business_requirements",
        "data_business_requirements",
        "reporting_business_requirements",
        "functional_requirements",
        "non_functional_requirements",
    )
    out: List[Mapping[str, Any]] = []
    for section_name in sections:
        section = None
        if isinstance(report, Mapping):
            section = report.get(section_name)
        else:
            section = getattr(report, section_name, None)
        if section is None:
            continue
        items = None
        if isinstance(section, Mapping):
            items = section.get("items")
        else:
            items = getattr(section, "items", None)
        for item in items or []:
            row: Dict[str, Any] = {}
            if isinstance(item, Mapping):
                row.update(item)
            else:
                for f in ("id", "category", "requirement", "detailed_requirement",
                          "regulation_alignment", "priority", "acceptance_criteria",
                          "confidence_level"):
                    row[f] = getattr(item, f, None)
            row["source_section"] = section_name
            row["normalized_id"] = str(row.get("id") or "")
            row["detail"] = row.get("detailed_requirement") or row.get("requirement") or ""
            row["alignment"] = row.get("regulation_alignment") or ""
            row["acceptance"] = row.get("acceptance_criteria") or ""
            out.append(row)
    return out


def _extract_requirements_from_package(package: Any) -> List[Mapping[str, Any]]:
    """Pull the flattened requirements list from a questionnaire package."""
    if package is None:
        return []
    pkg: Mapping[str, Any]
    if isinstance(package, Mapping):
        pkg = package
    else:
        raw = getattr(package, "package", None)
        pkg = raw if isinstance(raw, Mapping) else {}
    return _to_dict_list(pkg.get("requirements") or [])


def _extract_impact_pairs(package: Any) -> List[Mapping[str, Any]]:
    if package is None:
        return []
    pkg: Mapping[str, Any]
    if isinstance(package, Mapping):
        pkg = package
    else:
        raw = getattr(package, "package", None)
        pkg = raw if isinstance(raw, Mapping) else {}
    return _to_dict_list(pkg.get("impact_pairs") or [])


def _extract_questions(package: Any) -> List[Mapping[str, Any]]:
    if package is None:
        return []
    pkg: Mapping[str, Any]
    if isinstance(package, Mapping):
        pkg = package
    else:
        raw = getattr(package, "package", None)
        pkg = raw if isinstance(raw, Mapping) else {}
    return _to_dict_list(pkg.get("questions") or [])


def _extract_rtm_entries(rtm_artifact: Any) -> List[Mapping[str, Any]]:
    if rtm_artifact is None:
        return []
    if isinstance(rtm_artifact, Mapping):
        return _to_dict_list(rtm_artifact.get("entries") or [])
    return _to_dict_list(getattr(rtm_artifact, "entries", []) or [])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_weighted_impact(
    *,
    analysis: Any = None,
    brd_artifact: Any = None,
    rtm_artifact: Any = None,
    questionnaire: Any = None,
    readiness_result: Any = None,
    area_readiness: Optional[Mapping[str, float]] = None,
    pair_readiness: Optional[Mapping[Tuple[str, str], float]] = None,
    weights: Mapping[str, float] = DORA_IMPACT_FACTOR_WEIGHTS,
    top_n: int = 5,
    factor_overrides: Optional[Mapping[str, float]] = None,
) -> WeightedImpactResult:
    """Return a :class:`WeightedImpactResult` for the given inputs.

    Parameters
    ----------
    analysis:
        ``RegulatoryAnalysis`` (dataclass or dict). Provides obligations.
    brd_artifact:
        ``BRDArtifact`` (dataclass or dict). Provides BRD requirement items.
    rtm_artifact:
        ``RTMArtifact`` (dataclass or dict). Provides RTM trace rows.
    questionnaire:
        ``QuestionnairePackage`` (or its ``.package`` dict). Provides
        impact pairs + questions.
    readiness_result:
        Optional :class:`services.readiness_score.WeightedReadinessResult`.
        Used only to compute Priority = Impact x (100 - Readiness) / 100
        at the overall level - individual area priorities can be supplied
        via ``area_readiness``/``pair_readiness`` for higher fidelity.
    area_readiness:
        Optional ``{area: readiness_pct}``. When provided, per-area
        priority is calculated against these numbers.
    pair_readiness:
        Optional ``{(area, function): readiness_pct}``. Used for the
        Area x Function heatmap in the priority section.
    weights:
        Factor weight profile. Defaults to :data:`DORA_IMPACT_FACTOR_WEIGHTS`.
    factor_overrides:
        Optional ``{factor: score}`` map that overrides individual factor
        scores after they are computed. Useful for demo / test data or
        for manual overrides in the UI - the weighted overall is still
        recomputed from the (possibly-overridden) factor scores.
    """
    validate_weights(weights)

    obligations = _extract_obligations(analysis)
    requirements = _extract_requirements_from_report(brd_artifact)
    if not requirements:
        requirements = _extract_requirements_from_package(questionnaire)
    rtm_entries = _extract_rtm_entries(rtm_artifact)
    impact_pairs = _extract_impact_pairs(questionnaire)
    questions = _extract_questions(questionnaire)

    # --- Factor computations ------------------------------------------
    reg_score, reg_verdict, reg_signals = _score_regulatory_criticality(obligations)
    cap_score, cap_verdict, cap_signals = _score_business_capability(obligations, impact_pairs)
    proc_score, proc_verdict, proc_signals = _score_process_impact(requirements)
    tech_score, tech_verdict, tech_signals = _score_technology_impact(
        requirements, obligations, rtm_entries,
    )
    ctrl_score, ctrl_verdict, ctrl_signals = _score_control_impact(obligations, questions)
    data_score, data_verdict, data_signals = _score_data_reporting_impact(requirements, obligations)
    tp_score, tp_verdict, tp_signals = _score_third_party_impact(obligations, requirements)

    factor_scores: Dict[str, float] = {
        "Regulatory Obligation Criticality": reg_score,
        "Business Capability Impact": cap_score,
        "Process Change Impact": proc_score,
        "Technology / System Impact": tech_score,
        "Control & Compliance Impact": ctrl_score,
        "Data / Reporting Impact": data_score,
        "Third-Party / Vendor Impact": tp_score,
    }

    # Manual overrides (used by demo / test harness).
    if factor_overrides:
        for k, v in factor_overrides.items():
            if k in factor_scores:
                factor_scores[k] = round(float(v), 2)

    # --- Weighted overall ----------------------------------------------
    weighted_scores: Dict[str, float] = {}
    overall = 0.0
    for factor, weight in weights.items():
        score = factor_scores.get(factor, 0.0)
        weighted = score * (weight / 100.0)
        weighted_scores[factor] = round(weighted, 2)
        overall += weighted
    overall = round(overall, 2)

    # --- Factor details -----------------------------------------------
    rationales = {
        "Regulatory Obligation Criticality": (reg_verdict, reg_signals),
        "Business Capability Impact": (cap_verdict, cap_signals),
        "Process Change Impact": (proc_verdict, proc_signals),
        "Technology / System Impact": (tech_verdict, tech_signals),
        "Control & Compliance Impact": (ctrl_verdict, ctrl_signals),
        "Data / Reporting Impact": (data_verdict, data_signals),
        "Third-Party / Vendor Impact": (tp_verdict, tp_signals),
    }
    factor_details: List[ImpactFactorBreakdown] = []
    for factor, weight in weights.items():
        s = factor_scores.get(factor, 0.0)
        verdict, signals = rationales.get(factor, ("", []))
        factor_details.append(
            ImpactFactorBreakdown(
                factor=factor,
                weight=weight,
                factor_score=round(s, 2),
                weighted_score=round(s * (weight / 100.0), 2),
                rating=impact_rating(s),
                rationale=verdict,
                signals=list(signals),
            )
        )

    # --- Top impacted lists -------------------------------------------
    top_areas_counter: Counter[str] = Counter()
    for o in obligations:
        a = str(o.get("impacted_area") or "").strip()
        if a:
            top_areas_counter[a] += 2  # obligations weigh more than pairs
    for p in impact_pairs:
        a = str(p.get("area") or "").strip()
        if a:
            top_areas_counter[a] += 1
    top_business_caps = [
        {"area": a, "hit_count": n}
        for a, n in top_areas_counter.most_common(top_n)
    ]

    top_processes = proc_signals[:top_n]
    top_systems = tech_signals[:top_n]

    top_controls_counter: Counter[str] = Counter()
    for o in obligations:
        for ce in o.get("control_expectations") or []:
            key = str(ce).strip()
            if key:
                top_controls_counter[key] += 1
    top_controls = [
        f"{name} ({n} obligations)"
        for name, n in top_controls_counter.most_common(top_n)
    ]
    if not top_controls:
        top_controls = ctrl_signals[:top_n]

    top_third_parties = tp_signals[:top_n]

    # --- Priority / heatmap -------------------------------------------
    # Per-area impact: log-saturation of obligation + pair count so an area
    # touched by many obligations lands above one with a single mention.
    # Areas are counted by their *canonical* head (see
    # ``canonicalise_area``) so an LLM-emitted ``"Risk & Controls
    # framework / Cyber Security"`` folds into the ``"Risk & Controls
    # framework"`` bucket rather than producing a second peer card that
    # contradicts the parent's status.
    # Case-insensitive bucketing with a first-seen display registry so
    # the UI label reads consistently (see ``_area_bucket_key`` /
    # ``canonicalise_area``).
    _bucket_counts: Counter[str] = Counter()
    _bucket_display: Dict[str, str] = {}
    _bucket_raws: Dict[str, List[str]] = {}
    for o in obligations:
        raw = str(o.get("impacted_area") or "").strip()
        key = _area_bucket_key(raw)
        if key:
            _bucket_counts[key] += 1
            _bucket_display.setdefault(key, canonicalise_area(raw))
            if raw:
                _bucket_raws.setdefault(key, []).append(raw)
    for p in impact_pairs:
        raw = str(p.get("area") or "").strip()
        key = _area_bucket_key(raw)
        if key:
            _bucket_counts[key] += 1
            _bucket_display.setdefault(key, canonicalise_area(raw))
            if raw:
                _bucket_raws.setdefault(key, []).append(raw)

    # Re-key into the display form so downstream lookups (which the
    # dashboard hits by display name) work directly.
    area_signal_counter: Counter[str] = Counter()
    area_raw_labels: Dict[str, List[str]] = {}
    for key, cnt in _bucket_counts.items():
        disp = _bucket_display.get(key, key)
        area_signal_counter[disp] = cnt
        if _bucket_raws.get(key):
            area_raw_labels[disp] = list(_bucket_raws[key])

    def _area_impact(count: int) -> float:
        # Slightly steeper than the factor saturation so a "hot" area
        # gets a headline-worthy score.
        return _count_saturation(count, half_life=2.5)

    # Bridge the readiness dict to the (case-insensitive) canonical
    # buckets we just built. When the readiness dict was assembled
    # upstream from the raw ``impacted_area`` labels we now roll every
    # raw key into its bucket key, then look up the first-seen display
    # form for the output. Values are averaged (unweighted here — the
    # readiness dict has already been signal-weighted upstream in
    # ``scoring_engine.evaluate_state``).
    area_readiness_raw: Dict[str, float] = {
        str(k).strip(): float(v)
        for k, v in (area_readiness or {}).items()
    }
    _rdy_sum: Dict[str, float] = {}
    _rdy_n: Dict[str, int] = {}
    for raw_key, value in area_readiness_raw.items():
        bkey = _area_bucket_key(raw_key)
        if not bkey:
            continue
        _rdy_sum[bkey] = _rdy_sum.get(bkey, 0.0) + value
        _rdy_n[bkey] = _rdy_n.get(bkey, 0) + 1
    area_readiness_map: Dict[str, float] = {}
    for bkey, total in _rdy_sum.items():
        disp = _bucket_display.get(bkey, bkey)
        area_readiness_map[disp] = total / max(1, _rdy_n[bkey])

    priority_areas: List[PriorityAreaBreakdown] = []
    for area, count in area_signal_counter.most_common():
        area_imp = _area_impact(count)
        area_rdy = area_readiness_map.get(area, 0.0)
        prio = priority_score(area_imp, area_rdy)
        priority_areas.append(
            PriorityAreaBreakdown(
                area=area,
                impact_score=area_imp,
                readiness_score=round(area_rdy, 2),
                priority_score=prio,
                signal_count=count,
            )
        )
    priority_areas.sort(key=lambda r: r.priority_score, reverse=True)

    # Area x Function heatmap - based on impact_pairs. Canonicalise the
    # area side to match the priority-area rollup so drill-downs from a
    # card to its heatmap cells work with the same bucket names. Keyed
    # by the display label so it lines up with ``priority_areas``.
    pair_readiness_map: Dict[Tuple[str, str], float] = {}
    for k, v in (pair_readiness or {}).items():
        bkey = _area_bucket_key(k[0])
        if not bkey:
            continue
        disp = _bucket_display.get(bkey, canonicalise_area(k[0]))
        pair_readiness_map[(disp, str(k[1]).strip())] = float(v)

    heatmap_rows: List[Dict[str, Any]] = []
    pair_signal_counter: Counter[Tuple[str, str]] = Counter()
    for p in impact_pairs:
        bkey = _area_bucket_key(p.get("area"))
        f = str(p.get("function") or "").strip()
        if not bkey or not f:
            continue
        disp = _bucket_display.get(bkey, canonicalise_area(p.get("area")))
        # Weight by number of mapped requirements so hot cells surface first.
        reqs = p.get("requirement_ids") or []
        pair_signal_counter[(disp, f)] += max(1, len(reqs))
    for (area, func), count in pair_signal_counter.most_common(50):
        pair_imp = _count_saturation(count, half_life=3.0)
        pair_rdy = pair_readiness_map.get((area, func), 0.0)
        heatmap_rows.append({
            "area": area,
            "function": func,
            "impact_score": pair_imp,
            "readiness_score": round(pair_rdy, 2),
            "priority_score": priority_score(pair_imp, pair_rdy),
            "signal_count": count,
        })

    # Overall priority (single number for the KPI card).
    overall_readiness = 0.0
    if readiness_result is not None:
        overall_readiness = float(getattr(readiness_result, "overall_readiness_score", 0.0) or 0.0)
    overall_priority = priority_score(overall, overall_readiness)

    # --- Recommendation seeds -----------------------------------------
    rec_input: List[Dict[str, Any]] = []
    for row in factor_details:
        if row.factor_score < 40.0:
            continue
        rec_input.append({
            "factor": row.factor,
            "score": row.factor_score,
            "weighted_score": row.weighted_score,
            "rating": row.rating,
            "signals": list(row.signals[:3]),
        })
    rec_input.sort(key=lambda r: r["weighted_score"], reverse=True)

    # Deduplicate the raw labels we captured while rolling up each
    # canonical area bucket. Only entries that carried an actual
    # sub-classification (i.e. had something after the first ``/``)
    # end up in the returned dict, so an area whose LLM tags were all
    # bare parents contributes no tooltip content — the dashboard
    # renderer treats an empty list as "nothing to expand".
    area_sub_classifications: Dict[str, List[str]] = sub_classifications_for(
        raw
        for raws in area_raw_labels.values()
        for raw in raws
    )

    result = WeightedImpactResult(
        overall_impact_score=overall,
        impact_rating=impact_rating(overall),
        factor_scores=factor_scores,
        weighted_scores=weighted_scores,
        weights=dict(weights),
        factor_details=factor_details,
        top_impacted_business_capabilities=top_business_caps,
        top_impacted_processes=top_processes,
        top_impacted_systems=top_systems,
        top_impacted_controls=top_controls,
        top_impacted_third_parties=top_third_parties,
        heatmap_rows=heatmap_rows,
        priority_areas=priority_areas,
        overall_priority_score=overall_priority,
        recommendations_input=rec_input,
        area_sub_classifications=area_sub_classifications,
    )
    logger.info(
        "Weighted impact computed. overall=%.2f rating=%s priority=%.2f",
        overall, result.impact_rating, overall_priority,
    )
    return result


# ---------------------------------------------------------------------------
# Demo / validation helper
# ---------------------------------------------------------------------------


def demo_result() -> WeightedImpactResult:
    """Return the reference example from the product spec (Overall = 79.75).

    Factor scores per spec:

        Regulatory Obligation Criticality  25%  * 90 = 22.5
        Business Capability Impact         20%  * 80 = 16.0
        Process Change Impact              15%  * 70 = 10.5
        Technology / System Impact         15%  * 85 = 12.75
        Control & Compliance Impact        10%  * 75 =  7.5
        Data / Reporting Impact            10%  * 60 =  6.0
        Third-Party / Vendor Impact         5%  * 90 =  4.5
                                                    = 79.75
    """
    overrides = {
        "Regulatory Obligation Criticality": 90.0,
        "Business Capability Impact": 80.0,
        "Process Change Impact": 70.0,
        "Technology / System Impact": 85.0,
        "Control & Compliance Impact": 75.0,
        "Data / Reporting Impact": 60.0,
        "Third-Party / Vendor Impact": 90.0,
    }
    return compute_weighted_impact(factor_overrides=overrides)


__all__ = [
    "DORA_IMPACT_FACTOR_WEIGHTS",
    "ImpactFactorBreakdown",
    "PriorityAreaBreakdown",
    "WeightedImpactResult",
    "compute_weighted_impact",
    "demo_result",
    "impact_rating",
    "priority_score",
    "validate_weights",
]
