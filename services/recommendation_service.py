"""Recommendation service.

Generates concrete CXO recommendations from a scored assessment. The
deterministic baseline is fully self-contained (no AI needed). An optional
GenAI augmentation hook is provided for when the PwC GenAI Shared Service is
reachable, but it is not required for the MVP to run.

The original team scripts had no equivalent of this module — the live cockpit
only surfaced four static ``cxo_status`` action strings. This module turns a
scored package into a structured, exportable list of remediations keyed by
impacted area, impacted function, and the underlying BRD requirements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .scoring_engine import cxo_status

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """One actionable remediation item, derived from a scored assessment."""

    recommendation_id: str
    title: str
    severity: str
    area: Optional[str]
    function: Optional[str]
    compliance_pct: Optional[float]
    rationale: str
    suggested_action: str
    suggested_owner: str
    mapped_requirement_ids: List[str] = field(default_factory=list)
    horizon: str = "Short-term"
    # v12: branch-log driven evidence summarising *why* this gap exists, taken
    # from the option-level branch routing the user followed during the
    # assessment. Empty string when no relevant branch entries are available.
    branch_evidence: str = ""
    branch_rule_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Severity, owner, horizon defaults (deterministic playbook)
# ---------------------------------------------------------------------------

_SEVERITY_ACTIONS = {
    "Critical": (
        "Escalate to the management body within the current governance cycle. "
        "Sponsor a funded remediation plan with a named accountable owner and weekly status reviews."
    ),
    "At risk": (
        "Initiate a targeted remediation plan with a named accountable owner. "
        "Track issue, evidence and approval status against the next governance forum."
    ),
    "Watch": (
        "Close residual gaps before executive sign-off. "
        "Validate evidence sufficiency and ownership clarity through SME review."
    ),
    "Ready": (
        "Maintain controls, refresh evidence on cycle, and validate at the next periodic governance review."
    ),
}

_SEVERITY_HORIZON = {
    "Critical": "Immediate (0-30 days)",
    "At risk": "Short-term (30-90 days)",
    "Watch": "Medium-term (90-180 days)",
    "Ready": "Steady-state (periodic)",
}

# Suggested owner is a heuristic mapping derived from the impacted function.
# Falls back to "Compliance / Programme Owner" if no match.
_OWNER_BY_FUNCTION = {
    "Execution / Client Activity": "Front Office / Business Owner",
    "Risk Management": "Chief Risk Officer",
    "Compliance & Legal": "Chief Compliance Officer",
    "Technology / IT Operations": "Chief Technology Officer",
    "Cyber Security": "Chief Information Security Officer",
    "Business Continuity / Resilience": "Business Continuity Manager",
    "Incident Management": "ICT Incident Response Lead",
    "Vendor / Third-Party Management": "Head of Vendor / Third-Party Risk",
    "Data Governance / Reporting": "Chief Data Officer",
    "Internal Audit / Assurance": "Head of Internal Audit",
    "Operations / Settlement": "Head of Operations",
    "Programme Management": "DORA Programme Manager",
    "Human Resources / Training": "Head of HR / Talent",
}


def _suggested_owner(function: Optional[str]) -> str:
    if not function:
        return "Compliance / Programme Owner"
    return _OWNER_BY_FUNCTION.get(function, "Compliance / Programme Owner")


# ---------------------------------------------------------------------------
# Rationale templates
# ---------------------------------------------------------------------------

def _area_rationale(area: str, function: str, score: float, status: str, req_ids: Sequence[str]) -> str:
    requirement_snippet = ", ".join(req_ids[:6]) if req_ids else "the mapped DORA requirements"
    return (
        f"The combined assessment for {area} / {function} scored {score:.1f}%, classified as {status}. "
        f"This indicates insufficient coverage, evidence, or ownership against {requirement_snippet}. "
        f"Without remediation, this pair will limit overall DORA readiness and may be raised during "
        f"internal audit or supervisory review."
    )


def _requirement_rationale(req_id: str, req_title: str, score: float, status: str) -> str:
    return (
        f"Requirement {req_id} ({req_title}) scored {score:.1f}% across the questions mapped to it, "
        f"classified as {status}. Remediating this requirement directly improves the overall compliance score "
        f"and reduces the gap visible to internal audit and competent authorities."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_requirement_titles(package: Mapping[str, Any]) -> Dict[str, str]:
    return {r.get("normalized_id", ""): r.get("requirement", "") for r in package.get("requirements", [])}


def _pair_to_requirement_ids(package: Mapping[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    out: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for pair in package.get("impact_pairs", []):
        key = (pair.get("area", ""), pair.get("function", ""))
        for rid in pair.get("requirement_ids", []):
            if rid not in out[key]:
                out[key].append(rid)
    return out


def _summarise_branch_log_for(
    branch_log: Sequence[Mapping[str, Any]],
    area: Optional[str],
    function: Optional[str],
    requirement_ids: Sequence[str],
) -> Tuple[str, List[str]]:
    """Build a short evidence snippet from branch_log entries that match the rec's scope.

    Returns ``(snippet, branch_rule_ids)``. Empty snippet when nothing matches.
    """
    if not branch_log:
        return "", []
    req_set = set(requirement_ids or [])
    relevant: List[Mapping[str, Any]] = []
    for entry in branch_log:
        if area and entry.get("area") and entry.get("area") != area:
            continue
        if function and entry.get("function") and entry.get("function") != function:
            continue
        entry_reqs = set(entry.get("mapped_requirement_ids") or [])
        if req_set and entry_reqs and not (req_set & entry_reqs):
            continue
        relevant.append(entry)
    if not relevant:
        return "", []
    rule_ids: List[str] = []
    snippets: List[str] = []
    for entry in relevant[:5]:
        selected = entry.get("selected_answer") or []
        if isinstance(selected, list):
            sel = ", ".join(str(s) for s in selected)
        else:
            sel = str(selected)
        rule_id = entry.get("branch_rule_id", "")
        if rule_id and rule_id != "generic_dynamic_followup":
            rule_ids.append(rule_id)
        children = entry.get("child_question_ids") or []
        children_text = f" (asked {', '.join(children)})" if children else ""
        snippets.append(
            f"on {entry.get('parent_question_id', '?')} the user selected '{sel}' "
            f"[{rule_id or entry.get('branch_source', 'branch')}]" + children_text
        )
    snippet = "Branch trace: " + "; ".join(snippets) + "."
    return snippet, sorted(set(rule_ids))


def generate_recommendations(
    package: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    min_severity: str = "Watch",
    top_n_requirements: int = 10,
    branch_log: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Recommendation]:
    """Produce a sorted list of recommendations from a scored assessment.

    Parameters
    ----------
    package
        The questionnaire package dict produced by
        :mod:`services.questionnaire_generator` (or loaded from JSON).
    evaluation
        The output of :func:`services.scoring_engine.evaluate`.
    min_severity
        Lowest severity to include. "Watch" includes Watch / At risk / Critical.
        "At risk" excludes Watch. "Ready" returns everything.
    top_n_requirements
        Cap the number of requirement-level recommendations appended.

    Returns
    -------
    List[Recommendation]
        Sorted by severity descending, then compliance score ascending.
    """
    severity_order = {"Critical": 0, "At risk": 1, "Watch": 2, "Ready": 3}
    min_rank = severity_order.get(min_severity, severity_order["Watch"])

    pair_scores: Mapping[Tuple[str, str], float] = evaluation.get("pair_scores", {})  # type: ignore[assignment]
    area_summary: Mapping[str, Mapping[str, Any]] = evaluation.get("area_summary", {})
    req_scores: Mapping[str, float] = evaluation.get("requirement_scores", {})

    pair_req_ids = _pair_to_requirement_ids(package)
    req_titles = _resolve_requirement_titles(package)

    recommendations: List[Recommendation] = []

    # 1) Area × Function pair recommendations (most actionable for CXO)
    counter = 1
    pair_items: List[Tuple[Tuple[str, str], float]] = list(pair_scores.items())
    pair_items.sort(key=lambda kv: kv[1])
    for (area, function), score in pair_items:
        status, _action = cxo_status(score)
        if severity_order[status] > min_rank:
            continue
        req_ids = pair_req_ids.get((area, function), [])
        recommendations.append(Recommendation(
            recommendation_id=f"REC-{counter:03d}",
            title=f"Improve {area} / {function} readiness",
            severity=status,
            area=area,
            function=function,
            compliance_pct=round(float(score), 1),
            rationale=_area_rationale(area, function, score, status, req_ids),
            suggested_action=_SEVERITY_ACTIONS[status],
            suggested_owner=_suggested_owner(function),
            mapped_requirement_ids=req_ids[:8],
            horizon=_SEVERITY_HORIZON[status],
        ))
        counter += 1

    # 2) Area-only recommendations (fallback when pair scores are sparse)
    if not pair_scores and area_summary:
        for area, summary in area_summary.items():
            score = float(summary.get("Compliance %", 0))
            status = str(summary.get("CXO status", cxo_status(score)[0]))
            if severity_order.get(status, 3) > min_rank:
                continue
            recommendations.append(Recommendation(
                recommendation_id=f"REC-{counter:03d}",
                title=f"Improve {area} readiness",
                severity=status,
                area=area,
                function=None,
                compliance_pct=round(score, 1),
                rationale=(
                    f"The aggregate score for {area} is {score:.1f}% ({status}). "
                    f"Targeted remediation in this area improves overall DORA readiness."
                ),
                suggested_action=_SEVERITY_ACTIONS.get(status, _SEVERITY_ACTIONS["Watch"]),
                suggested_owner=_suggested_owner(None),
                mapped_requirement_ids=[],
                horizon=_SEVERITY_HORIZON.get(status, _SEVERITY_HORIZON["Watch"]),
            ))
            counter += 1

    # 3) Top weakest requirements
    weakest = sorted(req_scores.items(), key=lambda kv: kv[1])[:top_n_requirements]
    for req_id, score in weakest:
        status, _ = cxo_status(score)
        if severity_order.get(status, 3) > min_rank:
            continue
        title = req_titles.get(req_id, req_id)
        recommendations.append(Recommendation(
            recommendation_id=f"REC-{counter:03d}",
            title=f"Close gap on {req_id}",
            severity=status,
            area=None,
            function=None,
            compliance_pct=round(float(score), 1),
            rationale=_requirement_rationale(req_id, title, score, status),
            suggested_action=_SEVERITY_ACTIONS[status],
            suggested_owner="DORA Programme Manager",
            mapped_requirement_ids=[req_id],
            horizon=_SEVERITY_HORIZON[status],
        ))
        counter += 1

    # Attach option-level branch evidence (v12) to each recommendation when a
    # branch_log is available. Empty when no relevant entries match the rec's
    # scope so behaviour is unchanged for legacy callers.
    if branch_log:
        for rec in recommendations:
            snippet, rule_ids = _summarise_branch_log_for(
                branch_log, rec.area, rec.function, rec.mapped_requirement_ids,
            )
            if snippet:
                rec.branch_evidence = snippet
                rec.branch_rule_ids = rule_ids
                rec.rationale = f"{rec.rationale} {snippet}"

    recommendations.sort(
        key=lambda r: (
            severity_order.get(r.severity, 99),
            r.compliance_pct if r.compliance_pct is not None else 999.0,
        )
    )
    for new_idx, rec in enumerate(recommendations, start=1):
        rec.recommendation_id = f"REC-{new_idx:03d}"
    return recommendations


def recommendations_to_dicts(recs: Sequence[Recommendation]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in recs]


# ---------------------------------------------------------------------------
# Optional GenAI augmentation hook
# ---------------------------------------------------------------------------

def enrich_recommendations_with_genai(
    recommendations: Sequence[Recommendation],
    package: Mapping[str, Any],
    *,
    client: Optional[Any] = None,
) -> List[Recommendation]:
    """Optionally re-write each recommendation's ``suggested_action`` using GenAI.

    If ``client`` is None or unavailable, the recommendations are returned
    unchanged. The Streamlit UI can call this when ``GenAIClient.try_create()``
    returned a valid client; otherwise it simply skips this step.

    Kept intentionally minimal in the MVP: the prompt asks the model to produce
    a one-paragraph, audit-ready phrasing of the existing action.
    """
    if client is None or not recommendations:
        return list(recommendations)

    # Import here to avoid a hard dependency on langchain at module-import time
    # for callers that only need the deterministic baseline.
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        return list(recommendations)

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a regulatory compliance advisor. Rewrite the supplied action as one concise paragraph "
            "(60-100 words) in formal business English. Do not invent new evidence, owners, or deadlines. "
            "Preserve the original intent."
        ),
        ("user", "Severity: {severity}\nArea: {area}\nFunction: {function}\nCurrent action: {action}\n"),
    ])
    enriched: List[Recommendation] = []
    for rec in recommendations:
        try:
            chain = template | client.llm
            response = chain.invoke(
                {
                    "severity": rec.severity,
                    "area": rec.area or "(unspecified)",
                    "function": rec.function or "(unspecified)",
                    "action": rec.suggested_action,
                }
            )
            text = getattr(response, "content", None) or str(response)
            if text and text.strip():
                rec.suggested_action = text.strip()
        except Exception:
            # Fail soft — keep the deterministic action.
            pass
        enriched.append(rec)
    return enriched


__all__ = [
    "Recommendation",
    "enrich_recommendations_with_genai",
    "generate_recommendations",
    "recommendations_to_dicts",
]
