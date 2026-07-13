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

import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .owner_registry import owner_for as _suggested_owner
from .scoring_engine import cxo_status

logger = logging.getLogger(__name__)

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
    logger.info(
        "Generating recommendations. min_severity=%s top_n=%d evaluation_areas=%d",
        min_severity, top_n_requirements,
        len(evaluation.get("area_summary") or {}),
    )
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

# Pydantic payload used by :func:`enrich_recommendations_with_genai`.
# Defining it at module scope (rather than inside the function) so the
# schema class survives Streamlit's hot-reload reference-equality checks
# and so we can share the same payload across every enrichment call in a
# batch.
try:
    from pydantic import BaseModel, Field

    class _ActionRewritePayload(BaseModel):  # type: ignore[misc]
        """Structured output for a GenAI-driven recommendation rewrite.

        The whole point of using a Pydantic schema (rather than free-form
        text) here is that the LangChain structured-output wrapper will
        REJECT any response that doesn't populate ``suggested_action``
        with a string. That eliminates the two failure modes of the
        original free-form call:

        1. Empty / off-topic responses (the model apologised, refused,
           or returned meta-language) → schema violation → we fall back
           to the deterministic action.
        2. Half-formed JSON, code fences, prose around the payload →
           schema violation → same graceful fallback.

        Combined with :func:`services.guardrails.safe_generate`, this
        moves the rewrite from "highest risk" to the same tier as every
        other structured LLM call in the codebase.
        """

        suggested_action: str = Field(
            description=(
                "One concise paragraph (60-100 words), formal business "
                "English, that preserves the exact intent of the input "
                "action. Do not invent new evidence, owners, deadlines "
                "or regulatory citations. Do not use AI meta-language."
            ),
        )
except ImportError:  # pragma: no cover - pydantic is a project-wide dep
    _ActionRewritePayload = None  # type: ignore[assignment]


def enrich_recommendations_with_genai(
    recommendations: Sequence[Recommendation],
    package: Mapping[str, Any],
    *,
    client: Optional[Any] = None,
) -> List[Recommendation]:
    """Optionally re-write each recommendation's ``suggested_action`` using GenAI.

    Every rewrite is now a **structured** LLM call: the model must
    populate the ``_ActionRewritePayload`` schema (a single Pydantic
    string field). This eliminates the "unstructured output" risk that
    the previous free-form :class:`ChatPromptTemplate` chain carried —
    schema-invalid responses are rejected outright, and the wrapper
    :func:`services.guardrails.safe_generate` applies the full text
    guardrail stack on the way out (meta-leakage scrub, citation
    validation, scope validation, speculation / URL / numeric checks).

    Contract:

    * ``client`` is ``None`` or ``_ActionRewritePayload`` isn't
      importable → recommendations are returned unchanged (deterministic
      baseline retained).
    * The LLM call succeeds AND the guardrails accept the payload → the
      original ``suggested_action`` is replaced.
    * The LLM call fails OR the guardrails reject the payload (empty
      response, meta-leakage, invented citation etc.) → the original
      deterministic action is kept.
    """
    if client is None or not recommendations:
        return list(recommendations)
    if _ActionRewritePayload is None:  # pragma: no cover - pydantic missing
        return list(recommendations)

    from .guardrails import safe_generate

    regulation = str((package or {}).get("regulation") or "") or None
    recs_list = list(recommendations)

    def _rewrite_one(rec: Recommendation) -> Recommendation:
        instruction = (
            "You are a regulatory compliance advisor. Rewrite the "
            "supplied action as one concise paragraph (60-100 words) "
            "in formal business English. Do NOT invent new evidence, "
            "owners, deadlines or regulatory citations. Preserve the "
            "original intent. Return only the rewritten action in the "
            "'suggested_action' field."
        )
        context = (
            f"Severity: {rec.severity}\n"
            f"Area: {rec.area or '(unspecified)'}\n"
            f"Function: {rec.function or '(unspecified)'}\n"
            f"Current action: {rec.suggested_action}\n"
        )
        try:
            payload, report = safe_generate(
                client,
                _ActionRewritePayload,
                f"Recommendation rewrite: {rec.area or rec.recommendation_id}",
                instruction,
                context,
                regulation=regulation,
                # No source corpus here — the rewrite is bounded to the
                # existing deterministic action string, not to the source
                # regulation text. Citation-ratio enforcement is disabled
                # accordingly (min_citation_ratio=0) so a well-formed
                # rewrite that happens to name-check an in-scope Article
                # isn't rejected because the corpus is empty.
                source_corpus="",
                text_fields=("suggested_action",),
                min_citation_ratio=0.0,
            )
        except Exception:
            logger.exception(
                "Legacy recommendation rewrite crashed for %s (keeping deterministic action).",
                rec.recommendation_id,
            )
            return rec
        if payload is not None and report.ok:
            rewritten = str(payload.suggested_action or "").strip()
            if rewritten:
                rec.suggested_action = rewritten
        return rec

    max_workers = _resolve_legacy_enrich_worker_count(len(recs_list))
    if max_workers <= 1 or len(recs_list) == 1:
        return [_rewrite_one(r) for r in recs_list]

    t0 = time.perf_counter()
    logger.info(
        "Legacy GenAI recommendation enrichment. recs=%d max_workers=%d",
        len(recs_list), max_workers,
    )
    slots: List[Optional[Recommendation]] = [None] * len(recs_list)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="legacy-enrich") as pool:
        futures = {pool.submit(_rewrite_one, r): i for i, r in enumerate(recs_list)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                slots[idx] = fut.result()
            except Exception:
                logger.exception(
                    "Legacy recommendation enrichment future failed at index %d.", idx,
                )
                slots[idx] = recs_list[idx]
    logger.info(
        "Legacy GenAI recommendation enrichment done. recs=%d elapsed=%.2fs",
        len(recs_list), time.perf_counter() - t0,
    )
    return [r if r is not None else recs_list[i] for i, r in enumerate(slots)]


def _resolve_legacy_enrich_worker_count(rec_count: int) -> int:
    """Pick a bounded thread-pool size for parallel legacy enrichment.

    Reads ``LEGACY_RECOMMENDATION_ENRICH_WORKERS`` so ops can tune it
    without a code change. Default of 8 is comfortably below the shared
    service's burst ceiling while cutting the historical sequential
    ~120s down to ~15s for a typical dashboard.
    """
    if rec_count <= 1:
        return 1
    try:
        configured = int(os.getenv("LEGACY_RECOMMENDATION_ENRICH_WORKERS", "8"))
    except ValueError:
        configured = 8
    return max(1, min(configured, rec_count))


__all__ = [
    "Recommendation",
    "enrich_recommendations_with_genai",
    "generate_recommendations",
    "recommendations_to_dicts",
]
