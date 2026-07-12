"""Adaptive scoring engine for the live Streamlit cockpit.

Consolidates the funnel/heatmap/scoring logic that was duplicated between
``dora_readiness_streamlit_app_v11.py`` (live cockpit) and the deterministic
``evaluate_responses`` in ``services/questionnaire_generator.py``. There is
now a single source of truth.

The Streamlit-specific access to ``st.session_state`` has been replaced by an
explicit :class:`AssessmentState` dataclass. The engine functions are pure:
they take state in, return state out (or new artefacts), and never touch
Streamlit globals. The Phase 7 UI will hold one ``AssessmentState`` in
session and call these functions.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pandas as pd

from .ai_branch_generator import LLMInvoker, generate_option_followups
from .branch_registry import lookup_branch
from .questionnaire_generator import (
    ANSWER_SCORES,
    Question,
    option_label,
    option_labels,
    option_metadata,
    question_kind as _qg_question_kind,
)
from .severity import action_for_readiness, readiness_band, readiness_label

# ---------------------------------------------------------------------------
# Signal sets (lifted verbatim from dora_readiness_streamlit_app_v11.py)
# ---------------------------------------------------------------------------

GAP_SIGNALS = {
    "Partially", "Partially complete", "Mostly complete", "Not started", "No", "Unknown",
    "Ad hoc", "No evidence available", "No owner assigned", "Informal owner",
    "Medium", "High", "Critical",
    # v13 — canonical implementation-status family
    "Partially Implemented", "Not Implemented",
}
POSITIVE_SIGNALS = {
    "Yes", "Complete", "Implemented", "Measured / Optimised",
    "Named accountable owner", "Low",
    # v13 — canonical implementation-status family
    "Fully Implemented",
}
WEAK_OWNERSHIP = {"No owner assigned", "Informal owner", "Shared ownership", "Unknown"}
WEAK_EVIDENCE = {"No evidence available", "Unknown"}
HIGH_RISK = {"Medium", "High", "Critical", "Unknown"}
NEGATIVE_COVERAGE = {
    "Partially", "Partially complete", "Mostly complete", "Not started", "No", "Ad hoc", "Unknown",
    # v13 — canonical implementation-status family
    "Partially Implemented", "Not Implemented",
}

# Configurable adaptive-branching budget. Defaults are deliberately higher than
# the previous (1, 1, ∞) caps so registry-driven multi-step branches can fire,
# but still bounded to prevent runaway questioning.
MAX_DYNAMIC_FOLLOWUP_DEPTH = int(os.getenv("MAX_DYNAMIC_FOLLOWUP_DEPTH", "3"))
MAX_DYNAMIC_FOLLOWUPS_PER_PARENT = int(os.getenv("MAX_DYNAMIC_FOLLOWUPS_PER_PARENT", "3"))
MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT = int(os.getenv("MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT", "50"))


# ---------------------------------------------------------------------------
# Assessment state — replaces the st.session_state access in the old app
# ---------------------------------------------------------------------------

@dataclass
class AssessmentState:
    """In-flight assessment state. One per Streamlit session."""

    responses: Dict[str, Any] = field(default_factory=dict)
    dynamic_queue: List[Dict[str, Any]] = field(default_factory=list)
    skipped_ids: Set[str] = field(default_factory=set)
    display_numbers: Dict[str, int] = field(default_factory=dict)
    display_counter: int = 0
    history: List[str] = field(default_factory=list)

    # v12 adaptive-branching state -------------------------------------------
    # Audit trail capturing every parent-answer -> child-question routing
    # decision made during the assessment.
    branch_log: List[Dict[str, Any]] = field(default_factory=list)
    # Number of dynamic (registry + generic) follow-ups ever surfaced.
    # Hard cap protects against pathological loops or oversized registries.
    dynamic_questions_emitted: int = 0
    # Stable set of dynamic question_ids that have already been surfaced to
    # the user. Prevents the same branch firing twice if the user revisits a
    # parent question and re-submits.
    emitted_dynamic_ids: Set[str] = field(default_factory=set)

    def reset_responses(self) -> None:
        self.responses.clear()
        self.dynamic_queue.clear()
        self.skipped_ids.clear()
        self.display_numbers.clear()
        self.display_counter = 0
        self.history.clear()
        self.branch_log.clear()
        self.dynamic_questions_emitted = 0
        self.emitted_dynamic_ids.clear()

    def assign_display_number(self, question_id: str) -> int:
        if question_id not in self.display_numbers:
            self.display_counter += 1
            self.display_numbers[question_id] = self.display_counter
        return self.display_numbers[question_id]

    def remaining_dynamic_budget(self) -> int:
        return max(0, MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT - self.dynamic_questions_emitted)


# ---------------------------------------------------------------------------
# Question dict helpers (questions can arrive as dicts from JSON or as the
# Question dataclass; we normalize using dict-like access)
# ---------------------------------------------------------------------------

def _as_dict(q: Any) -> Dict[str, Any]:
    if isinstance(q, dict):
        return q
    if hasattr(q, "__dict__"):
        return dict(q.__dict__)
    raise TypeError(f"Unsupported question type: {type(q).__name__}")


def question_kind(q: Any) -> str:
    """Stable kind classification used by deduplication and funnel routing."""
    if isinstance(q, Question):
        return _qg_question_kind(q)
    d = _as_dict(q)
    text = (d.get("question") or "").lower()
    # v13 — canonical L1 status question maps to "coverage" so the registry,
    # scoring and skip rules keep working consistently.
    if text.startswith("is the ") and "implemented" in text:
        return "coverage"
    if "implementation coverage" in text or "current implementation" in text or "coverage" in text:
        return "coverage"
    if "ownership" in text or "accountable" in text or "owner" in text:
        return "ownership"
    if "evidence" in text or "substantiating" in text or "artefact" in text:
        return "evidence"
    if "residual" in text or "risk" in text:
        return "risk"
    if "remediation" in text or "gap" in text:
        return "remediation"
    if d.get("is_free_text"):
        return "free_text"
    return "general"


def response_values(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def is_positive_answer(value: Any) -> bool:
    vals = response_values(value)
    return bool(vals) and all(
        v in POSITIVE_SIGNALS or v == "Not applicable" or v == "Not Applicable"
        for v in vals
    )


def has_overlap(left: Iterable[str], right: Iterable[str], min_overlap: int = 1) -> bool:
    return len(set(left or []) & set(right or [])) >= min_overlap


def answered(q: Any, responses: Dict[str, Any]) -> bool:
    qid = _as_dict(q).get("question_id")
    return qid in responses and responses[qid] not in (None, "", [])


# ---------------------------------------------------------------------------
# Dynamic follow-up generation (lifted verbatim with state-arg refactor)
# ---------------------------------------------------------------------------

def stable_dynamic_id(parent_id: str, suffix: str) -> str:
    base = parent_id or "Q-0000"
    if base.startswith("DQ-"):
        base = base.split("__", 1)[0].replace("DQ-", "Q-")
    return f"DQ-{base.replace('Q-', '')}__{suffix}"


def dynamic_depth(q: Any) -> int:
    try:
        return int(_as_dict(q).get("dynamic_depth", 0))
    except Exception:
        return 0


def _child_explainability(parent: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """Inherit the parent's explainability bundle, overriding the reason field."""
    parent_explain = dict(parent.get("explainability") or {})
    if reason:
        parent_explain["reason"] = reason
    return parent_explain


def build_followup(parent: Dict[str, Any], suffix: str, question: str, options: List[Any], reason: str) -> Dict[str, Any]:
    root_id = parent.get("root_question_id") or parent.get("question_id", "Q-0000")
    return {
        "question_id": stable_dynamic_id(root_id, suffix),
        "area": parent.get("area", "Impacted area"),
        "function": parent.get("function", "Impacted function"),
        "question_type": "Single Select",
        "question": question,
        "options": options,
        "mapped_requirement_ids": parent.get("mapped_requirement_ids", []),
        "regulatory_basis": parent.get("regulatory_basis", "Mapped regulatory requirement"),
        "confidence": max(90, min(97, int(parent.get("confidence", 94)) - 1)),
        "scoring_weight": min(5, int(parent.get("scoring_weight", 2)) + 1),
        "funnel_parent_id": parent.get("question_id", ""),
        "source_parent_id": parent.get("source_parent_id") or parent.get("question_id", ""),
        "root_question_id": root_id,
        "dynamic_depth": dynamic_depth(parent) + 1,
        "trigger_answers": [],
        "rationale": reason,
        "is_free_text": False,
        "dynamic": True,
        "branch_theme": parent.get("branch_theme", ""),
        "branch_rule_id": "generic_dynamic_followup",
        "explainability": _child_explainability(parent, reason),
    }


# ---------------------------------------------------------------------------
# Registry-driven option-level branching (v12)
# ---------------------------------------------------------------------------

def _resolve_regulation(parent: Mapping[str, Any], package_regulation: Optional[str]) -> str:
    """Resolve the regulation label for registry lookup."""
    explicit = parent.get("regulation")
    if explicit:
        return str(explicit)
    if package_regulation:
        return str(package_regulation)
    basis = parent.get("regulatory_basis", "")
    if "DORA" in basis.upper():
        return "DORA"
    return "DORA"


def materialize_branch_spec(
    parent: Mapping[str, Any],
    spec: Mapping[str, Any],
    selected_answer: str,
    depth: int,
) -> Dict[str, Any]:
    """Turn a branch_registry spec dict into a queue-ready question dict.

    The spec ID is namespaced by the parent question_id so the same registry
    entry triggered from two different base questions yields two distinct,
    independently-answerable child questions.
    """
    parent_id = parent.get("question_id", "Q-0000")
    base_qid = str(spec.get("question_id") or "BRANCH").strip() or "BRANCH"
    namespaced_qid = f"DQ-{parent_id.replace('Q-', '').replace('DQ-', '')}__{base_qid}"
    options = spec.get("options") or ["Unknown"]
    rule_id = str(spec.get("branch_rule_id") or f"branch::{base_qid}")
    rationale = str(spec.get("rationale") or (
        f"Branch follow-up triggered because the user selected '{selected_answer}' on "
        f"{parent_id}. Sourced from the regulation-specific branch registry."
    ))
    return {
        "question_id": namespaced_qid,
        "area": spec.get("area") or parent.get("area", "Impacted area"),
        "function": spec.get("function") or parent.get("function", "Impacted function"),
        "question_type": spec.get("question_type") or "Single Select",
        "question": spec.get("question", ""),
        "options": options,
        "mapped_requirement_ids": list(
            spec.get("mapped_requirement_ids") or parent.get("mapped_requirement_ids", [])
        ),
        "regulatory_basis": spec.get("regulatory_basis") or parent.get("regulatory_basis", ""),
        "confidence": int(spec.get("confidence") or max(90, min(97, int(parent.get("confidence", 94)) - 1))),
        "scoring_weight": int(spec.get("scoring_weight") or min(5, int(parent.get("scoring_weight", 2)) + 1)),
        "funnel_parent_id": parent_id,
        "source_parent_id": parent.get("source_parent_id") or parent_id,
        "root_question_id": parent.get("root_question_id") or parent_id,
        "dynamic_depth": depth,
        "trigger_answers": [selected_answer],
        "rationale": rationale,
        "is_free_text": False,
        "dynamic": True,
        "branch_theme": spec.get("branch_theme") or parent.get("branch_theme", ""),
        "branch_rule_id": rule_id,
        "branch_path": [parent_id, base_qid],
        "explainability": _child_explainability(parent, rationale),
    }


def _infer_theme_from_parent(parent: Mapping[str, Any]) -> str:
    """Best-effort theme inference for legacy packages that pre-date branch_theme."""
    text = " ".join([
        str(parent.get("question", "")),
        str(parent.get("regulatory_basis", "")),
        str(parent.get("rationale", "")),
    ]).lower()
    rules = [
        ("Incident reporting", ["incident", "classification", "notification", "tlpt"]),
        ("Third-party risk", ["third-party", "third party", "vendor", "subcontract"]),
        ("Resilience testing", ["resilience test", "recovery", "backup", "restore", "continuity"]),
        ("Security and access", ["privileged access", "siem", "encryption", "vulnerability"]),
        ("Governance", ["management body", "governance", "board approval"]),
        ("Data and evidence", ["evidence dictionary", "data lineage", "metadata"]),
        ("Reporting", ["dashboard", "kri", "kpi", "management report"]),
    ]
    for theme, keys in rules:
        if any(k in text for k in keys):
            return theme
    return ""


def registry_followups(
    parent: Mapping[str, Any],
    response: Any,
    package_regulation: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Look up option-level branch specs for the parent's selected answer(s).

    Returns ``(materialised_questions, branch_rule_ids)``. An empty list of
    questions signals "no registered branch — caller should fall back to
    :func:`ai_option_followups` then ``dynamic_followups``".
    """
    if dynamic_depth(parent) >= MAX_DYNAMIC_FOLLOWUP_DEPTH:
        return [], []
    regulation = _resolve_regulation(parent, package_regulation)
    theme = (parent.get("branch_theme") or "").strip() or _infer_theme_from_parent(parent)
    kind = question_kind(parent)
    answers = response_values(response)
    if not answers or not theme:
        return [], []

    next_depth = dynamic_depth(parent) + 1
    materialised: List[Dict[str, Any]] = []
    rule_ids: List[str] = []
    seen_ids: Set[str] = set()
    for answer in answers:
        specs = lookup_branch(regulation, theme, kind, answer)
        for spec in specs:
            child = materialize_branch_spec(parent, spec, answer, next_depth)
            cid = child["question_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            materialised.append(child)
            rule_ids.append(child.get("branch_rule_id", ""))
            if len(materialised) >= MAX_DYNAMIC_FOLLOWUPS_PER_PARENT:
                return materialised, rule_ids
    return materialised, rule_ids


def ai_option_followups(
    parent: Mapping[str, Any],
    response: Any,
    *,
    llm_invoker: Optional[LLMInvoker] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Generate AI-driven option-specific follow-ups when no static branch is registered.

    Uses :func:`services.ai_branch_generator.generate_option_followups` which
    either calls GenAI (when ``llm_invoker`` is provided and succeeds) or
    falls back to deterministic option-aware templates. The returned dicts
    have the same shape as registry specs, so the same
    :func:`materialize_branch_spec` is used to namespace them under the
    parent's question_id.

    Returns ``(materialised_questions, branch_rule_ids)``. Empty list means
    the caller should fall through to :func:`dynamic_followups` (the legacy
    signal-banded family).
    """
    if dynamic_depth(parent) >= MAX_DYNAMIC_FOLLOWUP_DEPTH:
        return [], []
    answers = response_values(response)
    if not answers:
        return [], []

    next_depth = dynamic_depth(parent) + 1
    materialised: List[Dict[str, Any]] = []
    rule_ids: List[str] = []
    seen_ids: Set[str] = set()
    for answer in answers:
        specs = generate_option_followups(parent, answer, llm_invoker=llm_invoker)
        for spec in specs:
            child = materialize_branch_spec(parent, spec, answer, next_depth)
            cid = child["question_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            materialised.append(child)
            rule_ids.append(child.get("branch_rule_id", ""))
            if len(materialised) >= MAX_DYNAMIC_FOLLOWUPS_PER_PARENT:
                return materialised, rule_ids
    return materialised, rule_ids


def dynamic_followups(parent: Dict[str, Any], response: Any) -> List[Dict[str, Any]]:
    if dynamic_depth(parent) >= MAX_DYNAMIC_FOLLOWUP_DEPTH:
        return []
    vals = set(response_values(response))
    if not vals:
        return []
    area = parent.get("area", "the impacted area")
    function = parent.get("function", "the impacted function")
    basis = parent.get("regulatory_basis", "the mapped regulatory requirement")
    followups: List[Dict[str, Any]] = []
    if vals & NEGATIVE_COVERAGE:
        followups.append(build_followup(
            parent, "NEXT-GAP",
            f"Which specific gap is most likely to prevent {function} from meeting the {area} requirement?",
            ["Control design gap", "Evidence gap", "Ownership gap", "Technology or data gap",
             "Third-party or contract gap", "Governance approval gap", "Regulatory interpretation gap", "Unknown"],
            f"The previous response indicated incomplete readiness. This follow-up isolates the most material blocker against {basis}.",
        ))
    if vals & WEAK_EVIDENCE:
        followups.append(build_followup(
            parent, "NEXT-EVIDENCE",
            f"What is the earliest evidence that can be produced to substantiate {area} readiness for {function}?",
            ["Policy / procedure", "Workflow record", "System report", "Dashboard",
             "Attestation", "Audit trail", "Contract evidence", "No evidence available"],
            f"Tests the fastest evidence path for {basis}.",
        ))
    if vals & WEAK_OWNERSHIP:
        followups.append(build_followup(
            parent, "NEXT-OWNER",
            f"Who should be accountable for closing the {area} accountability gap for {function}?",
            ["Business owner", "Technology owner", "Compliance owner", "Risk owner",
             "Vendor management owner", "Joint ownership with named lead", "No owner agreed", "Unknown"],
            "Routes the assessment to ownership; remediation cannot be reliably scored without a named accountable owner.",
        ))
    if vals & HIGH_RISK:
        followups.append(build_followup(
            parent, "NEXT-RISK",
            f"What executive action is required for the residual {area} risk in {function}?",
            ["Accept risk temporarily", "Fund remediation", "Escalate to governance forum",
             "Redesign control", "Accelerate evidence collection", "Renegotiate third-party terms",
             "No action required", "Unknown"],
            f"Translates the regulatory gap into an executive decision point for {basis}.",
        ))
    if is_positive_answer(response):
        followups.append(build_followup(
            parent, "NEXT-VALIDATE",
            f"What final validation is needed before marking {area} / {function} as CXO-ready?",
            ["No further validation", "SME review", "Legal or Compliance review", "Evidence sample test",
             "Control effectiveness test", "Management approval", "Unknown"],
            f"Prevents over-scoring by confirming the readiness claim has been independently reviewed against {basis}.",
        ))
    for f in followups:
        f["trigger_answers"] = sorted(vals)
    return followups[:MAX_DYNAMIC_FOLLOWUPS_PER_PARENT]


# ---------------------------------------------------------------------------
# Queue management + funnel routing
# ---------------------------------------------------------------------------

def add_to_queue(state: AssessmentState, new_items: List[Dict[str, Any]]) -> None:
    """Append new dynamic questions to the queue, honouring all caps."""
    existing_ids = {q.get("question_id") for q in state.dynamic_queue}
    answered_ids = set(state.responses)
    for item in new_items:
        if state.remaining_dynamic_budget() <= 0:
            return
        if dynamic_depth(item) > MAX_DYNAMIC_FOLLOWUP_DEPTH:
            continue
        qid = item.get("question_id")
        if not qid or qid in existing_ids or qid in answered_ids:
            continue
        # Per-assessment idempotency: once we've emitted a dynamic question we
        # never re-emit it, even if the user revisits the parent — this is the
        # main guard against infinite loops in pathological registries.
        if qid in state.emitted_dynamic_ids:
            continue
        state.dynamic_queue.append(item)
        existing_ids.add(qid)
        state.emitted_dynamic_ids.add(qid)
        state.dynamic_questions_emitted += 1


def _log_branch_decision(
    state: AssessmentState,
    parent: Mapping[str, Any],
    selected_answer_values: List[str],
    child_questions: List[Dict[str, Any]],
    source: str,
    branch_rule_ids: List[str],
) -> None:
    """Append a structured audit row to ``state.branch_log``."""
    if not child_questions:
        return
    state.branch_log.append({
        "parent_question_id": parent.get("question_id", ""),
        "selected_answer": selected_answer_values,
        "branch_rule_id": ", ".join([r for r in branch_rule_ids if r]) or source,
        "branch_source": source,  # "registry" or "generic"
        "child_question_ids": [c.get("question_id") for c in child_questions],
        "regulation": _resolve_regulation(parent, None),
        "theme": parent.get("branch_theme", ""),
        "question_kind": question_kind(parent),
        "area": parent.get("area", ""),
        "function": parent.get("function", ""),
        "mapped_requirement_ids": list(parent.get("mapped_requirement_ids", [])),
        "depth": dynamic_depth(parent) + 1,
    })


def update_applicability_after_response(
    state: AssessmentState,
    answered_q: Dict[str, Any],
    value: Any,
    base_questions: List[Dict[str, Any]],
    package_regulation: Optional[str] = None,
    *,
    llm_invoker: Optional[LLMInvoker] = None,
) -> None:
    """Adjust the remaining queue + skip list + branch log based on the latest answer.

    Routing order (v13):
      1. Discard any pending dynamic questions tied to this question's root.
      2. If the answered question itself is dynamic, stop (children of dynamic
         questions are themselves children of the original root and use the
         same depth tracking, but we do not re-skip base questions from them).
      3. Try the regulation-aware **branch registry** first. This is the
         hand-curated, option-specific source of truth.
      4. If no static branch is registered, ask the **AI option-aware
         generator** for option-specific follow-ups. When ``llm_invoker``
         is ``None`` (or fails) the generator returns deterministic
         option-aware templates — selecting Fully vs Partially vs Not
         Implemented yields visibly different questions.
      5. Only if both 3 and 4 return nothing, fall through to the legacy
         signal-banded ``dynamic_followups`` family.
    """
    if answered_q.get("is_free_text"):
        return
    qid = answered_q.get("question_id")
    vals_list = response_values(value)
    vals = set(vals_list)
    parent_kind = question_kind(answered_q)
    area = answered_q.get("area")
    function = answered_q.get("function")
    reqs = answered_q.get("mapped_requirement_ids", [])

    root = answered_q.get("root_question_id") or qid
    state.dynamic_queue = [
        q for q in state.dynamic_queue
        if q.get("root_question_id") != root or answered(q, state.responses)
    ]

    # Step 3: try registry-driven option-level branching first. This path
    # applies whether the answer is positive, negative or neutral — the
    # registry decides per option what to do next.
    registry_questions, rule_ids = registry_followups(
        answered_q, value, package_regulation=package_regulation,
    )
    if registry_questions:
        before = state.dynamic_questions_emitted
        add_to_queue(state, registry_questions)
        emitted_count = state.dynamic_questions_emitted - before
        accepted = registry_questions[:emitted_count] if emitted_count else []
        if accepted:
            _log_branch_decision(state, answered_q, vals_list, accepted, "registry", rule_ids)
        if is_positive_answer(value) and not answered_q.get("dynamic"):
            _apply_positive_skip(state, answered_q, value, base_questions, area, function, reqs, parent_kind, qid)
        return

    if answered_q.get("dynamic"):
        return

    # Step 4: AI option-aware fallback BEFORE generic signal-banded
    # follow-ups. The AI generator emits truly per-option questions even
    # when GenAI is offline (deterministic templates), so this is the
    # mechanism that delivers "true adaptive branching" outside the
    # registered theme/kind combinations.
    ai_questions, ai_rule_ids = ai_option_followups(
        answered_q, value, llm_invoker=llm_invoker,
    )
    if ai_questions:
        before = state.dynamic_questions_emitted
        add_to_queue(state, ai_questions)
        emitted_count = state.dynamic_questions_emitted - before
        accepted = ai_questions[:emitted_count] if emitted_count else []
        if accepted:
            _log_branch_decision(state, answered_q, vals_list, accepted, "ai_option", ai_rule_ids)
        if is_positive_answer(value) and not answered_q.get("dynamic"):
            _apply_positive_skip(state, answered_q, value, base_questions, area, function, reqs, parent_kind, qid)
        return

    # Step 5a: weak answer -> generic engine
    if vals & GAP_SIGNALS:
        generic = dynamic_followups(answered_q, value)
        before = state.dynamic_questions_emitted
        add_to_queue(state, generic)
        emitted_count = state.dynamic_questions_emitted - before
        accepted = generic[:emitted_count] if emitted_count else []
        if accepted:
            _log_branch_decision(state, answered_q, vals_list, accepted, "generic",
                                 [a.get("branch_rule_id", "") for a in accepted])
        return

    # Step 5b: positive answer -> skip downstream + queue validation follow-up
    if is_positive_answer(value):
        _apply_positive_skip(state, answered_q, value, base_questions, area, function, reqs, parent_kind, qid)
        generic = dynamic_followups(answered_q, value)
        before = state.dynamic_questions_emitted
        add_to_queue(state, generic)
        emitted_count = state.dynamic_questions_emitted - before
        accepted = generic[:emitted_count] if emitted_count else []
        if accepted:
            _log_branch_decision(state, answered_q, vals_list, accepted, "generic",
                                 [a.get("branch_rule_id", "") for a in accepted])


def _apply_positive_skip(
    state: AssessmentState,
    answered_q: Mapping[str, Any],
    value: Any,
    base_questions: List[Dict[str, Any]],
    area: Optional[str],
    function: Optional[str],
    reqs: Iterable[str],
    parent_kind: str,
    qid: Optional[str],
) -> None:
    """Mark downstream risk/remediation questions as skipped for a positive answer."""
    skippable_kinds = {"risk", "remediation"}
    if parent_kind in {"ownership", "evidence"}:
        skippable_kinds = {"remediation"}
    for q in base_questions:
        if q.get("question_id") == qid or q.get("is_free_text"):
            continue
        same_context = (
            q.get("area") == area
            and q.get("function") == function
            and has_overlap(q.get("mapped_requirement_ids", []), reqs)
        )
        if same_context and question_kind(q) in skippable_kinds and q.get("question_id") not in state.responses:
            state.skipped_ids.add(q.get("question_id"))


def applicable_base_questions(
    state: AssessmentState, base_questions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [q for q in base_questions if not q.get("is_free_text") and q.get("question_id") not in state.skipped_ids]


def choose_next_question(
    state: AssessmentState,
    base_questions: List[Dict[str, Any]],
    focus_area: str = "All",
) -> Optional[Dict[str, Any]]:
    """Return the next question to ask, or ``None`` if the focus is complete."""
    pending_dynamic = [q for q in state.dynamic_queue if not answered(q, state.responses)]
    candidates = [q for q in applicable_base_questions(state, base_questions)
                  if not answered(q, state.responses)]

    if focus_area != "All":
        pending_dynamic = [q for q in pending_dynamic if q.get("area") == focus_area]
        candidates = [q for q in candidates if q.get("area") == focus_area]

    if pending_dynamic:
        return pending_dynamic[0]
    if not candidates:
        return None

    applicable = applicable_base_questions(state, base_questions)
    answered_area_counts = Counter(q.get("area") for q in applicable if answered(q, state.responses))
    answered_function_counts = Counter(q.get("function") for q in applicable if answered(q, state.responses))

    def rank(q: Dict[str, Any]) -> Tuple[int, int, float, str]:
        area_need = -answered_area_counts[q.get("area")]
        function_need = -answered_function_counts[q.get("function")]
        confidence_weight = float(q.get("confidence", 90)) + int(q.get("scoring_weight", 1)) * 10
        return (area_need, function_need, confidence_weight, q.get("question_id", ""))

    return sorted(candidates, key=rank, reverse=True)[0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_NA_LABEL_TOKENS = frozenset({
    "n/a", "na", "not applicable", "not available",
})

_ENUMERATION_NEGATIVE_KEYWORDS: Tuple[str, ...] = (
    "not audited", "not tested", "not implemented", "not covered",
    "not in place", "not documented", "not compliant", "not monitored",
    "not reviewed", "not owned", "not defined", "not established",
    "not started", "not addressed", "without", "missing", "outstanding",
    " gap", " gaps ", "unaudited", "untested", "unmonitored",
    "no evidence", "no owner", "no policy", "no procedure",
    "un-audited", "un-tested",
)


def _enumeration_directionality(question_text: str) -> int:
    """Return ``+1`` when picking = coverage (good) and ``-1`` when picking = gap (bad).

    Direction is inferred from question wording:

    * ``-1`` when the question asks about things that are *missing / not
      done / outstanding* (each pick is a gap; more picks → worse score).
    * ``+1`` otherwise (default treats picks as coverage / capability).
    """
    text = (question_text or "").lower()
    if any(kw in text for kw in _ENUMERATION_NEGATIVE_KEYWORDS):
        return -1
    return 1


def score_value(value: Any, question: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    """Score a single answer.

    Resolution order:

    1. **Per-option ``score_value`` metadata** — the AI questionnaire
       generator's authoritative signal. Explicit ``score_value=None``
       on an option means "this is a real N/A" and the answer is
       excluded from scoring.
    2. **Legacy answer table** — for plain-string option labels that
       appear in :data:`ANSWER_SCORES` (``Yes`` / ``Complete`` /
       ``Not started`` / …).
    3. **Enumeration fallback** — when the answer is a plain-string
       pick that carries no score metadata (e.g. picking ``"Software"``
       from a ``["Software", "Hardware", "Data", …]`` checklist), the
       scorer no longer silently discards it. Instead the answer is
       scored by pick-ratio, with directionality inferred from the
       question wording: "Which X are NOT audited" → each pick is a
       gap (fewer picks = higher readiness), "Which X are in place" →
       each pick is coverage (more picks = higher readiness).
    4. **Explicit N/A** — when the answer text itself is an N/A phrase
       (``"N/A"``, ``"Not applicable"``, …) the answer is excluded.

    Steps 1-3 combine so that questions where the AI didn't attach
    numeric per-option scores are still counted — the previous
    behaviour silently dropped these answers as "not applicable" which
    surprised reviewers.
    """
    vals = response_values(value)
    if not vals:
        return None

    # Step 4 first: an answer that is literally "N/A" / "Not applicable"
    # is always excluded, regardless of what the option metadata says.
    lower_vals = [str(v).strip().lower() for v in vals]
    if lower_vals and all(v in _NA_LABEL_TOKENS for v in lower_vals):
        return None

    # Step 1: per-option ``score_value`` metadata.
    if question is not None:
        opts = question.get("options") or []
        per_option_scores: List[float] = []
        na_metadata_hits = 0
        for v in vals:
            meta = option_metadata(opts, v)
            if not meta:
                continue
            if "score_value" in meta:
                raw = meta["score_value"]
                if raw is None:
                    na_metadata_hits += 1
                    continue  # explicit "N/A" on this option — skip
                try:
                    per_option_scores.append(float(raw))
                except (TypeError, ValueError):
                    pass
        if per_option_scores:
            return sum(per_option_scores) / len(per_option_scores)
        # If every picked option was explicitly N/A via metadata, exclude.
        if na_metadata_hits and na_metadata_hits == len(vals):
            return None

    # Step 2: legacy string-label table.
    scores = [ANSWER_SCORES[v] for v in vals if v in ANSWER_SCORES and ANSWER_SCORES[v] is not None]
    if scores:
        return sum(scores) / len(scores)

    # Step 3: enumeration fallback for real picks that neither the AI
    # metadata nor the legacy table can score. Direction is inferred from
    # the question wording so "Which X are NOT audited?" penalises picks
    # and "Which controls are in place?" rewards them.
    if question is not None:
        opts = question.get("options") or []
        total = len(opts)
        if total > 0:
            direction = _enumeration_directionality(
                str(question.get("question") or "")
            )
            picked_ratio = min(1.0, len(vals) / float(total))
            if direction < 0:
                return round(max(0.0, 100.0 * (1.0 - picked_ratio)), 1)
            return round(min(100.0, 100.0 * picked_ratio), 1)

    return None


# ---------------------------------------------------------------------------
# Free-text quality scoring
# ---------------------------------------------------------------------------

# Filler tokens shared with ``services.ai_assessment_intelligence`` — kept
# inline here so this module has no runtime dependency on the AI package.
_FT_FILLER_TOKENS = frozenset({
    "yes", "no", "maybe", "ok", "okay", "sure", "n/a", "na", "none",
    "tbd", "unknown", "not sure", "not applicable", "not started",
    "in progress", "partial", "some", "few", "many", "several",
    "nothing", "everything", "all", "any", "idk", "n a", "not yet",
})

# Simple positive / negative signal lexicons. A well-formed SME narrative
# tends to mention concrete artefacts, owners, dates or metrics — the
# scorer rewards that signal density and penalises pure vagueness.
_FT_POSITIVE_TOKENS = (
    "policy", "policies", "control", "controls", "framework", "process",
    "processes", "procedure", "procedures", "runbook", "playbook",
    "documented", "documentation", "evidence", "artefact", "artifact",
    "raci", "owner", "accountable", "responsible", "committee", "board",
    "governance", "signed", "approved", "audit", "audited", "reviewed",
    "monitored", "monitoring", "tested", "testing", "tracked", "sla",
    "kpi", "kri", "metric", "metrics", "target", "threshold", "deadline",
    "quarterly", "monthly", "annually", "weekly", "reported", "report",
    "escalation", "budget", "funded", "invested", "trained", "training",
)
_FT_NEGATIVE_TOKENS = (
    "not sure", "unknown", "no idea", "unclear", "tbd", "to be defined",
    "not started", "not yet", "no evidence", "no owner", "ad hoc",
    "ad-hoc", "informal", "manual only", "no policy", "no process",
    "no documentation", "no funding", "no budget",
)


def score_free_text_answer(
    answer: Any,
    *,
    question_text: Optional[str] = None,
) -> Optional[float]:
    """Score a free-text SME answer for quality, returning 0-100 or ``None``.

    Signals:

    * length (words / characters) — very short answers score low;
    * signal density — mentions of policies, owners, evidence, cadences,
      metrics etc. push the score up;
    * negative-vagueness density — "not sure", "no owner", "tbd" etc.
      push the score down;
    * pure filler answers ("yes", "no", "n/a", "unknown") are scored 0
      unless the token is a valid explicit N/A (which returns ``None`` so
      the answer is *excluded* from scoring, matching the closed-question
      behaviour).

    Returns ``None`` when the answer is empty / N/A so the caller can skip
    it in the numerator/denominator.
    """
    if answer is None:
        return None
    raw = str(answer).strip()
    if not raw:
        return None

    normalised = raw.lower().strip(" .!?,;:'\"()[]-")
    if normalised in {"n/a", "na", "not applicable"}:
        return None
    if normalised in _FT_FILLER_TOKENS:
        return 5.0  # very low but not None, so this counts as "answered poorly"

    words = [w for w in normalised.split() if w]
    word_count = len(words)
    char_count = len(raw)

    length_score = 0.0
    if word_count >= 60 or char_count >= 400:
        length_score = 55.0
    elif word_count >= 30 or char_count >= 200:
        length_score = 45.0
    elif word_count >= 18 or char_count >= 120:
        length_score = 32.0
    elif word_count >= 10 or char_count >= 60:
        length_score = 20.0
    elif word_count >= 5 or char_count >= 30:
        length_score = 10.0

    text_lower = normalised
    positive_hits = sum(1 for tok in _FT_POSITIVE_TOKENS if tok in text_lower)
    negative_hits = sum(1 for tok in _FT_NEGATIVE_TOKENS if tok in text_lower)

    signal_bonus = min(35.0, positive_hits * 6.0)
    negative_penalty = min(25.0, negative_hits * 8.0)

    # Small extra credit when the answer references the question context
    # (a keyword from the question re-appears in the answer). This catches
    # focused, on-topic answers versus generic boilerplate.
    context_bonus = 0.0
    if question_text:
        q_words = {
            w.strip(".,;:!?()[]'\"").lower()
            for w in question_text.split()
            if len(w) > 4
        }
        answer_words = {w.strip(".,;:!?()[]'\"") for w in words}
        overlap = len(q_words & answer_words)
        context_bonus = min(10.0, overlap * 1.5)

    raw_score = length_score + signal_bonus + context_bonus - negative_penalty
    return max(0.0, min(100.0, round(raw_score, 1)))


def cxo_status(score: float) -> Tuple[str, str]:
    """Return the canonical severity label + executive action for a
    readiness / compliance score.

    Thin wrapper around :mod:`services.severity` — kept as a module-level
    function for backward compatibility with the many callers across the
    app that import ``cxo_status`` directly.

    Readiness bands (higher readiness = better):
        - score >= 75  -> Ready
        - score >= 50  -> Watch
        - score >= 25  -> At risk
        - score <  25  -> Critical
    """
    band = readiness_band(score)
    return readiness_label(band), action_for_readiness(band)


def evaluate(
    questions: List[Dict[str, Any]],
    state: AssessmentState,
) -> Dict[str, Any]:
    """Live evaluation suitable for the adaptive cockpit.

    Returns a dict containing the overall compliance %, evaluation confidence %,
    per-requirement scores, area summary, function summary, and the
    area×function pair matrix.

    Branch-specific dynamic questions in ``state.dynamic_queue`` are included
    in the scoring loop so registry-driven follow-ups contribute to readiness.
    Each dynamic Q's answer is scored once (de-duplicated by question_id).
    """
    area_num: Dict[str, float] = defaultdict(float)
    area_den: Dict[str, float] = defaultdict(float)
    area_counts: Dict[str, int] = defaultdict(int)
    func_num: Dict[str, float] = defaultdict(float)
    func_den: Dict[str, float] = defaultdict(float)
    func_counts: Dict[str, int] = defaultdict(int)
    pair_num: Dict[Tuple[str, str], float] = defaultdict(float)
    pair_den: Dict[Tuple[str, str], float] = defaultdict(float)
    req_num: Dict[str, float] = defaultdict(float)
    req_den: Dict[str, float] = defaultdict(float)
    total_num = 0.0
    total_den = 0.0
    answered_count = 0
    unanswered_count = 0

    seen_qids: Set[str] = set()
    all_questions: List[Dict[str, Any]] = []
    for q in list(questions) + list(state.dynamic_queue):
        qid = q.get("question_id")
        if not qid or qid in seen_qids:
            continue
        seen_qids.add(qid)
        all_questions.append(q)

    for q in all_questions:
        if q.get("question_id") in state.skipped_ids:
            continue
        # Composite weight = base scoring weight × impact weight (from the
        # AI-driven ``impact_severity``) × confidence.
        #   base   ∈ [1..5]  (AI-set + enhancer-bumped)
        #   impact ∈ [1..5]  (severity ladder; defaults to base when missing)
        #   conf   ∈ [0..1]  (per-question grounding confidence)
        # This makes questions in Critical/High-severity areas contribute
        # proportionally more to the overall readiness score, matching the
        # "high-impact areas score more heavily" requirement.
        base_w = float(q.get("scoring_weight", 1))
        impact_w = float(q.get("impact_weight", base_w))
        conf = float(q.get("confidence", 90)) / 100
        weight = base_w * max(1.0, impact_w) * conf

        is_free_text = bool(q.get("is_free_text"))
        raw_answer = state.responses.get(q.get("question_id"))
        if is_free_text:
            # Free-text answers now contribute to readiness / impact using
            # a deterministic quality scorer (length + signal density +
            # negative-vagueness penalty). The narrative bucket carries
            # ~30% of a closed question's weight so a well-written answer
            # nudges the score without letting SME essays dominate.
            score = score_free_text_answer(
                raw_answer, question_text=str(q.get("question") or ""),
            )
            weight = weight * 0.3
        else:
            score = score_value(raw_answer, q)

        if score is None:
            unanswered_count += 1
            continue
        answered_count += 1
        total_num += score * weight
        total_den += 100 * weight
        area = q.get("area")
        function = q.get("function")
        area_num[area] += score * weight
        area_den[area] += 100 * weight
        area_counts[area] += 1
        func_num[function] += score * weight
        func_den[function] += 100 * weight
        func_counts[function] += 1
        pair_num[(area, function)] += score * weight
        pair_den[(area, function)] += 100 * weight
        for rid in q.get("mapped_requirement_ids", []):
            req_num[rid] += score * weight
            req_den[rid] += 100 * weight

    compliance = round(total_num / total_den * 100, 1) if total_den else 0.0
    scored_questions = [q for q in all_questions if not q.get("is_free_text") and q.get("question_id") not in state.skipped_ids]
    # Dynamic, evidence-driven confidence. Historic behaviour clamped the value
    # to the 90-99 band which meant a thin questionnaire and a rich one looked
    # identical. The value is now driven by three real signals:
    #
    # 1) mean per-question confidence (already carried on each question)
    # 2) response coverage (answered / applicable)
    # 3) quantitative depth (proportion of quantitative options actually scored)
    #
    # Downstream consumers can further refine this via the AI Assessment
    # Intelligence service (which produces the full ConfidenceAssessment with
    # sub-scores and a reasoning paragraph).
    avg_conf = sum(float(q.get("confidence", 90)) for q in scored_questions) / max(1, len(scored_questions))
    total_applicable = max(1, answered_count + unanswered_count)
    coverage_ratio = answered_count / total_applicable
    coverage_bonus = coverage_ratio * 10.0
    coverage_penalty = (1.0 - coverage_ratio) * 12.0
    quant_scored = sum(
        1 for q in scored_questions
        if q.get("quantitative_type") and q.get("question_id") in state.responses
    )
    quant_bonus = min(4.0, quant_scored * 0.4)
    raw = avg_conf + coverage_bonus - coverage_penalty + quant_bonus
    eval_conf = round(max(40.0, min(99.0, raw)), 1)

    area_scores = {k: round(area_num[k] / area_den[k] * 100, 1) for k in area_den if area_den[k]}
    func_scores = {k: round(func_num[k] / func_den[k] * 100, 1) for k in func_den if func_den[k]}
    pair_scores = {k: round(pair_num[k] / pair_den[k] * 100, 1) for k in pair_den if pair_den[k]}
    req_scores = {k: round(req_num[k] / req_den[k] * 100, 1) for k in req_den if req_den[k]}

    area_summary: Dict[str, Dict[str, Any]] = {}
    for area, score in area_scores.items():
        status, action = cxo_status(score)
        area_summary[area] = {
            "Compliance %": score,
            "CXO status": status,
            "Questions scored": area_counts[area],
            "Recommended executive action": action,
        }

    function_summary: Dict[str, Dict[str, Any]] = {}
    for fn, score in func_scores.items():
        status, action = cxo_status(score)
        function_summary[fn] = {
            "Compliance %": score,
            "CXO status": status,
            "Questions scored": func_counts[fn],
            "Recommended executive action": action,
        }

    return {
        "compliance_score_pct": compliance,
        "evaluation_confidence_pct": eval_conf,
        "answered_count": answered_count,
        "unanswered_count": unanswered_count,
        "area_scores": area_scores,
        "function_scores": func_scores,
        "pair_scores": pair_scores,
        "requirement_scores": req_scores,
        "area_summary": area_summary,
        "function_summary": function_summary,
    }


# ---------------------------------------------------------------------------
# Rationale text + DataFrame helpers for the UI
# ---------------------------------------------------------------------------

def rationale_text(q: Dict[str, Any], responses: Dict[str, Any]) -> str:
    ids = ", ".join(q.get("mapped_requirement_ids", [])[:5]) or "mapped BRD requirements"
    basis = q.get("regulatory_basis", "mapped regulatory basis")
    kind = question_kind(q).replace("_", " ")
    area = q.get("area", "the impacted area")
    function = q.get("function", "the impacted function")
    if q.get("dynamic"):
        parent = q.get("funnel_parent_id", "the prior question")
        triggers = ", ".join(q.get("trigger_answers", [])) or "the prior response"
        rule = q.get("branch_rule_id", "")
        rule_clause = f" Branch rule: `{rule}`." if rule and rule != "generic_dynamic_followup" else ""
        return (
            f"Your answer to {parent} indicated **{triggers}**. For {area} / {function}, the assessment needs "
            f"one targeted clarification on {kind} before it can score the requirement reliably. "
            f"This is a branch question tied to {ids}.{rule_clause} The answer will either confirm the residual gap, "
            f"identify the accountable remediation route, or provide evidence strong enough to close this branch. "
            f"Regulatory interpretation used: {basis}."
        )
    return (
        f"This is the next highest-value baseline checkpoint for **{area} / {function}** because this pair "
        f"still needs coverage against {ids}. The question tests the {kind} dimension most relevant to the "
        f"mapped regulatory requirement. A strong answer can retire lower-value follow-ups for the same "
        f"cluster; a weak or uncertain answer will trigger only one targeted clarification. "
        f"Regulatory interpretation used: {basis}."
    )


def summary_dataframe(summary: Dict[str, Dict[str, Any]], label: str) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(summary, orient="index").reset_index().rename(columns={"index": label})
    return df.sort_values("Compliance %", ascending=True)


def heatmap_dataframe(area_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    return summary_dataframe(area_summary, "Impacted Area")


def pair_heatmap_rows(pair_scores: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    if not pair_scores:
        return pd.DataFrame()
    functions = sorted({fn for _, fn in pair_scores})
    areas = sorted({area for area, _ in pair_scores})
    rows = []
    for area in areas:
        row = {"Impacted Area": area}
        for fn in functions:
            row[fn] = pair_scores.get((area, fn), None)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "AssessmentState",
    "GAP_SIGNALS",
    "HIGH_RISK",
    "LLMInvoker",
    "MAX_DYNAMIC_FOLLOWUPS_PER_PARENT",
    "MAX_DYNAMIC_FOLLOWUP_DEPTH",
    "MAX_DYNAMIC_QUESTIONS_PER_ASSESSMENT",
    "NEGATIVE_COVERAGE",
    "POSITIVE_SIGNALS",
    "WEAK_EVIDENCE",
    "WEAK_OWNERSHIP",
    "add_to_queue",
    "ai_option_followups",
    "answered",
    "applicable_base_questions",
    "build_followup",
    "choose_next_question",
    "cxo_status",
    "dynamic_depth",
    "dynamic_followups",
    "evaluate",
    "has_overlap",
    "heatmap_dataframe",
    "is_positive_answer",
    "materialize_branch_spec",
    "pair_heatmap_rows",
    "question_kind",
    "rationale_text",
    "registry_followups",
    "response_values",
    "score_value",
    "stable_dynamic_id",
    "summary_dataframe",
    "update_applicability_after_response",
]
