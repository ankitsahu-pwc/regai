"""AI-driven option-aware follow-up generator.

This is the third tier in the v13 adaptive routing chain:

    1. :func:`services.branch_registry.lookup_branch` — static, hand-curated
       branches for the canonical (regulation, theme, kind, answer) tuples.
    2. :func:`generate_option_followups` (this module) — when no static
       branch is registered, ask GenAI to produce 1-3 follow-up questions
       that are **specific to the option the user just selected**.
    3. :func:`services.scoring_engine.dynamic_followups` — final fallback.
       Signal-banded generic follow-ups that work without any knowledge of
       which exact option was selected.

The module is intentionally self-contained:

* It does **not** import the scoring engine or branch registry to avoid
  circular imports.
* It accepts a ``llm_invoker`` callable so callers can pass any wrapper they
  like (the orchestrator passes a function bound to
  :class:`services.genai_service.GenAIClient`). When ``llm_invoker`` is
  ``None`` we use a deterministic offline template that is still
  **option-aware** — selecting "Partially Implemented" yields different
  questions than selecting "Not Implemented", which is the contract the
  scoring engine expects.

Returned shape is the same as ``branch_registry`` specs so the scoring
engine's :func:`materialize_branch_spec` can consume them unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

LLMInvoker = Callable[[str], str]


# ---------------------------------------------------------------------------
# Offline option-aware templates
# ---------------------------------------------------------------------------
#
# Each entry produces TWO follow-up questions. They are deliberately worded
# so the engine surfaces clearly different questions depending on whether the
# user selected Fully / Partially / Not Implemented / Not Applicable /
# Unknown. This is the single biggest behavioural difference from the legacy
# signal-banded ``dynamic_followups`` family.
# ---------------------------------------------------------------------------

_OFFLINE_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "fully implemented": [
        {
            "suffix": "EVIDENCE",
            "question": (
                "What evidence proves the control for '{focus}' is operating effectively "
                "in {area} / {function}?"
            ),
            "options": [
                "Independent test / audit report",
                "Live operational log + management review",
                "Regulatory submission or attestation",
                "Multiple of the above",
                "No documented evidence yet",
            ],
            "rationale": (
                "Substantiates the Fully Implemented claim against {basis} — without it the "
                "score cannot be supported at audit."
            ),
        },
        {
            "suffix": "CADENCE",
            "question": (
                "How often is the control for '{focus}' re-tested or independently reviewed?"
            ),
            "options": [
                "Continuous monitoring",
                "Quarterly",
                "Annually",
                "Less than annually",
                "Not on a defined cadence",
                "Unknown",
            ],
            "rationale": (
                "Even a fully-implemented control degrades without periodic validation against {basis}."
            ),
        },
    ],
    "partially implemented": [
        {
            "suffix": "WHICH-PART",
            "question": (
                "Which part of the control for '{focus}' in {area} / {function} is incomplete today?"
            ),
            "options": [
                "Design / documentation",
                "Operational rollout",
                "Evidence capture",
                "Independent assurance",
                "Reporting to the management body",
                "Unknown",
            ],
            "rationale": (
                "Isolates the specific stage that is partially complete — drives a targeted "
                "remediation against {basis}."
            ),
        },
        {
            "suffix": "GAP-IMPACT",
            "question": (
                "Which teams / processes in {area} are most exposed by the partial state of '{focus}'?"
            ),
            "options": [
                "Critical / important functions only",
                "Critical + supporting functions",
                "All in-scope teams (uniform partial state)",
                "Only a known exclusion list",
                "Unknown",
            ],
            "rationale": (
                "Quantifies the population affected by the partial control so Agent 4 can prioritise "
                "remediation against {basis}."
            ),
        },
    ],
    "not implemented": [
        {
            "suffix": "BLOCKER",
            "question": (
                "Why has implementation of the control for '{focus}' not started in {area} / {function}?"
            ),
            "options": [
                "Awaiting regulatory clarification",
                "No funded programme yet",
                "Awaiting executive sponsorship",
                "Dependency on a third party / vendor",
                "Resource / capability gap",
                "Lower priority than other workstreams",
                "Unknown",
            ],
            "rationale": (
                "Identifies the root blocker so the recommendations engine can pair the gap with "
                "the right unblock action under {basis}."
            ),
        },
        {
            "suffix": "OWNERSHIP",
            "question": (
                "Has accountable ownership been assigned for delivering '{focus}' in {area} / {function}?"
            ),
            "options": [
                "Named accountable owner",
                "Shared ownership",
                "Informal owner",
                "No owner assigned",
                "Unknown",
            ],
            "rationale": (
                "Without a named owner the project will not start; quantifies the governance gap "
                "under {basis}."
            ),
        },
    ],
    "not applicable": [
        {
            "suffix": "JUSTIFY",
            "question": (
                "On what basis is '{focus}' treated as Not Applicable for {area} / {function}?"
            ),
            "options": [
                "Out of scope per regulatory proportionality",
                "Outsourced to a third party with full control transfer",
                "Covered by a parent / group entity",
                "Pending scope decision",
                "Other — see comments",
            ],
            "rationale": (
                "Records why the requirement is scoped out so the assessment remains auditable "
                "against {basis}. Not Applicable answers are excluded from scoring."
            ),
        },
    ],
    "unknown": [
        {
            "suffix": "WHO",
            "question": (
                "Who is best placed to confirm the current state of '{focus}' for {area} / {function}?"
            ),
            "options": [
                "Business / function owner",
                "Technology owner",
                "Risk / Compliance owner",
                "Internal Audit",
                "Vendor management owner",
                "No clear owner — needs to be assigned",
                "Unknown",
            ],
            "rationale": (
                "Routes the assessment to the person who can authoritatively answer the question "
                "before any deeper readiness questions are surfaced for {basis}."
            ),
        },
    ],
}


def _normalise(label: str) -> str:
    return (label or "").strip().lower()


def _focus_phrase(parent: Mapping[str, Any]) -> str:
    explain = parent.get("explainability") or {}
    if explain.get("control_objective"):
        return str(explain["control_objective"])
    basis = str(parent.get("regulatory_basis") or "")
    return basis.split("|", 1)[0].strip() or "the mapped control"


def _build_offline_specs(parent: Mapping[str, Any], answer: str) -> List[Dict[str, Any]]:
    key = _normalise(answer)
    # Accept legacy aliases by collapsing to the canonical key.
    aliases = {
        "complete": "fully implemented",
        "mostly complete": "partially implemented",
        "partially complete": "partially implemented",
        "partially": "partially implemented",
        "not started": "not implemented",
    }
    key = aliases.get(key, key)
    templates = _OFFLINE_TEMPLATES.get(key)
    if not templates:
        return []
    focus = _focus_phrase(parent)
    area = parent.get("area", "the impacted area")
    function = parent.get("function", "the impacted function")
    basis = parent.get("regulatory_basis", "the mapped regulatory clause")
    parent_id = parent.get("question_id", "Q-0000")
    out: List[Dict[str, Any]] = []
    for tpl in templates:
        suffix = tpl["suffix"]
        out.append({
            "question_id": f"AI_{parent_id}_{suffix}",
            "question": tpl["question"].format(focus=focus, area=area, function=function),
            "options": list(tpl["options"]),
            "question_type": "Single Select",
            "rationale": tpl["rationale"].format(basis=basis),
            "branch_rule_id": f"ai_option_followup__{_normalise(answer).replace(' ', '_')}__{suffix.lower()}",
            "scoring_weight": 2,
        })
    return out


# ---------------------------------------------------------------------------
# GenAI integration
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are a regulatory compliance senior manager generating adaptive "
    "questionnaire follow-ups. Given a parent question, the option the user "
    "selected, and the regulatory context, propose 1-3 highly specific "
    "follow-up questions that ONLY make sense for the selected option. "
    "Each question must be:\n"
    "- closed-ended (Single Select),\n"
    "- specific to the regulation/article and business context,\n"
    "- distinct from generic 'evidence/ownership/risk' framing unless the "
    "selected option specifically warrants it,\n"
    "- written in formal business English.\n"
    "\nReturn STRICT JSON of the form:\n"
    "{\"followups\": [{\"question\": str, \"options\": [str], "
    "\"rationale\": str}]}.\n"
    "Do not include any prose outside the JSON."
)


def _build_llm_prompt(parent: Mapping[str, Any], answer: str) -> str:
    explain = parent.get("explainability") or {}
    parts = [
        f"Regulation: {explain.get('regulation') or parent.get('regulatory_basis', 'DORA')}",
        f"Regulator: {explain.get('regulator', '')}",
        f"Article / clause: {explain.get('article') or parent.get('regulatory_basis', '')}",
        f"Theme: {explain.get('theme') or parent.get('branch_theme', '')}",
        f"Control objective: {explain.get('control_objective', '')}",
        f"Impacted area: {parent.get('area', '')}",
        f"Impacted function: {parent.get('function', '')}",
        f"Parent question: {parent.get('question', '')}",
        f"User selected option: {answer}",
        f"Mapped BRD/RTM IDs: {', '.join(parent.get('mapped_requirement_ids') or [])}",
    ]
    return "\n".join(parts)


def _parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate code-fenced JSON ("```json\n{...}\n```").
        stripped = raw.strip().strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    items = payload.get("followups") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        question = str(item.get("question", "")).strip()
        options = item.get("options") or []
        if not question or not isinstance(options, list) or len(options) < 2:
            continue
        cleaned.append({
            "question_id": f"AI_LLM_{idx + 1}",
            "question": question,
            "options": [str(o).strip() for o in options if str(o).strip()],
            "question_type": "Single Select",
            "rationale": str(item.get("rationale", "")).strip()
                          or "Generated by GenAI for the selected option.",
            "branch_rule_id": f"ai_option_followup__llm__{idx + 1}",
            "scoring_weight": 2,
        })
    return cleaned


def generate_option_followups(
    parent: Mapping[str, Any],
    selected_answer: str,
    *,
    llm_invoker: Optional[LLMInvoker] = None,
    max_followups: int = 3,
) -> List[Dict[str, Any]]:
    """Generate option-specific follow-up specs.

    Args:
        parent: the parent question dict (must include ``question_id``,
            ``area``, ``function``, ``regulatory_basis`` and ideally an
            ``explainability`` bundle for richer prompting).
        selected_answer: the exact option label the user chose.
        llm_invoker: optional callable that takes a prompt string and
            returns the raw LLM response text. When omitted (or when it
            raises) we use the offline template family.
        max_followups: hard cap on the number of follow-ups returned.

    Returns:
        List of follow-up specs in the same shape as ``branch_registry``
        entries — ready to be fed into
        :func:`services.scoring_engine.materialize_branch_spec`.
    """
    selected_answer = (selected_answer or "").strip()
    if not selected_answer:
        return []

    llm_results: List[Dict[str, Any]] = []
    if llm_invoker is not None:
        try:
            prompt = (
                _LLM_SYSTEM_PROMPT
                + "\n\nContext:\n"
                + _build_llm_prompt(parent, selected_answer)
            )
            raw = llm_invoker(prompt) or ""
            llm_results = _parse_llm_response(raw)
        except Exception:
            llm_results = []

    if llm_results:
        return llm_results[:max_followups]

    return _build_offline_specs(parent, selected_answer)[:max_followups]


__all__ = [
    "LLMInvoker",
    "generate_option_followups",
]
