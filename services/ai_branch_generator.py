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


# ---------------------------------------------------------------------------
# Structured (Pydantic-schema-enforced) LLM path
# ---------------------------------------------------------------------------
#
# The legacy :func:`generate_option_followups` accepted a generic ``llm_invoker``
# callable that returned raw string text and relied on :func:`_parse_llm_response`
# to hand-parse the JSON. That leaves two failure modes:
#
# 1. The model returns prose, or code-fenced JSON, or JSON with wrong keys —
#    the parser drops the entire response silently.
# 2. Nothing enforces that ``options`` is a real list of strings, that the
#    question is closed-ended, or that no meta-language leaked in.
#
# The structured path below asks LangChain to bind the LLM to a Pydantic
# schema. If the model can't populate the schema, the call raises and we
# fall back to the deterministic offline templates. On success, every text
# field is passed through the full guardrail stack (meta-leakage, citation
# validation, scope validation, speculation / URL / numeric detectors).

try:
    from pydantic import BaseModel, Field

    class _FollowupQuestion(BaseModel):  # type: ignore[misc]
        question: str = Field(
            description=(
                "One closed-ended (Single Select) follow-up question in formal "
                "business English. Must be specific to the option the user "
                "selected — DO NOT reuse generic evidence / ownership / risk "
                "framing unless the selected option specifically warrants it."
            ),
        )
        options: List[str] = Field(
            description=(
                "Two to five closed-ended answer choices. Every option must be "
                "a short, self-contained phrase. Must NOT be free-text."
            ),
            default_factory=list,
        )
        rationale: str = Field(
            description=(
                "One-sentence explanation of why this follow-up is being asked "
                "given the selected option. Grounded in the regulatory basis."
            ),
        )

    class _FollowupPayload(BaseModel):  # type: ignore[misc]
        followups: List[_FollowupQuestion] = Field(
            description=(
                "Between 1 and 3 follow-up specs. Ordered from most to least "
                "important. Each must be distinct — do not repeat questions."
            ),
            default_factory=list,
        )
except ImportError:  # pragma: no cover - pydantic is a project-wide dep
    _FollowupQuestion = None  # type: ignore[assignment]
    _FollowupPayload = None  # type: ignore[assignment]


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


def _followups_from_structured_payload(
    payload: Any, parent: Mapping[str, Any], answer: str,
) -> List[Dict[str, Any]]:
    """Convert a :class:`_FollowupPayload` into branch-registry-shaped dicts.

    Kept private because the shape is stable and only used by
    :func:`generate_option_followups`. We also enforce the same minimum
    invariants the legacy hand-parser enforced (non-empty question,
    ≥ 2 options) so a malformed but schema-valid payload still degrades
    to the deterministic offline templates.
    """
    if payload is None:
        return []
    items = getattr(payload, "followups", None) or []
    parent_id = parent.get("question_id", "Q-0000")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        question = str(getattr(item, "question", "") or "").strip()
        options_raw = list(getattr(item, "options", None) or [])
        options = [str(o).strip() for o in options_raw if str(o).strip()]
        if not question or len(options) < 2:
            continue
        rationale = str(getattr(item, "rationale", "") or "").strip()
        out.append({
            "question_id": f"AI_{parent_id}_{idx + 1}",
            "question": question,
            "options": options,
            "question_type": "Single Select",
            "rationale": rationale,
            "branch_rule_id": (
                f"ai_option_followup__{_normalise(answer).replace(' ', '_')}"
                f"__llm_{idx + 1}"
            ),
            "scoring_weight": 2,
        })
    return out


def generate_option_followups(
    parent: Mapping[str, Any],
    selected_answer: str,
    *,
    llm_invoker: Optional[LLMInvoker] = None,
    client: Optional[Any] = None,
    max_followups: int = 3,
) -> List[Dict[str, Any]]:
    """Generate option-specific follow-up specs.

    Routing priority:

    1. **Structured path** — when ``client`` (a
       :class:`services.genai_service.GenAIClient`) is provided AND
       :class:`_FollowupPayload` was importable, the follow-ups are
       produced through :func:`services.guardrails.safe_generate`. The
       LLM is bound to a Pydantic schema (schema-invalid responses are
       rejected outright) and every text field passes through the full
       anti-hallucination guardrail stack. This is the recommended path.
    2. **Legacy unstructured path** — when only ``llm_invoker`` (a raw
       ``prompt -> str`` callable) is provided, the pre-existing manual
       JSON parser is used. The prompt is still hardened with the
       shared anti-hallucination directive. Kept for backward compat
       with any caller that already wires an ``llm_invoker``.
    3. **Offline path** — when neither is provided (or both fail /
       return nothing), the hand-curated offline templates are used.
       This path is hallucination-free by construction.

    Args:
        parent: the parent question dict (must include ``question_id``,
            ``area``, ``function``, ``regulatory_basis`` and ideally an
            ``explainability`` bundle for richer prompting).
        selected_answer: the exact option label the user chose.
        llm_invoker: optional ``prompt -> str`` callable (legacy path).
        client: optional :class:`GenAIClient` (structured / preferred path).
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

    # ---- Preferred: structured Pydantic-schema call via safe_generate ----
    if client is not None and _FollowupPayload is not None:
        try:
            from .guardrails import safe_generate
            regulation = str(
                (parent.get("explainability") or {}).get("regulation")
                or parent.get("regulatory_basis")
                or ""
            ).strip() or None
            instruction = _LLM_SYSTEM_PROMPT
            context = _build_llm_prompt(parent, selected_answer)
            payload, report = safe_generate(
                client,
                _FollowupPayload,
                f"AI branch follow-ups for {parent.get('question_id', 'Q-0000')}",
                instruction,
                context,
                regulation=regulation,
                # No source corpus for question generation — the LLM is
                # bound to the parent question's context, not to a
                # regulation snippet. Citation-ratio enforcement is
                # accordingly disabled.
                source_corpus="",
                text_fields=("followups",),
                min_citation_ratio=0.0,
            )
            if payload is not None and report.ok:
                llm_results = _followups_from_structured_payload(
                    payload, parent, selected_answer,
                )
        except Exception:
            # Any error → transparent fallback to legacy / offline path.
            llm_results = []

    # ---- Legacy: unstructured invoker (kept for backward compat) ----
    if not llm_results and llm_invoker is not None:
        try:
            try:
                from .guardrails import harden_instruction
                hardened_system = harden_instruction(_LLM_SYSTEM_PROMPT)
            except Exception:  # pragma: no cover
                hardened_system = _LLM_SYSTEM_PROMPT
            prompt = (
                hardened_system
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
