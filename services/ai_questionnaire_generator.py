"""AI-driven adaptive questionnaire generation.

This is the *functional AI-agent* replacement for the deterministic template
pipeline that used to live inside
:mod:`services.questionnaire_generator`. Every question, every answer option,
every score, every follow-up branch and every routing decision is produced by
the LLM from the live regulatory context — nothing is hardcoded.

Inputs the generator considers when producing a questionnaire:

* the selected **regulation** (e.g. DORA, MiFID II),
* the extracted **regulatory obligations** (Agent 1 output),
* the generated **BRD requirements** (Agent 2 output),
* the generated **RTM entries** (Agent 2 output — evidence + functional impact),
* the **impact assessment** (high-impact areas get more depth),
* the **selected client roles** (institution types) so questions are scoped
  to the applicable obligations only,
* the **impact pairs** derived from the BRD (area × function),
* the **client profile keyword bundle** (Page 1 profile so the questions
  reference the client's real business context).

The generator emits questions in the standard package-dict shape validated by
:mod:`utils.json_utils` so the rest of the pipeline (scoring engine, Excel
export, UI cards) stays backwards-compatible. New per-question metadata added
on top of the existing schema:

* ``owning_team`` — Front Office / Back Office / Middle Office / Risk /
  Compliance / Legal / Technology / Operations / Data / Vendor Management /
  Governance / Finance (routing hint for the client).
* ``team_rationale`` — one sentence explaining why the question should go
  to that team.
* ``impact_level`` — Critical / High / Medium / Low.
* ``impact_reason`` — why the question carries that impact.
* ``evidence_expectations`` — list of artefacts the client should produce.
* ``plain_language_explainer`` — plain-English explanation of the regulatory
  term(s) used in the question so business users can answer without a
  legal dictionary.
* ``child_question_ids`` — list of dependent follow-up question IDs.
* ``is_parent`` / ``is_child`` — booleans (derivable, kept for UI clarity).
* ``requires_manual_review`` — True when the generator lacked enough
  context to produce a grounded question and left a placeholder for SME
  input instead of inventing content.

Each answer option is a dict with these fields (all AI-generated per
question — never reused across questions):

* ``label`` — plain-English answer.
* ``score_value`` — numeric readiness score (0-100). ``None`` = excluded.
* ``readiness_interpretation`` — Ready / Watch / At risk / Critical / N/A.
* ``triggers_followup`` — bool. True when this option should surface a
  follow-up question.
* ``followup_question_id`` — the target follow-up question's ID (present
  only when ``triggers_followup`` is True).
* ``option_rationale`` — short explanation of what the option means.

Offline mode
------------
When ``GenAIClient`` is unavailable the generator does **not** fabricate
questionnaire content. It produces a small package of Free-Text
"Manual Review Required" placeholders (one per detected impact pair)
carrying the BRD/regulatory context so an SME can complete them by hand.
This preserves the "no hallucination" contract stated in the product
brief.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from pydantic.v1 import BaseModel, Field, validator
except Exception:  # pragma: no cover - pydantic v2 fallback
    from pydantic import BaseModel, Field, validator  # type: ignore

from services.genai_service import GenAIClient


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Canonical team-routing labels. The LLM is instructed to pick from this
# list so the UI can render a consistent team tag. Extra labels emitted by
# the model are preserved verbatim (never dropped) so we never lose
# regulator-specific routing hints.
CANONICAL_TEAMS: Tuple[str, ...] = (
    "Front Office",
    "Middle Office",
    "Back Office",
    "Risk",
    "Compliance",
    "Legal",
    "Technology",
    "Operations",
    "Data",
    "Vendor Management",
    "Governance",
    "Finance",
    "Cyber Security",
    "Internal Audit",
    "Human Resources",
    "Programme Management",
)

CANONICAL_IMPACT_LEVELS: Tuple[str, ...] = ("Critical", "High", "Medium", "Low")

CANONICAL_READINESS_LABELS: Tuple[str, ...] = (
    "Ready", "Watch", "At risk", "Critical", "N/A",
)

# Canonical question-purpose tags. Every AI question is classified so the
# UI + downstream analytics can guarantee a balanced set of impact-probing
# and readiness-probing questions per impacted area.
CANONICAL_QUESTION_PURPOSES: Tuple[str, ...] = (
    "impact",              # tests what would happen / is affected
    "readiness",           # tests current state / control maturity / evidence
    "impact+readiness",    # tests both simultaneously (e.g. maturity of an incident-response SLA)
)

# Maximum number of impact pairs to seed the AI generator with. This caps
# prompt size and keeps generation predictable.
DEFAULT_MAX_PAIRS = 12
# Number of AI-generated closed-question funnels to request per impact pair,
# split by purpose so every high-impact area gets a guaranteed balance of
# impact-probing AND readiness-probing questions (the manager-requested
# "ask the most-required info to best assess impact AND readiness"
# behaviour). Tuple is ``(impact_count, readiness_count)``.
FUNNELS_PER_PAIR: Dict[str, Tuple[int, int]] = {
    "Critical": (2, 3),   # 2 impact + 3 readiness = 5 parents (deep funnel)
    "High":     (1, 2),   # 1 impact + 2 readiness = 3 parents
    "Medium":   (1, 1),   # 1 impact + 1 readiness = 2 parents
    "Low":      (0, 1),   # readiness only for Low-severity pairs
}
# Free-text SME-narrative prompts requested from the LLM.
DEFAULT_FREE_TEXT_COUNT = 8


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class AIOption(BaseModel):
    """One answer option on an AI-generated question."""

    label: str = Field(
        description=(
            "Plain-English answer the user selects. Options must be relevant "
            "to the specific question — do not reuse a generic option set "
            "across questions."
        )
    )
    score_value: Optional[float] = Field(
        default=None,
        description=(
            "Readiness score contribution for this option, 0-100. "
            "Use 0 for 'not implemented / no evidence / high open risk', "
            "50 for partial / manual / limited coverage, "
            "100 for fully implemented / evidenced / measured. "
            "Use null for 'Not applicable' options that should be excluded from scoring."
        ),
    )
    readiness_interpretation: str = Field(
        default="",
        description=(
            "Executive band: Ready / Watch / At risk / Critical / N/A. "
            "Must be consistent with the numeric score_value: "
            "score>=75 -> Ready, 50-75 -> Watch, 25-50 -> At risk, <25 -> Critical, "
            "None -> N/A."
        ),
    )
    triggers_followup: bool = Field(
        default=False,
        description=(
            "True when selecting this option should reveal a follow-up "
            "question. Only 1-3 options per question should trigger a "
            "follow-up — never every option."
        ),
    )
    followup_question: Optional[str] = Field(
        default=None,
        description=(
            "The exact plain-English follow-up question surfaced when this "
            "option is picked. MUST be directly related to what this option "
            "reveals (e.g. option 'Partially documented across some functions' "
            "-> followup 'Which functions are currently not covered by the "
            "framework?'). Leave null when triggers_followup is False."
        ),
    )
    followup_options: List[str] = Field(
        default_factory=list,
        description=(
            "Answer options for the follow-up question, tailored to what the "
            "follow-up is actually asking. 3-6 options, plain English. "
            "Leave empty when the follow-up is a free-text prompt."
        ),
    )
    followup_is_free_text: bool = Field(
        default=False,
        description=(
            "True when the follow-up asks the user to write a short "
            "narrative answer rather than pick from a list."
        ),
    )
    option_rationale: str = Field(
        default="",
        description=(
            "One sentence explaining what selecting this option implies about "
            "the client's readiness."
        ),
    )


class AIQuestion(BaseModel):
    """One AI-generated closed-ended question."""

    question: str = Field(
        description=(
            "The question in plain, business-friendly English. Avoid "
            "regulatory jargon unless necessary; when it appears, restate it "
            "in plain words in `plain_language_explainer`."
        )
    )
    question_type: str = Field(
        default="Single Select",
        description="Single Select or Multi Select.",
    )
    area: str = Field(
        description=(
            "Impacted business area (e.g. 'ICT Risk Management', "
            "'Incident Reporting', 'Third-Party Risk', 'Front Office')."
        )
    )
    function: str = Field(
        description=(
            "Impacted business function (e.g. 'Risk Management', "
            "'Compliance & Legal', 'Technology / IT Operations')."
        )
    )
    owning_team: str = Field(
        description=(
            "Which team at the client should answer this question. Prefer one "
            "of: Front Office, Middle Office, Back Office, Risk, Compliance, "
            "Legal, Technology, Operations, Data, Vendor Management, "
            "Governance, Finance, Cyber Security, Internal Audit, "
            "Human Resources, Programme Management."
        )
    )
    team_rationale: str = Field(
        default="",
        description=(
            "One sentence explaining why the selected team is best placed to "
            "answer this question (e.g. 'Compliance owns policy approvals')."
        ),
    )
    impact_level: str = Field(
        default="Medium",
        description="Critical / High / Medium / Low.",
    )
    impact_reason: str = Field(
        default="",
        description=(
            "Why the question carries that impact level (e.g. 'Article 17 "
            "notification deadline breach risk is Critical'). Do not repeat "
            "the impact level word — explain it."
        ),
    )
    plain_language_explainer: str = Field(
        default="",
        description=(
            "Plain-English explanation of any regulatory terms in the "
            "question so a business user without regulatory training can "
            "answer it (e.g. 'A resilience test simulates a disruption to "
            "verify recovery time'). Leave empty when the question is "
            "already fully plain."
        ),
    )
    evidence_expectations: List[str] = Field(
        default_factory=list,
        description=(
            "List of concrete artefacts the client should be able to produce "
            "to substantiate a positive answer (e.g. 'RACI matrix', "
            "'Signed policy PDF', 'SIEM dashboard export')."
        ),
    )
    mapped_brd_requirement_ids: List[str] = Field(
        default_factory=list,
        description=(
            "The BRD requirement IDs (e.g. BR-PRO-001) this question tests. "
            "MUST reference IDs that actually appear in the provided BRD "
            "context. Never invent IDs. Leave empty if none apply."
        ),
    )
    mapped_obligation_ids: List[str] = Field(
        default_factory=list,
        description=(
            "The regulatory obligation IDs this question tests. MUST match "
            "IDs from the provided obligations. Never invent IDs. "
            "Leave empty if none apply."
        ),
    )
    regulatory_basis: str = Field(
        default="",
        description=(
            "The specific regulation clause / article this question tests "
            "(e.g. 'DORA Article 17 - Incident Notification'). Use exact "
            "text from the provided obligations when possible."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Short paragraph explaining why this question is being asked "
            "(what regulatory risk it isolates, how it feeds scoring). "
            "This is the 'Why this question?' text surfaced in the UI."
        ),
    )
    options: List[AIOption] = Field(
        default_factory=list,
        description=(
            "Answer options. Provide 3-6 well-differentiated options that "
            "cover the realistic range of readiness for THIS specific "
            "question. Include percentage/frequency/SLA/coverage/maturity "
            "signals where relevant (e.g. 'Fully implemented across "
            "90-100% of applicable processes'). Include one 'Not applicable "
            "with justification' option only when it is realistic for the "
            "question. Never re-use the same option set across questions."
        ),
    )
    confidence: int = Field(
        default=92,
        description=(
            "How confident the AI is that this question is well-grounded "
            "in the provided BRD/obligation context, 0-100."
        ),
    )
    question_purpose: str = Field(
        default="readiness",
        description=(
            "What this question is designed to elicit from the client:\n"
            "  * 'impact'    — probes what would be affected / at risk / broken "
            "if the regulation is not met (systems, data, processes, "
            "stakeholders, SLA exposure, financial exposure). Use for "
            "questions whose primary job is to assess business impact.\n"
            "  * 'readiness' — probes the current state (control implementation, "
            "evidence, maturity, coverage, testing frequency). Use for "
            "questions whose primary job is to assess readiness / preparedness.\n"
            "  * 'impact+readiness' — the question genuinely tests both at "
            "once (e.g. 'How mature is your 24h incident notification workflow?' "
            "tests both the readiness of the workflow AND the impact of a "
            "notification breach). Use sparingly and only when both are "
            "unambiguously tested.\n"
            "Every parent question MUST be classified. The generator uses "
            "this to guarantee a balanced impact/readiness mix per area."
        ),
    )
    targets_impact_dimension: str = Field(
        default="",
        description=(
            "For impact-probing questions, the specific ImpactAssessment "
            "dimension being tested (business_functions / processes / "
            "systems / data / controls / stakeholders). Leave empty for "
            "readiness questions or when no single dimension dominates."
        ),
    )
    targets_readiness_dimension: str = Field(
        default="",
        description=(
            "For readiness-probing questions, the specific "
            "ReadinessAssessment dimension being tested (existing_controls / "
            "process_maturity / policy_coverage / technology_readiness / "
            "documentation_completeness / implementation_gaps / "
            "organizational_preparedness). Leave empty for impact questions "
            "or when no single dimension dominates."
        ),
    )

    @validator("question_type", pre=True, always=True, allow_reuse=True)
    def _normalise_qtype(cls, value):  # noqa: N805
        v = str(value or "").strip().lower()
        if "multi" in v:
            return "Multi Select"
        if "free" in v or "open" in v or "text" in v:
            return "Open Ended"
        return "Single Select"

    @validator("impact_level", pre=True, always=True, allow_reuse=True)
    def _normalise_impact(cls, value):  # noqa: N805
        v = str(value or "").strip().title()
        for canon in CANONICAL_IMPACT_LEVELS:
            if canon.lower() == v.lower():
                return canon
        return "Medium"

    @validator("owning_team", pre=True, always=True, allow_reuse=True)
    def _normalise_team(cls, value):  # noqa: N805
        v = str(value or "").strip()
        if not v:
            return "Compliance"
        # Case-insensitive canonical match; otherwise pass through so
        # regulator-specific team labels (e.g. "Model Risk") survive.
        for canon in CANONICAL_TEAMS:
            if canon.lower() == v.lower():
                return canon
        return v

    @validator("question_purpose", pre=True, always=True, allow_reuse=True)
    def _normalise_purpose(cls, value):  # noqa: N805
        v = str(value or "").strip().lower().replace(" ", "").replace("_", "+")
        for canon in CANONICAL_QUESTION_PURPOSES:
            if canon.lower() == v:
                return canon
        # Common near-misses.
        if "impact" in v and "ready" in v:
            return "impact+readiness"
        if "impact" in v:
            return "impact"
        return "readiness"


class AIFreeTextQuestion(BaseModel):
    """One AI-generated free-text SME narrative question."""

    question: str = Field(
        description=(
            "Plain-English narrative prompt the SME answers in free text. "
            "Reference the specific BRD requirement(s) or regulatory theme."
        )
    )
    area: str = Field(default="Free Text / SME Narrative")
    function: str = Field(default="Cross-functional")
    owning_team: str = Field(default="Compliance")
    impact_level: str = Field(default="Medium")
    mapped_brd_requirement_ids: List[str] = Field(default_factory=list)
    mapped_obligation_ids: List[str] = Field(default_factory=list)
    regulatory_basis: str = Field(default="")
    rationale: str = Field(default="")
    plain_language_explainer: str = Field(default="")
    evidence_expectations: List[str] = Field(default_factory=list)
    confidence: int = Field(default=90)
    question_purpose: str = Field(
        default="impact+readiness",
        description=(
            "Cross-cutting SME narratives usually probe both impact and "
            "readiness, so this defaults to 'impact+readiness'. Override "
            "with 'impact' or 'readiness' when the prompt is single-purpose."
        ),
    )

    @validator("question_purpose", pre=True, always=True, allow_reuse=True)
    def _normalise_purpose_ft(cls, value):  # noqa: N805
        v = str(value or "").strip().lower().replace(" ", "").replace("_", "+")
        for canon in CANONICAL_QUESTION_PURPOSES:
            if canon.lower() == v:
                return canon
        if "impact" in v and "ready" in v:
            return "impact+readiness"
        if "impact" in v:
            return "impact"
        return "readiness"


class AIQuestionBank(BaseModel):
    """Full LLM response: closed + free-text questions for one impact pair."""

    closed_questions: List[AIQuestion] = Field(default_factory=list)
    free_text_questions: List[AIFreeTextQuestion] = Field(default_factory=list)


class AIFreeTextBank(BaseModel):
    """Cross-cutting free-text SME narrative questions."""

    questions: List[AIFreeTextQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _requirement_snippet(req: Any) -> str:
    """Compact one-line summary of a BRD Requirement dataclass."""
    parts = [
        f"[{getattr(req, 'normalized_id', '')}]",
        (getattr(req, 'requirement', '') or '').strip(),
    ]
    detail = (getattr(req, 'detail', '') or '').strip()
    if detail:
        parts.append(f"— {detail[:220]}")
    priority = (getattr(req, 'priority', '') or '').strip()
    if priority:
        parts.append(f"(priority: {priority})")
    alignment = (getattr(req, 'alignment', '') or '').strip()
    if alignment:
        parts.append(f"[basis: {alignment[:120]}]")
    return " ".join(parts)


def _obligation_snippet(obl: Any) -> str:
    ctrl = ", ".join((getattr(obl, 'control_expectations', None) or [])[:3])
    evid = ", ".join((getattr(obl, 'evidence_needs', None) or [])[:3])
    line = (
        f"[{getattr(obl, 'obligation_id', '')}] {getattr(obl, 'title', '')}"
        f" — theme: {getattr(obl, 'theme', '')}"
        f" | area: {getattr(obl, 'impacted_area', '')}"
        f" / fn: {getattr(obl, 'impacted_function', '')}"
        f" | basis: {(getattr(obl, 'regulatory_basis', '') or '')[:120]}"
    )
    if ctrl:
        line += f" | controls: {ctrl}"
    if evid:
        line += f" | evidence: {evid}"
    reqid = getattr(obl, 'source_requirement_id', '') or ''
    if reqid:
        line += f" | source_req: {reqid}"
    return line


def _rtm_snippet(entry: Any) -> str:
    return (
        f"[{getattr(entry, 'traceability_id', '')}] "
        f"BR: {getattr(entry, 'business_requirement_id', '')} — "
        f"{(getattr(entry, 'business_requirement', '') or '')[:140]}"
        f" | evidence: {(getattr(entry, 'evidence_required', '') or '')[:120]}"
        f" | function: {getattr(entry, 'impacted_function', '')}"
    )


def _text_matches_area(area: str, *fields: str) -> bool:
    """Heuristic keyword-overlap match between an impacted area and text."""
    area_l = (area or "").lower().strip()
    if not area_l:
        return False
    for text in fields:
        t = (text or "").lower()
        if not t:
            continue
        if area_l in t or any(tok in t for tok in area_l.split() if len(tok) > 3):
            return True
    return False


def _impact_summary_for_area(impact: Any, area: str) -> str:
    """Render the ImpactAssessment context that is specifically relevant to
    the impacted area currently being questioned.

    This is the "ask the most-required info to best assess impact" input
    the manager asked for. The LLM sees, per pair, the dominant impact
    dimension(s), the affected items (systems, data, stakeholders,
    processes, ...), the severity score, and the rationale — so it can
    ground each impact-probing question in a concrete item or a concrete
    consequence rather than a generic "impact question".
    """
    if impact is None:
        return "(no impact assessment supplied for this area)"
    lines: List[str] = []
    overall = getattr(impact, "overall_severity", "") or ""
    if overall:
        lines.append(f"Overall regulation severity: {overall}")
    exec_summary = getattr(impact, "executive_summary", "") or ""
    if exec_summary:
        lines.append(f"Executive summary: {exec_summary[:300]}")

    dims = getattr(impact, "dimensions", lambda: [])() or []
    ranked_dims: List[Tuple[Any, bool]] = []
    for dim in dims:
        items = list(getattr(dim, "items", None) or [])
        rationale = getattr(dim, "rationale", "") or ""
        # A dimension is "area-specific" if any of its items or its
        # rationale mentions the area label.
        area_specific = _text_matches_area(area, *items, rationale)
        ranked_dims.append((dim, area_specific))
    # Area-specific dimensions first, then the rest.
    ranked_dims.sort(key=lambda t: (0 if t[1] else 1))

    for dim, is_specific in ranked_dims[:6]:
        items = list(getattr(dim, "items", None) or [])[:6]
        rationale = (getattr(dim, "rationale", "") or "").strip()
        severity = getattr(dim, "severity", "Medium")
        sev_score = getattr(dim, "severity_score", None)
        tag = " (area-specific)" if is_specific else ""
        head = (
            f"- {getattr(dim, 'dimension', '')} [{severity}"
            f"{f' / {sev_score:.0f}' if isinstance(sev_score, (int, float)) else ''}]{tag}"
        )
        if items:
            head += f" — items: {', '.join(str(x) for x in items)}"
        lines.append(head)
        if rationale:
            lines.append(f"    rationale: {rationale[:220]}")
    return "\n".join(lines) or "(impact assessment carried no dimension data)"


def _readiness_summary_for_area(readiness: Any, area: str) -> str:
    """Render the ReadinessAssessment context that is specifically relevant
    to the impacted area currently being questioned.

    This is the "ask the most-required info to best assess readiness"
    input. The LLM sees the current maturity + gaps + strengths for
    each readiness lens (existing_controls, process_maturity,
    policy_coverage, technology_readiness, documentation_completeness,
    implementation_gaps, organizational_preparedness) so it can craft
    readiness questions that actually target the client's *weakest*
    dimensions rather than asking a generic "is your control
    documented?" for every area.
    """
    if readiness is None:
        return "(no readiness assessment supplied for this area)"
    lines: List[str] = []
    overall_level = getattr(readiness, "overall_level", "") or ""
    overall_score = getattr(readiness, "overall_score", None)
    if overall_level:
        head = f"Overall readiness: {overall_level}"
        if isinstance(overall_score, (int, float)):
            head += f" ({overall_score:.0f}/100)"
        lines.append(head)
    exec_summary = getattr(readiness, "executive_summary", "") or ""
    if exec_summary:
        lines.append(f"Executive summary: {exec_summary[:300]}")
    key_gaps = list(getattr(readiness, "key_gaps", None) or [])[:5]
    if key_gaps:
        lines.append("Top readiness gaps: " + "; ".join(str(g) for g in key_gaps))
    key_strengths = list(getattr(readiness, "key_strengths", None) or [])[:5]
    if key_strengths:
        lines.append("Existing strengths: " + "; ".join(str(s) for s in key_strengths))

    dims = getattr(readiness, "dimensions", lambda: [])() or []
    ranked: List[Tuple[Any, bool]] = []
    for dim in dims:
        gaps = list(getattr(dim, "gaps", None) or [])
        strengths = list(getattr(dim, "strengths", None) or [])
        rationale = getattr(dim, "rationale", "") or ""
        area_specific = _text_matches_area(area, *gaps, *strengths, rationale)
        ranked.append((dim, area_specific))
    ranked.sort(key=lambda t: (0 if t[1] else 1))

    for dim, is_specific in ranked[:7]:
        gaps = list(getattr(dim, "gaps", None) or [])[:4]
        strengths = list(getattr(dim, "strengths", None) or [])[:3]
        rationale = (getattr(dim, "rationale", "") or "").strip()
        maturity = getattr(dim, "maturity_level", "Developing")
        score = getattr(dim, "score", None)
        tag = " (area-specific)" if is_specific else ""
        head = (
            f"- {getattr(dim, 'dimension', '')} [{maturity}"
            f"{f' / {score:.0f}' if isinstance(score, (int, float)) else ''}]{tag}"
        )
        lines.append(head)
        if gaps:
            lines.append(f"    gaps: {'; '.join(str(g) for g in gaps)}")
        if strengths:
            lines.append(f"    strengths: {'; '.join(str(s) for s in strengths)}")
        if rationale:
            lines.append(f"    rationale: {rationale[:200]}")
    return "\n".join(lines) or "(readiness assessment carried no dimension data)"


def _prioritise_pairs(
    pairs: Sequence[Any], impact: Any, max_pairs: int,
) -> List[Tuple[Any, str]]:
    """Return ``[(pair, severity)]`` sorted by severity DESC then confidence DESC."""
    from services.questionnaire_enhancer import _area_severity, _severity_rank

    ranked: List[Tuple[Any, str, int, int]] = []
    for pair in pairs:
        area = getattr(pair, "area", "") or ""
        severity = _area_severity(area, impact, {})
        rank = _severity_rank(severity)
        conf = int(getattr(pair, "confidence", 90) or 90)
        ranked.append((pair, severity, rank, conf))
    ranked.sort(key=lambda t: (-t[2], -t[3]))
    return [(p, s) for (p, s, _, _) in ranked[:max_pairs]]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = (
    "You are a principal regulatory readiness architect and senior business "
    "analyst. You are generating a client-facing regulatory readiness "
    "questionnaire for a financial services institution.\n\n"
    "Absolute rules:\n"
    "1. Every question, answer option, follow-up and score MUST be grounded in "
    "the provided BRD requirements, regulatory obligations, RTM entries, "
    "impact assessment and readiness assessment. Never invent BRD IDs, "
    "obligation IDs, articles or regulatory clauses.\n"
    "2. Options must be DIFFERENT for every question. Do not reuse a generic "
    "'Yes / Partially / No / Unknown' set across questions. Include "
    "quantitative anchors where relevant (percentages, frequency, SLA, "
    "coverage, maturity level, evidence availability, control status).\n"
    "3. Questions must be in plain, business-friendly English. When a "
    "regulatory term is unavoidable, restate it in plain words in "
    "`plain_language_explainer`.\n"
    "4. Adaptive branching is MANDATORY: every closed question MUST mark at "
    "least ONE option with `triggers_followup: true` and provide a "
    "`followup_question` + `followup_options` (or `followup_is_free_text: "
    "true`) that are DIRECTLY related to what that option revealed. When "
    "an option indicates a healthy state, the follow-up should probe "
    "evidence / assurance. When an option indicates a weak state, the "
    "follow-up should probe the specific gap. No closed question may be "
    "emitted without at least one follow-up.\n"
    "5. Route every question to a specific business team via `owning_team` "
    "(Front / Middle / Back Office, Risk, Compliance, Legal, Technology, "
    "Operations, Data, Vendor Management, Governance, Finance, Cyber Security, "
    "Internal Audit, Human Resources, Programme Management).\n"
    "6. Score each option realistically. 0 = not implemented / no evidence; "
    "50 = partial / manual / limited coverage; 100 = fully implemented / "
    "evidenced / measured. Use null (Not applicable) sparingly.\n"
    "7. For high-impact areas, prefer 3+ options and a deeper follow-up "
    "tree. For low-impact areas, 3-4 options with at most one follow-up.\n"
    "8. If context is thin, produce fewer questions rather than fabricating "
    "content.\n\n"
    "IMPACT vs READINESS BALANCE — the questionnaire's job is to elicit "
    "the most-required information to assess BOTH the business impact of "
    "the regulation AND the client's current readiness to meet it. For "
    "every impacted area you MUST:\n"
    "9.  Classify every question via `question_purpose`:\n"
    "     - 'impact'    -> tests what would be affected / broken / exposed "
    "if the client is not compliant (systems, data, processes, "
    "stakeholders, financial or reputational exposure, SLA breach risk).\n"
    "     - 'readiness' -> tests the client's current state (control "
    "implementation, evidence, maturity, coverage, testing frequency, "
    "documentation, ownership).\n"
    "     - 'impact+readiness' -> genuinely tests both (use sparingly).\n"
    "10. Deliver the exact impact/readiness split requested by the caller "
    "(see 'Requested purpose mix' in the task section). Do NOT collapse "
    "impact questions into readiness questions.\n"
    "11. For every IMPACT-purpose question, anchor it in a specific item "
    "from the ImpactAssessment (a named system, process, data set, "
    "stakeholder group, or business function) whenever such an item is "
    "supplied. Set `targets_impact_dimension` to the dimension the item "
    "came from (business_functions / processes / systems / data / "
    "controls / stakeholders). Ask about the CONSEQUENCE — SLA/notification "
    "breach risk, downstream process impact, data exposure, stakeholder "
    "harm, financial or reputational exposure — not the control.\n"
    "12. For every READINESS-purpose question, anchor it in a specific "
    "ReadinessAssessment dimension (existing_controls / process_maturity / "
    "policy_coverage / technology_readiness / documentation_completeness / "
    "implementation_gaps / organizational_preparedness). Preferentially "
    "target the client's WEAKEST dimensions and known gaps supplied in "
    "the readiness assessment. Set `targets_readiness_dimension`.\n"
    "13. Never duplicate an impact question with a near-identical readiness "
    "question — each parent must add distinct diagnostic value."
)


def _funnel_prompt(
    *,
    regulation: str,
    client_roles: Sequence[str],
    client_profile_lines: Sequence[str],
    pair: Any,
    pair_severity: str,
    mapped_requirements: Sequence[Any],
    mapped_obligations: Sequence[Any],
    mapped_rtm: Sequence[Any],
    impact_summary: str,
    readiness_summary: str,
    impact_count: int,
    readiness_count: int,
) -> str:
    """Prompt asking the LLM to generate closed questions for one pair.

    Injects the ImpactAssessment slice + ReadinessAssessment slice
    relevant to this pair so the LLM can craft questions that specifically
    target the affected items / dimensions, and asks for a balanced
    impact-vs-readiness split.
    """
    lines: List[str] = []
    lines.append(f"Regulation: {regulation}")
    if client_roles:
        lines.append(
            "Client institution type(s) IN SCOPE: " + ", ".join(client_roles)
        )
    if client_profile_lines:
        lines.append("Client profile signals (business lines, products, "
                     "geographies, legal entities, vendor / third parties):")
        lines.extend(f"  - {ln}" for ln in client_profile_lines)
    if client_roles or client_profile_lines:
        lines.append("")
        lines.append("== HARD SCOPING RULES (must be respected) ==")
        if client_roles:
            lines.append(
                "1. Every question must be answerable by the specific role "
                "combination listed above. If a mapped BRD requirement or "
                "obligation clearly does NOT apply to any of those roles, "
                "SKIP it - do not invent applicability."
            )
            lines.append(
                "2. Frame every question, option and rationale in terms of "
                "the operating model of the selected role(s). Do not use "
                "generic banking language when a role-specific phrasing is "
                "possible (e.g. use 'trading desk' for a broker-dealer, "
                "'core banking' for a retail bank, 'payment rails' for a "
                "PSP)."
            )
        if client_profile_lines:
            lines.append(
                "3. Cross-check every question against the client profile "
                "signals above. If a profile keyword flags a specific "
                "product (e.g. OTC Derivatives), geography (e.g. EU), or "
                "third-party dependency (e.g. Cloud Service Provider) that "
                "is relevant to the pair, target that keyword directly in "
                "the question text or its answer options."
            )
            lines.append(
                "4. If the selected profile clearly excludes a dimension "
                "(e.g. no vendor keywords -> no third-party dependencies), "
                "do NOT generate questions probing that dimension; those "
                "would be non-applicable noise."
            )
        lines.append(
            "5. Set `mapped_brd_requirement_ids` and "
            "`mapped_obligation_ids` only to IDs that ACTUALLY appear in "
            "the lists below AND are in scope for the selected role(s)."
        )
        lines.append("")
    lines.append(
        f"Impact pair: {pair.area} / {pair.function}"
        f" — severity: {pair_severity}"
    )
    if getattr(pair, "regulatory_basis", ""):
        lines.append(f"Regulatory basis (pair-level): {pair.regulatory_basis}")
    lines.append("")

    lines.append("== IMPACT context (for impact-purpose questions) ==")
    lines.append(impact_summary)
    lines.append("")
    lines.append("== READINESS context (for readiness-purpose questions) ==")
    lines.append(readiness_summary)
    lines.append("")

    lines.append("Mapped BRD requirements for this pair:")
    for req in mapped_requirements[:12]:
        lines.append(f"  - {_requirement_snippet(req)}")
    if not mapped_requirements:
        lines.append("  (no explicit BRD rows for this pair; use obligations only)")
    lines.append("")
    if mapped_obligations:
        lines.append("Regulatory obligations covering this pair:")
        for obl in mapped_obligations[:8]:
            lines.append(f"  - {_obligation_snippet(obl)}")
        lines.append("")
    if mapped_rtm:
        lines.append("RTM entries (evidence + traceability) for this pair:")
        for entry in mapped_rtm[:8]:
            lines.append(f"  - {_rtm_snippet(entry)}")
        lines.append("")

    total = impact_count + readiness_count
    lines.append(
        f"Task: Generate exactly {total} parent CLOSED questions "
        f"for the {pair.area} / {pair.function} pair with the following "
        f"REQUIRED purpose mix:"
    )
    lines.append(
        f"  - {impact_count} question(s) with question_purpose='impact' — each "
        "must anchor on a specific item from the IMPACT context above "
        "(named system, process, data set, stakeholder, or business "
        "function) and ask about the CONSEQUENCE of non-compliance "
        "(SLA / notification breach exposure, downstream process impact, "
        "data exposure, stakeholder harm, financial or reputational risk). "
        "Set `targets_impact_dimension` accordingly."
    )
    lines.append(
        f"  - {readiness_count} question(s) with question_purpose='readiness' — "
        "each must target the client's WEAKEST readiness dimension for "
        "this area (from the READINESS context above), reference a "
        "specific known gap where possible, and probe the current state "
        "(control implementation, evidence, maturity, coverage, testing "
        "frequency, documentation, ownership). Set "
        "`targets_readiness_dimension` accordingly."
    )
    if impact_count == 0:
        lines.append(
            "  (No impact questions requested for this Low-severity pair — "
            "focus purely on readiness.)"
        )
    lines.append(
        f"For a {pair_severity}-severity pair the questions must go deeper: "
        "each parent should offer 4-6 well-differentiated options and 2-3 of "
        "them should carry an option-specific follow-up. Use quantitative "
        "options (percentage / frequency / SLA / coverage / maturity / "
        "evidence availability / implementation status) wherever relevant."
    )
    lines.append(
        "Route each question to the correct owning_team. Prefer the team "
        "closest to the mapped obligation and function."
    )
    lines.append(
        "For each question, set mapped_brd_requirement_ids and "
        "mapped_obligation_ids to IDs that ACTUALLY appear in the lists "
        "above. Never invent IDs. Never duplicate impact + readiness "
        "questions that ask the same thing in different words."
    )
    lines.append(
        "Do not generate free-text questions in this call — closed questions "
        "only."
    )
    return "\n".join(lines)


def _free_text_prompt(
    *,
    regulation: str,
    client_roles: Sequence[str],
    top_pairs: Sequence[Tuple[Any, str]],
    obligations: Sequence[Any],
    requested_count: int,
) -> str:
    lines: List[str] = []
    lines.append(f"Regulation: {regulation}")
    if client_roles:
        lines.append(
            "Client institution type(s) IN SCOPE: " + ", ".join(client_roles)
        )
        lines.append(
            "Every free-text question must probe qualitative evidence "
            "that specifically applies to the operating model of those "
            "role(s). If a topic does not apply to any of the selected "
            "roles, do NOT generate a question about it - drop it and "
            "reallocate the slot to a more in-scope theme."
        )
    lines.append("")
    lines.append("Top impact pairs (in priority order):")
    for pair, sev in top_pairs[:8]:
        lines.append(
            f"  - {pair.area} / {pair.function} [{sev}] — "
            f"basis: {(getattr(pair, 'regulatory_basis', '') or '')[:120]}"
        )
    if obligations:
        lines.append("")
        lines.append("Sample regulatory obligations:")
        for obl in obligations[:6]:
            lines.append(f"  - {_obligation_snippet(obl)}")
    lines.append("")
    lines.append(
        f"Task: Generate {requested_count} plain-English SME narrative "
        "(free-text) questions that capture qualitative evidence which the "
        "closed questions cannot score. Each free-text question must "
        "reference REAL BRD IDs or obligation IDs from the context above. "
        "Route each to the most relevant owning_team."
    )
    lines.append(
        "Avoid duplication with the closed questions — focus on gaps, "
        "assumptions, dependencies, sponsorship, exceptions, and lessons "
        "learned that require narrative context."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score / interpretation normalisation
# ---------------------------------------------------------------------------


def _clamp_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return 0.0
    if v > 100:
        return 100.0
    return v


def _readiness_from_score(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score >= 75:
        return "Ready"
    if score >= 50:
        return "Watch"
    if score >= 25:
        return "At risk"
    return "Critical"


def _normalise_readiness_label(raw: str, score: Optional[float]) -> str:
    v = (raw or "").strip()
    for canon in CANONICAL_READINESS_LABELS:
        if canon.lower() == v.lower():
            return canon
    return _readiness_from_score(score)


# ---------------------------------------------------------------------------
# Materialisation: AIQuestion -> package question dicts (with follow-ups)
# ---------------------------------------------------------------------------


def _build_explainability(
    *,
    regulation: str,
    ai_q: AIQuestion,
    pair: Any,
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    rtm_ids: Sequence[str],
    obligations_by_id: Mapping[str, Any],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Assemble the explainability bundle with team routing + impact + evidence."""
    mapped_brd_ids = list(dict.fromkeys(
        [b for b in ai_q.mapped_brd_requirement_ids if b in set(all_brd_ids)]
    ))
    mapped_obl_ids = list(dict.fromkeys(
        [o for o in ai_q.mapped_obligation_ids if o in set(all_obligation_ids)]
    ))
    obligation_id = mapped_obl_ids[0] if mapped_obl_ids else ""
    obligation = obligations_by_id.get(obligation_id) if obligation_id else None
    article = ai_q.regulatory_basis
    if not article and obligation is not None:
        article = getattr(obligation, "regulatory_basis", "") or ""
    regulator = ""
    if obligation is not None:
        regulator = str(getattr(obligation, "regulatory_basis", "") or "")

    source_refs: List[Dict[str, Any]] = []
    for bid in mapped_brd_ids:
        refs = source_refs_by_brd.get(f"REQ:{bid}") or source_refs_by_brd.get(bid) or []
        for ref in refs:
            if isinstance(ref, dict) and ref not in source_refs:
                source_refs.append(ref)

    return {
        "regulation": regulation,
        "regulator": regulator,
        "article": article or f"{regulation} mapped clause",
        "obligation_id": obligation_id,
        "brd_requirement_ids": mapped_brd_ids,
        "rtm_trace_ids": list(rtm_ids),
        "business_function": ai_q.function or getattr(pair, "function", ""),
        "business_area": ai_q.area or getattr(pair, "area", ""),
        "control_objective": ai_q.rationale[:140] or (
            f"Control objective for {ai_q.area} / {ai_q.function}."
        ),
        "theme": ai_q.area or getattr(pair, "area", ""),
        "reason": ai_q.rationale or (
            f"AI-generated question testing readiness against "
            f"{article or regulation}."
        ),
        "expected_evidence": ", ".join(ai_q.evidence_expectations) if ai_q.evidence_expectations else "",
        "risk_if_negative": ai_q.impact_reason or (
            f"Weak coverage of this control weakens readiness against "
            f"{article or regulation}."
        ),
        "source_references": source_refs,
        # New enrichment fields
        "owning_team": ai_q.owning_team,
        "team_rationale": ai_q.team_rationale,
        "impact_level": ai_q.impact_level,
        "impact_reason": ai_q.impact_reason,
        "plain_language_explainer": ai_q.plain_language_explainer,
        "evidence_expectations": list(ai_q.evidence_expectations),
    }


def _materialise_parent_question(
    ai_q: AIQuestion,
    *,
    question_id: str,
    child_ids: Sequence[str],
    regulation: str,
    pair: Any,
    obligations_by_id: Mapping[str, Any],
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    rtm_ids: Sequence[str],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
    impact_weight: int,
) -> Dict[str, Any]:
    """Convert an AIQuestion into the package question dict format."""
    options_out: List[Dict[str, Any]] = []
    child_idx = 0
    for opt in ai_q.options:
        score_v = _clamp_score(opt.score_value)
        readiness = _normalise_readiness_label(opt.readiness_interpretation, score_v)
        opt_dict: Dict[str, Any] = {
            "label": opt.label,
            "score_value": score_v,
            "readiness_interpretation": readiness,
            "triggers_followup": bool(opt.triggers_followup and opt.followup_question),
            "option_rationale": opt.option_rationale,
        }
        if opt_dict["triggers_followup"] and child_idx < len(child_ids):
            opt_dict["followup_question_id"] = child_ids[child_idx]
            child_idx += 1
        options_out.append(opt_dict)

    explain = _build_explainability(
        regulation=regulation,
        ai_q=ai_q,
        pair=pair,
        all_brd_ids=all_brd_ids,
        all_obligation_ids=all_obligation_ids,
        rtm_ids=rtm_ids,
        obligations_by_id=obligations_by_id,
        source_refs_by_brd=source_refs_by_brd,
    )
    mapped_brd = list(explain["brd_requirement_ids"])

    # Persist the impact/readiness classification on both the top-level
    # dict (for cheap UI access) and inside the explainability bundle
    # (for Excel export + downstream analytics).
    purpose = str(getattr(ai_q, "question_purpose", "") or "readiness")
    targets_impact = str(getattr(ai_q, "targets_impact_dimension", "") or "")
    targets_readiness = str(getattr(ai_q, "targets_readiness_dimension", "") or "")
    explain["question_purpose"] = purpose
    if targets_impact:
        explain["targets_impact_dimension"] = targets_impact
    if targets_readiness:
        explain["targets_readiness_dimension"] = targets_readiness

    return {
        "question_id": question_id,
        "area": ai_q.area,
        "function": ai_q.function,
        "question_type": ai_q.question_type,
        "question": ai_q.question,
        "options": options_out,
        "mapped_requirement_ids": mapped_brd,
        "mapped_obligation_ids": [
            o for o in dict.fromkeys(ai_q.mapped_obligation_ids)
            if o in set(all_obligation_ids)
        ],
        "regulatory_basis": explain["article"],
        "confidence": max(0, min(100, int(ai_q.confidence or 90))),
        "scoring_weight": impact_weight,
        "impact_weight": impact_weight,
        "impact_severity": ai_q.impact_level,
        "impact_level": ai_q.impact_level,
        "impact_reason": ai_q.impact_reason,
        "owning_team": ai_q.owning_team,
        "team_rationale": ai_q.team_rationale,
        "evidence_expectations": list(ai_q.evidence_expectations),
        "plain_language_explainer": ai_q.plain_language_explainer,
        "question_purpose": purpose,
        "targets_impact_dimension": targets_impact,
        "targets_readiness_dimension": targets_readiness,
        "funnel_parent_id": "",
        "trigger_answers": [],
        "rationale": ai_q.rationale,
        "is_free_text": False,
        "is_parent": bool(child_ids),
        "is_child": False,
        "child_question_ids": list(child_ids),
        "branch_theme": ai_q.area,
        "branch_rule_id": "",
        "source_parent_id": "",
        "dynamic_depth": 0,
        "requires_manual_review": False,
        "generated_by_ai": True,
        "explainability": explain,
    }


def _materialise_child_question(
    *,
    parent_ai_q: AIQuestion,
    parent_qid: str,
    child_qid: str,
    triggering_option_label: str,
    followup_question: str,
    followup_options: Sequence[str],
    followup_is_free_text: bool,
    regulation: str,
    pair: Any,
    obligations_by_id: Mapping[str, Any],
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    rtm_ids: Sequence[str],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
    impact_weight: int,
) -> Dict[str, Any]:
    """Materialise an option-triggered follow-up as a child question dict."""
    # Child inherits the parent's explainability but with an override on
    # the "reason" so the UI can quote the triggering option.
    parent_explain = _build_explainability(
        regulation=regulation,
        ai_q=parent_ai_q,
        pair=pair,
        all_brd_ids=all_brd_ids,
        all_obligation_ids=all_obligation_ids,
        rtm_ids=rtm_ids,
        obligations_by_id=obligations_by_id,
        source_refs_by_brd=source_refs_by_brd,
    )
    child_explain = dict(parent_explain)
    child_explain["reason"] = (
        f"Follow-up question triggered because the previous answer was "
        f"'{triggering_option_label}'. {followup_question}"
    )
    parent_purpose = str(getattr(parent_ai_q, "question_purpose", "") or "readiness")
    child_explain["question_purpose"] = parent_purpose

    options_out: List[Dict[str, Any]] = []
    if not followup_is_free_text:
        for label in followup_options:
            options_out.append({
                "label": label,
                "score_value": None,  # AI didn't score follow-up opts; scoring engine will treat as unscored unless legacy label matches
                "readiness_interpretation": "",
                "triggers_followup": False,
                "option_rationale": "",
            })

    return {
        "question_id": child_qid,
        "area": parent_ai_q.area,
        "function": parent_ai_q.function,
        "question_type": "Open Ended" if followup_is_free_text else (
            "Single Select" if len(followup_options) <= 6 else "Multi Select"
        ),
        "question": followup_question,
        "options": options_out if not followup_is_free_text else ["Free text response"],
        "mapped_requirement_ids": list(parent_explain["brd_requirement_ids"]),
        "regulatory_basis": parent_explain["article"],
        "confidence": max(0, min(100, int(parent_ai_q.confidence or 90))),
        "scoring_weight": max(1, impact_weight - 1),
        "impact_weight": max(1, impact_weight - 1),
        "impact_severity": parent_ai_q.impact_level,
        "impact_level": parent_ai_q.impact_level,
        "impact_reason": parent_ai_q.impact_reason,
        "owning_team": parent_ai_q.owning_team,
        "team_rationale": parent_ai_q.team_rationale,
        "evidence_expectations": list(parent_ai_q.evidence_expectations),
        "plain_language_explainer": parent_ai_q.plain_language_explainer,
        "question_purpose": parent_purpose,
        "targets_impact_dimension": str(getattr(parent_ai_q, "targets_impact_dimension", "") or ""),
        "targets_readiness_dimension": str(getattr(parent_ai_q, "targets_readiness_dimension", "") or ""),
        "funnel_parent_id": parent_qid,
        "trigger_answers": [triggering_option_label],
        "rationale": (
            f"Adaptive follow-up. Triggered when the user answered "
            f"'{triggering_option_label}' to question {parent_qid}."
        ),
        "is_free_text": bool(followup_is_free_text),
        "is_parent": False,
        "is_child": True,
        "child_question_ids": [],
        "branch_theme": parent_ai_q.area,
        "branch_rule_id": f"ai_option_followup::{parent_qid}",
        "source_parent_id": parent_qid,
        "dynamic_depth": 1,
        "requires_manual_review": False,
        "generated_by_ai": True,
        "explainability": child_explain,
    }


def _materialise_free_text_question(
    ai_ft: AIFreeTextQuestion,
    *,
    question_id: str,
    regulation: str,
    obligations_by_id: Mapping[str, Any],
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    mapped_brd = [b for b in ai_ft.mapped_brd_requirement_ids if b in set(all_brd_ids)]
    mapped_obl = [o for o in ai_ft.mapped_obligation_ids if o in set(all_obligation_ids)]
    obligation_id = mapped_obl[0] if mapped_obl else ""
    obligation = obligations_by_id.get(obligation_id) if obligation_id else None
    article = ai_ft.regulatory_basis or (
        getattr(obligation, "regulatory_basis", "") or regulation
    )
    source_refs: List[Dict[str, Any]] = []
    for bid in mapped_brd:
        refs = source_refs_by_brd.get(f"REQ:{bid}") or source_refs_by_brd.get(bid) or []
        for ref in refs:
            if isinstance(ref, dict) and ref not in source_refs:
                source_refs.append(ref)

    explain = {
        "regulation": regulation,
        "regulator": "",
        "article": article,
        "obligation_id": obligation_id,
        "brd_requirement_ids": mapped_brd,
        "rtm_trace_ids": [],
        "business_function": ai_ft.function,
        "business_area": ai_ft.area,
        "control_objective": ai_ft.rationale[:140] or "Qualitative SME narrative.",
        "theme": ai_ft.area,
        "reason": ai_ft.rationale or (
            "Free-text SME narrative captures qualitative context that "
            "closed questions cannot score."
        ),
        "expected_evidence": ", ".join(ai_ft.evidence_expectations),
        "risk_if_negative": (
            "Without SME narrative context, scoring may miss qualitative "
            "risk signals."
        ),
        "source_references": source_refs,
        "owning_team": ai_ft.owning_team,
        "team_rationale": "",
        "impact_level": ai_ft.impact_level,
        "impact_reason": "",
        "plain_language_explainer": ai_ft.plain_language_explainer,
        "evidence_expectations": list(ai_ft.evidence_expectations),
        "question_purpose": str(getattr(ai_ft, "question_purpose", "") or "impact+readiness"),
    }
    return {
        "question_id": question_id,
        "area": ai_ft.area or "Free Text / SME Narrative",
        "function": ai_ft.function or "Cross-functional",
        "question_type": "Open Ended",
        "question": ai_ft.question,
        "options": ["Free text response"],
        "mapped_requirement_ids": mapped_brd,
        "regulatory_basis": article,
        "confidence": max(0, min(100, int(ai_ft.confidence or 90))),
        "scoring_weight": 1,
        "impact_weight": 1,
        "impact_severity": ai_ft.impact_level,
        "impact_level": ai_ft.impact_level,
        "impact_reason": "",
        "owning_team": ai_ft.owning_team,
        "team_rationale": "",
        "evidence_expectations": list(ai_ft.evidence_expectations),
        "plain_language_explainer": ai_ft.plain_language_explainer,
        "question_purpose": str(getattr(ai_ft, "question_purpose", "") or "impact+readiness"),
        "targets_impact_dimension": "",
        "targets_readiness_dimension": "",
        "funnel_parent_id": "",
        "trigger_answers": [],
        "rationale": ai_ft.rationale,
        "is_free_text": True,
        "is_parent": False,
        "is_child": False,
        "child_question_ids": [],
        "branch_theme": ai_ft.area,
        "branch_rule_id": "",
        "source_parent_id": "",
        "dynamic_depth": 0,
        "requires_manual_review": False,
        "generated_by_ai": True,
        "explainability": explain,
    }


def _manual_review_placeholder(
    *,
    question_id: str,
    regulation: str,
    pair: Any,
    pair_severity: str,
    obligations_by_id: Mapping[str, Any],
    mapped_brd_ids: Sequence[str],
    mapped_obligation_ids: Sequence[str],
) -> Dict[str, Any]:
    obligation_id = mapped_obligation_ids[0] if mapped_obligation_ids else ""
    obligation = obligations_by_id.get(obligation_id) if obligation_id else None
    article = (getattr(obligation, "regulatory_basis", "") if obligation else "") or regulation
    reason = (
        "AI generation was not available or returned no grounded content "
        f"for {pair.area} / {pair.function}. An SME must draft the readiness "
        f"question by hand — referring to {article} and BRD requirements "
        f"{', '.join(mapped_brd_ids[:5]) or '(none mapped)'}."
    )
    return {
        "question_id": question_id,
        "area": pair.area,
        "function": pair.function,
        "question_type": "Open Ended",
        "question": (
            f"[Manual review required] Draft the readiness question for "
            f"{pair.area} / {pair.function} using the mapped BRD "
            f"requirements and {article}. Describe the current state, key "
            f"gaps and evidence available."
        ),
        "options": ["Free text response"],
        "mapped_requirement_ids": list(mapped_brd_ids)[:8],
        "regulatory_basis": article,
        "confidence": 50,
        "scoring_weight": 1,
        "impact_weight": 1,
        "impact_severity": pair_severity,
        "impact_level": pair_severity,
        "impact_reason": (
            f"{pair_severity}-severity area but no AI-generated grounded "
            "content was produced; SME review required to avoid hallucination."
        ),
        "owning_team": "Compliance",
        "team_rationale": (
            "Routed to Compliance by default for SME review. Please re-assign "
            "when a domain owner is confirmed."
        ),
        "evidence_expectations": [],
        "plain_language_explainer": "",
        "question_purpose": "impact+readiness",
        "targets_impact_dimension": "",
        "targets_readiness_dimension": "",
        "funnel_parent_id": "",
        "trigger_answers": [],
        "rationale": reason,
        "is_free_text": True,
        "is_parent": False,
        "is_child": False,
        "child_question_ids": [],
        "branch_theme": pair.area,
        "branch_rule_id": "",
        "source_parent_id": "",
        "dynamic_depth": 0,
        "requires_manual_review": True,
        "generated_by_ai": False,
        "explainability": {
            "regulation": regulation,
            "regulator": "",
            "article": article,
            "obligation_id": obligation_id,
            "brd_requirement_ids": list(mapped_brd_ids)[:8],
            "rtm_trace_ids": [],
            "business_function": pair.function,
            "business_area": pair.area,
            "control_objective": (
                f"Control objective for {pair.area} / {pair.function} — "
                "awaiting SME confirmation."
            ),
            "theme": pair.area,
            "reason": reason,
            "expected_evidence": "SME to specify",
            "risk_if_negative": (
                "Without SME review this control cannot be scored reliably."
            ),
            "source_references": [],
            "owning_team": "Compliance",
            "team_rationale": "SME triage",
            "impact_level": pair_severity,
            "impact_reason": (
                "Preserved as a placeholder because AI generation was "
                "unavailable — do not treat as hallucinated content."
            ),
            "plain_language_explainer": "",
            "evidence_expectations": [],
        },
    }


# ---------------------------------------------------------------------------
# Mapping helpers between BRD IDs / obligations / RTM
# ---------------------------------------------------------------------------


def _mapped_obligations_for_pair(
    pair: Any, obligations: Sequence[Any], mapped_brd_ids: Sequence[str],
) -> List[Any]:
    """Return obligations whose source_requirement_id overlaps the pair, or
    whose impacted_area/function matches the pair. Best-effort filter used
    to seed the LLM context — the AI still validates final IDs."""
    if not obligations:
        return []
    brd_set = set(mapped_brd_ids or [])
    area_l = (getattr(pair, "area", "") or "").lower()
    fn_l = (getattr(pair, "function", "") or "").lower()

    def _match(obl: Any) -> bool:
        src = str(getattr(obl, "source_requirement_id", "") or "")
        if src and src in brd_set:
            return True
        obl_area = (getattr(obl, "impacted_area", "") or "").lower()
        obl_fn = (getattr(obl, "impacted_function", "") or "").lower()
        if obl_area and area_l and (obl_area in area_l or area_l in obl_area):
            return True
        if obl_fn and fn_l and (obl_fn in fn_l or fn_l in obl_fn):
            return True
        return False

    matched = [o for o in obligations if _match(o)]
    return matched if matched else list(obligations[:6])


def _mapped_rtm_for_pair(
    pair: Any, rtm_entries: Sequence[Any], mapped_brd_ids: Sequence[str],
) -> List[Any]:
    if not rtm_entries:
        return []
    brd_set = set(mapped_brd_ids or [])
    area_l = (getattr(pair, "area", "") or "").lower()

    def _match(entry: Any) -> bool:
        if getattr(entry, "business_requirement_id", "") in brd_set:
            return True
        entry_area = (getattr(entry, "impacted_area", "") or "").lower()
        if entry_area and area_l and (entry_area in area_l or area_l in entry_area):
            return True
        return False

    matched = [e for e in rtm_entries if _match(e)]
    return matched[:8]


def _obligation_applies_to_roles(obligation: Any, roles: Sequence[str]) -> bool:
    """Return True when ``obligation`` is applicable or partial for any role.

    Obligations produced by Agent 1 carry per-role tags
    (``applicable_roles``, ``partial_roles``, ``uncertain_roles``,
    ``not_applicable_roles``) plus an ``is_applicable_for`` method. We
    prefer the method when available and fall back to a tag lookup
    otherwise. Anything unknown is treated as applicable so the filter
    never blocks legitimate content just because Agent 1 could not
    classify it.
    """
    if not roles:
        return True
    if hasattr(obligation, "is_applicable_for"):
        try:
            return bool(obligation.is_applicable_for(roles))
        except Exception:
            return True
    role_set = {r.strip() for r in roles if r}
    applicable = set(getattr(obligation, "applicable_roles", []) or [])
    partial = set(getattr(obligation, "partial_roles", []) or [])
    uncertain = set(getattr(obligation, "uncertain_roles", []) or [])
    not_applicable = set(getattr(obligation, "not_applicable_roles", []) or [])
    if applicable & role_set:
        return True
    if partial & role_set:
        return True
    if uncertain & role_set:
        return True
    # Explicit not-applicable classification - drop it.
    if not_applicable and role_set.issubset(not_applicable):
        return False
    # Any positive-side tags exist AND cover other roles but not ours -
    # drop it (the interpretation engine classified this obligation and
    # our roles simply are not on the applicability list).
    if applicable or partial or uncertain:
        return False
    # No classification info at all -> keep the obligation.
    return True


def _client_profile_lines(profile: Optional[Mapping[str, Any]]) -> List[str]:
    if not profile:
        return []
    out: List[str] = []
    for key, values in profile.items():
        if not values:
            continue
        if isinstance(values, (list, tuple)):
            preview = ", ".join(str(v) for v in list(values)[:6])
        else:
            preview = str(values)
        if preview:
            out.append(f"{key}: {preview}")
    return out


def _weight_for_severity(severity: str) -> int:
    return {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
    }.get((severity or "").strip().lower(), 3)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def generate_ai_questionnaire(
    *,
    regulation: str,
    requirements: Sequence[Any],
    impact_pairs: Sequence[Any],
    obligations: Sequence[Any],
    rtm_entries: Sequence[Any] = (),
    impact: Optional[Any] = None,
    readiness: Optional[Any] = None,
    client_roles: Sequence[str] = (),
    client_profile: Optional[Mapping[str, Any]] = None,
    source_refs_by_item: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
    client: Optional[GenAIClient] = None,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    free_text_count: int = DEFAULT_FREE_TEXT_COUNT,
) -> List[Dict[str, Any]]:
    """Generate the full AI-driven questionnaire question list.

    The ``impact`` and ``readiness`` assessments (from Agent 1 /
    :mod:`services.ai_assessment_intelligence`) are sliced per pair
    and injected into the per-pair prompt so the LLM crafts questions
    that specifically target the affected items and the client's
    weakest readiness dimensions — the "ask the most-required info
    to best assess impact + readiness" behaviour.

    Returns a list of question dicts ready to be dropped into
    ``package_dict()``. When ``client`` is None the function returns a
    small list of "Manual review required" placeholders (one per top
    impact pair) so downstream code still gets a valid package.
    """
    if not impact_pairs:
        return []

    # ------------------------------------------------------------------
    # Role-based obligation filtering
    # ------------------------------------------------------------------
    # The BRD text is intentionally client-agnostic (see
    # ``services.brd_frd_generator``). Client-specific tailoring happens
    # here: we filter the obligation set down to those that are actually
    # applicable to the selected role combination BEFORE we ask the LLM
    # to draft questions for each pair. That way the LLM only ever sees
    # regulatory surface that matters for this client - questions about
    # "third-party ICT concentration" are dropped for a client with no
    # vendor keywords, questions about "trading desk resilience" are
    # dropped for a retail bank, and so on.
    #
    # We only apply the filter when at least one role is selected AND the
    # filter keeps at least one obligation - otherwise we would leave the
    # generator with nothing to talk about, which is worse than an
    # over-scoped pack.
    if client_roles and obligations:
        role_filtered = [
            o for o in obligations
            if _obligation_applies_to_roles(o, client_roles)
        ]
        if role_filtered:
            obligations = role_filtered

    # Prioritise pairs by severity so the most-impacted areas are asked first.
    ranked = _prioritise_pairs(impact_pairs, impact, max_pairs=max_pairs)

    req_by_id = {getattr(r, "normalized_id", ""): r for r in requirements or []}
    all_brd_ids = list(req_by_id.keys())
    obligations_by_id = {
        getattr(o, "obligation_id", ""): o for o in obligations or []
    }
    all_obligation_ids = list(obligations_by_id.keys())
    source_refs_by_brd = dict(source_refs_by_item or {})
    profile_lines = _client_profile_lines(client_profile)

    all_questions: List[Dict[str, Any]] = []
    q_counter = 1

    for pair, severity in ranked:
        mapped_brd_ids = list(getattr(pair, "requirement_ids", []) or [])
        mapped_reqs = [req_by_id[b] for b in mapped_brd_ids if b in req_by_id]
        mapped_oblig = _mapped_obligations_for_pair(pair, obligations, mapped_brd_ids)
        mapped_rtm = _mapped_rtm_for_pair(pair, rtm_entries, mapped_brd_ids)
        rtm_ids = [getattr(e, "traceability_id", "") for e in mapped_rtm]
        impact_weight = _weight_for_severity(severity)
        impact_count, readiness_count = FUNNELS_PER_PAIR.get(severity, (1, 2))
        area = getattr(pair, "area", "") or ""
        impact_summary_area = _impact_summary_for_area(impact, area)
        readiness_summary_area = _readiness_summary_for_area(readiness, area)

        pair_questions = _generate_for_pair(
            regulation=regulation,
            client_roles=client_roles,
            client_profile_lines=profile_lines,
            pair=pair,
            pair_severity=severity,
            requirements=mapped_reqs,
            obligations=mapped_oblig,
            rtm=mapped_rtm,
            impact_summary=impact_summary_area,
            readiness_summary=readiness_summary_area,
            impact_count=impact_count,
            readiness_count=readiness_count,
            client=client,
            obligations_by_id=obligations_by_id,
            all_brd_ids=all_brd_ids,
            all_obligation_ids=all_obligation_ids,
            source_refs_by_brd=source_refs_by_brd,
            rtm_ids=rtm_ids,
            mapped_brd_ids=mapped_brd_ids,
            impact_weight=impact_weight,
            q_start_counter=q_counter,
        )
        all_questions.extend(pair_questions)
        q_counter += len(pair_questions)

    # Cross-cutting free-text SME narratives.
    free_text_qs = _generate_free_text(
        regulation=regulation,
        client_roles=client_roles,
        top_pairs=ranked,
        obligations=obligations,
        obligations_by_id=obligations_by_id,
        all_brd_ids=all_brd_ids,
        all_obligation_ids=all_obligation_ids,
        source_refs_by_brd=source_refs_by_brd,
        client=client,
        requested_count=free_text_count,
        start_qid_counter=q_counter,
    )
    all_questions.extend(free_text_qs)

    # Renumber question IDs to a stable Q-0001 scheme + fix child references.
    return _renumber_and_relink(all_questions)


def _generate_for_pair(
    *,
    regulation: str,
    client_roles: Sequence[str],
    client_profile_lines: Sequence[str],
    pair: Any,
    pair_severity: str,
    requirements: Sequence[Any],
    obligations: Sequence[Any],
    rtm: Sequence[Any],
    impact_summary: str,
    readiness_summary: str,
    impact_count: int,
    readiness_count: int,
    client: Optional[GenAIClient],
    obligations_by_id: Mapping[str, Any],
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
    rtm_ids: Sequence[str],
    mapped_brd_ids: Sequence[str],
    impact_weight: int,
    q_start_counter: int,
) -> List[Dict[str, Any]]:
    """Generate the AI question funnel for one impact pair (with fallback).

    Produces a balanced set of ``impact_count`` impact-purpose parents
    and ``readiness_count`` readiness-purpose parents. Every parent
    receives its adaptive follow-up children as before.
    """
    if client is None or (impact_count + readiness_count) <= 0:
        return [_manual_review_placeholder(
            question_id=f"Q-{q_start_counter:04d}",
            regulation=regulation,
            pair=pair,
            pair_severity=pair_severity,
            obligations_by_id=obligations_by_id,
            mapped_brd_ids=mapped_brd_ids,
            mapped_obligation_ids=[getattr(o, "obligation_id", "") for o in obligations],
        )]

    prompt = _funnel_prompt(
        regulation=regulation,
        client_roles=client_roles,
        client_profile_lines=client_profile_lines,
        pair=pair,
        pair_severity=pair_severity,
        mapped_requirements=requirements,
        mapped_obligations=obligations,
        mapped_rtm=rtm,
        impact_summary=impact_summary,
        readiness_summary=readiness_summary,
        impact_count=impact_count,
        readiness_count=readiness_count,
    )
    try:
        ai_bank: AIQuestionBank = client.generate(
            schema_model=AIQuestionBank,
            component_name=f"Questionnaire funnel: {pair.area} / {pair.function}",
            component_instruction=prompt,
            context="",
            system_instruction=_SYSTEM_INSTRUCTION,
            regulation=regulation,
            client_roles=client_roles,
        )
    except Exception:
        return [_manual_review_placeholder(
            question_id=f"Q-{q_start_counter:04d}",
            regulation=regulation,
            pair=pair,
            pair_severity=pair_severity,
            obligations_by_id=obligations_by_id,
            mapped_brd_ids=mapped_brd_ids,
            mapped_obligation_ids=[getattr(o, "obligation_id", "") for o in obligations],
        )]

    if not ai_bank or not ai_bank.closed_questions:
        return [_manual_review_placeholder(
            question_id=f"Q-{q_start_counter:04d}",
            regulation=regulation,
            pair=pair,
            pair_severity=pair_severity,
            obligations_by_id=obligations_by_id,
            mapped_brd_ids=mapped_brd_ids,
            mapped_obligation_ids=[getattr(o, "obligation_id", "") for o in obligations],
        )]

    out: List[Dict[str, Any]] = []
    q_counter = q_start_counter
    for ai_q in ai_bank.closed_questions:
        parent_qid = f"Q-{q_counter:04d}"
        q_counter += 1
        # Pre-count children so we can reserve their IDs.
        child_labels: List[Tuple[str, str, List[str], bool]] = []  # (label, followup_q, options, free_text)
        for opt in ai_q.options:
            if opt.triggers_followup and opt.followup_question:
                child_labels.append((
                    opt.label, opt.followup_question,
                    list(opt.followup_options or []),
                    bool(opt.followup_is_free_text),
                ))
        child_ids = [f"Q-{q_counter + i:04d}" for i in range(len(child_labels))]

        parent_dict = _materialise_parent_question(
            ai_q,
            question_id=parent_qid,
            child_ids=child_ids,
            regulation=regulation,
            pair=pair,
            obligations_by_id=obligations_by_id,
            all_brd_ids=all_brd_ids,
            all_obligation_ids=all_obligation_ids,
            rtm_ids=rtm_ids,
            source_refs_by_brd=source_refs_by_brd,
            impact_weight=impact_weight,
        )
        out.append(parent_dict)

        for (label, fu_q, fu_opts, fu_ft), child_qid in zip(child_labels, child_ids):
            child_dict = _materialise_child_question(
                parent_ai_q=ai_q,
                parent_qid=parent_qid,
                child_qid=child_qid,
                triggering_option_label=label,
                followup_question=fu_q,
                followup_options=fu_opts,
                followup_is_free_text=fu_ft,
                regulation=regulation,
                pair=pair,
                obligations_by_id=obligations_by_id,
                all_brd_ids=all_brd_ids,
                all_obligation_ids=all_obligation_ids,
                rtm_ids=rtm_ids,
                source_refs_by_brd=source_refs_by_brd,
                impact_weight=impact_weight,
            )
            out.append(child_dict)
        q_counter += len(child_labels)

    return out


def _generate_free_text(
    *,
    regulation: str,
    client_roles: Sequence[str],
    top_pairs: Sequence[Tuple[Any, str]],
    obligations: Sequence[Any],
    obligations_by_id: Mapping[str, Any],
    all_brd_ids: Sequence[str],
    all_obligation_ids: Sequence[str],
    source_refs_by_brd: Mapping[str, List[Dict[str, Any]]],
    client: Optional[GenAIClient],
    requested_count: int,
    start_qid_counter: int,
) -> List[Dict[str, Any]]:
    """Generate cross-cutting free-text SME narrative questions."""
    if client is None or requested_count <= 0 or not top_pairs:
        return []

    prompt = _free_text_prompt(
        regulation=regulation,
        client_roles=client_roles,
        top_pairs=top_pairs,
        obligations=obligations,
        requested_count=requested_count,
    )
    try:
        bank: AIFreeTextBank = client.generate(
            schema_model=AIFreeTextBank,
            component_name="Free-text SME narrative questions",
            component_instruction=prompt,
            context="",
            system_instruction=_SYSTEM_INSTRUCTION,
            regulation=regulation,
            client_roles=client_roles,
        )
    except Exception:
        return []

    if not bank or not bank.questions:
        return []

    out: List[Dict[str, Any]] = []
    q_counter = start_qid_counter
    for ai_ft in bank.questions[:requested_count]:
        qid = f"Q-{q_counter:04d}"
        q_counter += 1
        out.append(_materialise_free_text_question(
            ai_ft,
            question_id=qid,
            regulation=regulation,
            obligations_by_id=obligations_by_id,
            all_brd_ids=all_brd_ids,
            all_obligation_ids=all_obligation_ids,
            source_refs_by_brd=source_refs_by_brd,
        ))
    return out


def _renumber_and_relink(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign contiguous Q-0001... IDs and update parent/child references."""
    if not questions:
        return []
    old_to_new = {
        q["question_id"]: f"Q-{i:04d}"
        for i, q in enumerate(questions, start=1)
    }
    for i, q in enumerate(questions, start=1):
        new_id = f"Q-{i:04d}"
        q["question_id"] = new_id
    for q in questions:
        parent = q.get("funnel_parent_id") or ""
        if parent:
            q["funnel_parent_id"] = old_to_new.get(parent, parent)
            q["source_parent_id"] = old_to_new.get(q.get("source_parent_id", parent), parent)
        # Options may carry followup_question_id references.
        for opt in q.get("options") or []:
            if isinstance(opt, dict) and opt.get("followup_question_id"):
                opt["followup_question_id"] = old_to_new.get(
                    opt["followup_question_id"], opt["followup_question_id"],
                )
        # child_question_ids
        if q.get("child_question_ids"):
            q["child_question_ids"] = [
                old_to_new.get(cid, cid) for cid in q["child_question_ids"]
            ]
    return questions


__all__ = [
    "AIFreeTextBank",
    "AIFreeTextQuestion",
    "AIOption",
    "AIQuestion",
    "AIQuestionBank",
    "CANONICAL_IMPACT_LEVELS",
    "CANONICAL_READINESS_LABELS",
    "CANONICAL_TEAMS",
    "DEFAULT_FREE_TEXT_COUNT",
    "DEFAULT_MAX_PAIRS",
    "FUNNELS_PER_PAIR",
    "generate_ai_questionnaire",
]
