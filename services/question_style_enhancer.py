"""Deterministic diversification of question styles.

The AI questionnaire generator historically emits almost every question as a
Single Select with generic readiness options ("Fully / Partially / Not
implemented", "Yes / No / Ad hoc", …). Product feedback: reviewers want a
richer answering surface — some questions should be **Multi Select** so a
user can flag "several of these apply simultaneously", and some should offer
**quantitative brackets** (Budget, Timeline, Coverage %, Frequency, Team size)
so answers roll up into meaningful readiness bands without long free-text.

This module is a small, deterministic post-processor that runs once, right
after ``dedupe_and_resequence_questions``. It:

* Converts a question to ``Multi Select`` when its wording clearly invites
  multiple answers ("which of the following", "select all that apply",
  "check all that apply", "which teams", …). The existing options stay
  because they remain individually scorable — the widget just lets users
  pick more than one to reflect a mixed reality.

* Replaces the options on questions whose wording is intrinsically
  quantitative — Budget, Cost, Investment, Timeline, Deadline, When,
  Coverage %, Percentage, Frequency, Team size, Headcount, SLA — with a
  curated bracket set carrying per-option ``score_value`` metadata. The
  bracket set is chosen from a small taxonomy so downstream scoring
  works out of the box.

The module is deterministic (no LLM calls) and idempotent — running it on
already-diversified questions is a no-op.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Multi Select detection
# ---------------------------------------------------------------------------

_MULTI_SELECT_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bselect all that apply\b",
        r"\bcheck all that apply\b",
        r"\bwhich of the following\b",
        r"\bwhich of these\b",
        r"\ball of the following that apply\b",
        # Broad "which <noun>" patterns — up to three adjective/noun
        # tokens in front of the head noun (hyphens allowed inside a
        # token) so we catch phrasings like "which asset categories",
        # "which risk domains", "which critical systems", "which
        # third-party providers", "which top-tier vendors", …
        r"\bwhich(?:\s+[\w-]+){0,3}\s+"
        r"(?:teams?|departments?|functions?|business\s+units?|"
        r"systems?|platforms?|tools?|applications?|"
        r"controls?|policies|procedures?|processes?|"
        r"stakeholders?|owners?|roles?|"
        r"categories|category|types?|assets?|classes|domains|areas?|"
        r"items?|components?|artefacts?|artifacts?|"
        r"data\s+(?:types?|categories|assets?|domains?)|"
        r"vendors?|third[- ]parties|providers?|counterparties?|"
        r"risks?|threats?|scenarios?|jurisdictions?|regulators?)\b",
    )
)


def _looks_multi_select(question_text: str) -> bool:
    """True when the question wording clearly invites multiple answers."""
    if not question_text:
        return False
    return any(pat.search(question_text) for pat in _MULTI_SELECT_PATTERNS)


# ---------------------------------------------------------------------------
# Quantitative bracket sets
# ---------------------------------------------------------------------------
#
# Every bracket carries ``score_value`` (0-100, higher = better readiness),
# ``impact_value`` (mirror of score_value, higher = worse impact) and a
# ``rationale`` for the audit trail. ``score_value=None`` means the answer is
# excluded from scoring (used for "Unknown" / "Not budgeted yet").

def _bracket(label: str, score: Optional[float], rationale: str) -> Dict[str, Any]:
    """Build a bracket option dict compatible with the scoring engine."""
    return {
        "label": label,
        "score_value": score,
        "impact_value": None if score is None else round(100.0 - float(score), 1),
        "risk_value": None if score is None else round(100.0 - float(score), 1),
        "rationale": rationale,
        "quantitative": True,
    }


BUDGET_OPTIONS: List[Dict[str, Any]] = [
    _bracket("Not budgeted yet",       0,   "No funding assigned — remediation cannot start."),
    _bracket("< $100K",                25,  "Very light funding — likely limited to policy work."),
    _bracket("$100K – $500K",          50,  "Moderate funding — covers targeted controls."),
    _bracket("$500K – $2M",            75,  "Solid funding — end-to-end remediation is feasible."),
    _bracket("> $2M",                  100, "Well-funded — supports full-scope programme."),
    _bracket("Not applicable",         None, "Explicit N/A — excluded from scoring."),
]


TIMELINE_OPTIONS: List[Dict[str, Any]] = [
    _bracket("No timeline defined",    0,   "No target date — programme not committed."),
    _bracket("> 12 months",            25,  "Long horizon — regulatory deadline risk is high."),
    _bracket("6 – 12 months",          50,  "Reasonable horizon but requires close tracking."),
    _bracket("3 – 6 months",           75,  "Tight but feasible — aligned to typical deadlines."),
    _bracket("< 3 months",             100, "Imminent completion — deadline risk is low."),
    _bracket("Not applicable",         None, "Explicit N/A — excluded from scoring."),
]


COVERAGE_OPTIONS: List[Dict[str, Any]] = [
    _bracket("0 – 25%",                15,  "Very low coverage — most scope is not addressed."),
    _bracket("25 – 50%",               40,  "Partial coverage — significant scope still open."),
    _bracket("50 – 75%",               65,  "Majority coverage — targeted gaps remain."),
    _bracket("75 – 90%",               85,  "High coverage — a small residual to close."),
    _bracket("90 – 100%",              100, "Effectively full coverage."),
    _bracket("Unknown",                None, "Coverage not measured — excluded from scoring."),
]


FREQUENCY_OPTIONS: List[Dict[str, Any]] = [
    _bracket("Never",                  0,   "The activity is not performed at all."),
    _bracket("Ad hoc / unplanned",     25,  "Runs only in response to incidents — not a control."),
    _bracket("Annually",               60,  "Runs at least once per year — baseline cadence."),
    _bracket("Quarterly",              80,  "Regular quarterly cadence — solid control frequency."),
    _bracket("Monthly or more often",  100, "Continuous / near-real-time cadence."),
    _bracket("Not applicable",         None, "Explicit N/A — excluded from scoring."),
]


TEAM_SIZE_OPTIONS: List[Dict[str, Any]] = [
    _bracket("None dedicated",         0,   "No one is assigned — capacity gap is severe."),
    _bracket("1 – 2 people",           35,  "Very small team — capacity is a material risk."),
    _bracket("3 – 5 people",           65,  "Reasonable team size for a focused programme."),
    _bracket("6 – 10 people",          85,  "Well-staffed team — should meet demand."),
    _bracket("More than 10",           100, "Large team — capacity is not a constraint."),
    _bracket("Not applicable",         None, "Explicit N/A — excluded from scoring."),
]


SLA_OPTIONS: List[Dict[str, Any]] = [
    _bracket("No SLA defined",         0,   "No SLA target — nothing to measure against."),
    _bracket("Defined but not tracked", 30, "SLA exists on paper only — no measurement."),
    _bracket("Tracked, missed often",  55,  "Measured but frequently missed — remediation needed."),
    _bracket("Tracked, mostly met",    80,  "Reliably met — targeted improvements remain."),
    _bracket("Consistently met / exceeded", 100, "Fully compliant with the defined SLA."),
    _bracket("Not applicable",         None, "Explicit N/A — excluded from scoring."),
]


# ---------------------------------------------------------------------------
# Quantitative-question detection
# ---------------------------------------------------------------------------
#
# Each entry maps (compiled regex → bracket set + short type tag). The tag is
# stored on the question as ``quantitative_type`` so the UI / scoring engine
# can render / weight it accordingly.

_QUANT_RULES: Tuple[Tuple[re.Pattern, str, List[Dict[str, Any]]], ...] = (
    (
        re.compile(
            r"\b(?:budget|budgeted|funding|invest(?:ment|ed)?|spend|cost|capex|opex)\b",
            re.IGNORECASE,
        ),
        "budget",
        BUDGET_OPTIONS,
    ),
    (
        re.compile(
            r"\b(?:timeline|by when|deadline|target date|go[- ]?live|target completion|"
            r"when (?:will|do you plan|is)|how long|time (?:frame|line)|expected date)\b",
            re.IGNORECASE,
        ),
        "timeline",
        TIMELINE_OPTIONS,
    ),
    (
        re.compile(
            r"\b(?:coverage|what (?:proportion|percentage|share|fraction)|"
            r"how much of|% of|percent of|what % )\b",
            re.IGNORECASE,
        ),
        "coverage",
        COVERAGE_OPTIONS,
    ),
    (
        re.compile(
            r"\b(?:how (?:often|frequently)|cadence|frequency|schedule|"
            r"testing frequency|running (?:frequency|cadence))\b",
            re.IGNORECASE,
        ),
        "frequency",
        FREQUENCY_OPTIONS,
    ),
    (
        re.compile(
            r"\b(?:how many people|team size|headcount|fte|full[- ]time equivalent|"
            r"resources dedicated|dedicated resources|staffing level)\b",
            re.IGNORECASE,
        ),
        "team_size",
        TEAM_SIZE_OPTIONS,
    ),
    (
        re.compile(
            r"\bsla\b|\bservice[- ]level(?: agreement)?\b|"
            r"\brto\b|\brpo\b|\brecovery time\b|\bresponse time\b|\bturnaround\b",
            re.IGNORECASE,
        ),
        "sla",
        SLA_OPTIONS,
    ),
)


def _find_quantitative_rule(question_text: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Return ``(type_tag, bracket_options)`` when the question is quantitative."""
    if not question_text:
        return None
    for pattern, tag, options in _QUANT_RULES:
        if pattern.search(question_text):
            # Return a deep-copyable snapshot so mutations don't leak.
            return tag, [dict(o) for o in options]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get(question: Any, key: str, default: Any = None) -> Any:
    """Read a field on either a ``Question`` dataclass or a plain dict."""
    if isinstance(question, Mapping):
        return question.get(key, default)
    return getattr(question, key, default)


def _set(question: Any, key: str, value: Any) -> None:
    """Write a field on either a ``Question`` dataclass or a plain dict."""
    if isinstance(question, dict):
        question[key] = value
        return
    try:
        setattr(question, key, value)
    except Exception:
        # Best-effort — some dataclasses are frozen; skip silently.
        pass


# ---------------------------------------------------------------------------
# Empty-options sanitiser
# ---------------------------------------------------------------------------
#
# The AI generator occasionally emits closed (Single/Multi Select) questions
# with an empty ``options`` list - the SME then sees a dropdown that only
# shows the "— Select an answer —" placeholder and cannot answer. Product
# feedback: never render an unanswerable question.
#
# ``sanitize_questions_without_options`` fixes those in place. If the wording
# clearly invites a narrative response ("What ...?", "How ...?", "Describe
# ...", "Explain ...", "Why ...", "To what extent ..."), the question is
# converted to Open Ended so the SME can type. Anything else that still has
# no usable options is dropped from the returned list.


_FREE_TEXT_WORDING = re.compile(
    r"^\s*(?:"
    r"what\s+(?:specific|are|is|does|would|could|steps|resources|challenges|"
    r"processes|controls|risks|gaps|artefacts|artifacts|actions|barriers|"
    r"kpis|metrics|dependencies|considerations|assumptions|mitigations)"
    r"|how\s+(?:do|does|would|could|will|are|is)"
    r"|describe|explain|why|in\s+what\s+way|to\s+what\s+extent"
    r"|please\s+(?:describe|explain|provide|list|share|detail|outline)"
    r"|provide\s+(?:details|a\s+description|examples?|context)"
    r"|list\s+(?:the|any|all)"
    r"|walk\s+us\s+through|share\s+your"
    r")\b",
    re.IGNORECASE,
)


_PLACEHOLDER_OPTION_TOKENS = frozenset({
    "", "-", "—", "n/a", "na", "select an answer", "— select an answer —",
    "select…", "select...", "choose one", "choose an option",
    "please select", "please select an option", "pick an option",
    "tbd", "to be determined", "not sure", "unknown",
})


def _usable_option_count(options: Any) -> int:
    """Count how many entries in ``options`` are meaningful selections.

    Options may be plain strings or dicts (``{"label": "...", "score_value":
    ..., ...}``). Blank strings, dashes, and generic placeholder phrases are
    ignored so they don't fool the answerable-question check.
    """
    if not options:
        return 0
    count = 0
    for opt in options:
        if isinstance(opt, Mapping):
            label = str(opt.get("label") or opt.get("value") or "").strip()
        else:
            label = str(opt or "").strip()
        if not label:
            continue
        if label.lower() in _PLACEHOLDER_OPTION_TOKENS:
            continue
        count += 1
    return count


def sanitize_questions_without_options(questions: Sequence[Any]) -> List[Any]:
    """Remove or convert closed questions that have no answerable options.

    Rules (in order):

    1. Free-text / Open Ended questions (``is_free_text=True`` or an "open"
       / "free" ``question_type``) are always kept as-is.
    2. Closed questions with at least two usable options are kept as-is.
    3. Closed questions with fewer than two usable options are converted to
       Open Ended when their wording clearly invites a narrative reply
       ("What ...", "How ...", "Describe ...", "Explain ...", "Why ...",
       "To what extent ...").
    4. Anything else that still has no usable options is dropped.

    Returns a new list containing only the surviving questions. The input
    sequence is not modified in place (aside from the type/flag updates on
    kept questions), so callers can compare lengths to know how many were
    filtered.
    """
    if not questions:
        return list(questions)

    kept: List[Any] = []
    for q in questions:
        qtype_raw = str(_get(q, "question_type", "") or "").strip().lower()
        is_free = bool(_get(q, "is_free_text"))
        is_open_type = (
            "open" in qtype_raw or "free" in qtype_raw or "text" in qtype_raw
        )
        if is_free or is_open_type:
            if is_open_type and not is_free:
                _set(q, "is_free_text", True)
            kept.append(q)
            continue

        options = _get(q, "options", []) or []
        if _usable_option_count(options) >= 2:
            kept.append(q)
            continue

        wording = str(_get(q, "question", "") or "")
        if _FREE_TEXT_WORDING.search(wording):
            _set(q, "question_type", "Open Ended")
            _set(q, "is_free_text", True)
            _set(q, "options", [])
            kept.append(q)
            continue

        continue

    return kept


def diversify_question_styles(questions: Sequence[Any]) -> List[Any]:
    """Rewrite question types + options in place based on wording.

    Returns the same list (mutated) so callers can chain directly. Free-text
    questions and questions carrying ``requires_manual_review`` are skipped —
    their surface is deliberate. Already-quantitative questions are also
    skipped so the transform is idempotent.
    """
    if not questions:
        return list(questions)

    for q in questions:
        if _get(q, "is_free_text"):
            continue
        if _get(q, "requires_manual_review"):
            continue
        # Idempotency guard.
        if _get(q, "quantitative_type"):
            continue

        text = str(_get(q, "question", "") or "")

        # --- 1. Quantitative bracket injection -----------------------------
        quant = _find_quantitative_rule(text)
        if quant is not None:
            tag, bracket_options = quant
            _set(q, "options", bracket_options)
            _set(q, "quantitative_type", tag)
            # Quantitative questions are single-choice by design (pick one
            # bracket). Ensure the type is Single Select even if the AI
            # emitted a Multi Select for the same question.
            _set(q, "question_type", "Single Select")
            continue

        # --- 2. Multi Select conversion -----------------------------------
        if _looks_multi_select(text):
            existing_type = str(_get(q, "question_type", "") or "").lower()
            if "multi" not in existing_type:
                _set(q, "question_type", "Multi Select")

    return list(questions)


__all__ = [
    "BUDGET_OPTIONS",
    "TIMELINE_OPTIONS",
    "COVERAGE_OPTIONS",
    "FREQUENCY_OPTIONS",
    "TEAM_SIZE_OPTIONS",
    "SLA_OPTIONS",
    "diversify_question_styles",
    "sanitize_questions_without_options",
]
