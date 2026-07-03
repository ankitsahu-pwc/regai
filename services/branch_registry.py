"""Per-option, regulation-aware branch registry.

This is the **source of truth for true option-level adaptive branching**. It is
a pure-data module deliberately kept free of any UI/state imports so the
authoring team can extend it (or hot-swap it for a YAML/JSON file) without
touching the scoring engine.

Lookup keys are 4-tuples::

    (regulation, theme, question_kind, selected_answer_label)

* ``regulation`` matches ``QuestionnairePackage.metadata.regulation`` (case
  insensitive — looked up with ``.upper()`` for stability).
* ``theme`` matches the canonical theme labels emitted by
  :func:`services.questionnaire_generator.infer_themes` and tagged on each
  generated Question (we propagate the theme via the new
  ``branch_theme`` field on Question — see ``questionnaire_generator``).
* ``question_kind`` matches :func:`services.scoring_engine.question_kind`
  (``"coverage"``, ``"ownership"``, ``"evidence"``, ``"risk"``,
  ``"remediation"``, or any theme-specific kind). The canonical v13
  implementation-status root question maps to ``"coverage"``.
* ``selected_answer_label`` is the exact option label the user picked
  (e.g. ``"Fully Implemented"``, ``"Partially Implemented"``,
  ``"Not Implemented"``, ``"Not Applicable"``, plus the legacy v12 labels).

Each entry is a list of follow-up specs. A spec is a ``dict`` with at minimum
``question_id``, ``question``, ``options`` and ``question_type``; richer
metadata (``rationale``, ``mapped_requirement_ids``, ``scoring_weight``,
``confidence``, ``branch_rule_id``, ``branch_theme`` etc.) is optional and
falls back to sensible defaults derived from the parent question.

The engine treats a missing key as **"no specific branch — fall back to the
AI-driven option-aware generator, then to the generic dynamic_followups
family"**. Adding a new path therefore never risks breaking an existing one.

v13 — Full canonical coverage
=============================

Every supported theme has registered branches for the four canonical
implementation-status answers:

* "Fully Implemented"     → evidence + last-test validation branch
* "Partially Implemented" → which stage incomplete + teams affected + evidence
* "Not Implemented"       → blocker + ownership + planning + support branch
* "Not Applicable"        → single justification follow-up
* "Unknown"               → discovery branch (who can answer this?)

Themes covered: Governance, ICT risk management, Incident reporting,
Resilience testing, Third-party risk, Security and access, Data and evidence,
Reporting.

Legacy v10/v11 coverage answer labels ("Complete", "Mostly complete",
"Partially complete", "Not started") are aliased to the canonical labels so
saved questionnaires from older versions keep routing correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

BranchKey = Tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# Branch-spec factories
# ---------------------------------------------------------------------------
#
# Each factory takes the canonical theme label (e.g. ``"Incident reporting"``)
# and returns a list of follow-up specs for one canonical answer. Centralising
# the spec generation here means we have one place to tune the partial /
# not-started / fully-implemented question families for every theme.
# ---------------------------------------------------------------------------


def _theme_slug(theme: str) -> str:
    return theme.replace(" ", "_").replace("-", "_").replace("/", "_").lower()


# Per-theme noun used as the subject of the follow-up question text.
_THEME_NOUNS = {
    "Governance": "governance framework",
    "ICT risk management": "ICT risk-management framework",
    "Incident reporting": "ICT incident classification and reporting process",
    "Resilience testing": "operational-resilience testing programme",
    "Third-party risk": "ICT third-party oversight and contracting process",
    "Security and access": "security, access and vulnerability-management controls",
    "Data and evidence": "evidence dictionary, lineage and retention controls",
    "Reporting": "management reporting framework",
    "General regulatory coverage": "regulatory control",
}


# Per-theme regulatory citation (used in rationale text only).
_THEME_CITATIONS = {
    "Governance": "DORA Article 5 (governance and organisation)",
    "ICT risk management": "DORA Article 6 (ICT risk-management framework)",
    "Incident reporting": "DORA Articles 17-20 (major-incident management and reporting)",
    "Resilience testing": "DORA Articles 24-27 (resilience testing, including TLPT)",
    "Third-party risk": "DORA Articles 28-30 (third-party ICT risk and key contractual clauses)",
    "Security and access": "DORA Articles 8-10 (identification, protection, detection)",
    "Data and evidence": "DORA Article 6 + RTS on ICT risk-management evidence requirements",
    "Reporting": "DORA Article 5 + RTS on management-body oversight",
    "General regulatory coverage": "the mapped regulatory clause",
}


def _fully_implemented_specs(theme: str) -> List[Dict[str, Any]]:
    slug = _theme_slug(theme)
    noun = _THEME_NOUNS.get(theme, _THEME_NOUNS["General regulatory coverage"])
    citation = _THEME_CITATIONS.get(theme, _THEME_CITATIONS["General regulatory coverage"])
    return [
        {
            "question_id": f"{slug.upper()}_FULL_EVIDENCE",
            "question": (
                f"What evidence validates that the {noun} is operating effectively today?"
            ),
            "options": [
                "Independent test / audit report",
                "Operational log + control-effectiveness review",
                "Management-body / governance pack approval evidence",
                "Regulatory submission or attestation",
                "Multiple of the above",
                "No documented evidence yet",
            ],
            "question_type": "Single Select",
            "rationale": (
                f"'Fully Implemented' must be substantiated to score 100 — this branch confirms the "
                f"control is auditable against {citation}, not just claimed on paper."
            ),
            "branch_rule_id": f"{slug}__fully__evidence",
            "scoring_weight": 3,
        },
        {
            "question_id": f"{slug.upper()}_FULL_LAST_TESTED",
            "question": (
                f"When was the {noun} last tested or independently reviewed end-to-end?"
            ),
            "options": [
                "Within the last 6 months",
                "Within the last 12 months",
                "More than 12 months ago",
                "Never tested end-to-end",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                f"Periodic testing is expected under {citation}. Anything older than 12 months "
                f"downgrades CXO readiness regardless of the paper coverage claim."
            ),
            "branch_rule_id": f"{slug}__fully__last_tested",
            "scoring_weight": 2,
        },
    ]


def _partially_implemented_specs(theme: str) -> List[Dict[str, Any]]:
    slug = _theme_slug(theme)
    noun = _THEME_NOUNS.get(theme, _THEME_NOUNS["General regulatory coverage"])
    citation = _THEME_CITATIONS.get(theme, _THEME_CITATIONS["General regulatory coverage"])
    return [
        {
            "question_id": f"{slug.upper()}_PART_WHICH_STAGE",
            "question": f"Which stage of the {noun} is still incomplete?",
            "options": _theme_stage_options(theme),
            "question_type": "Single Select",
            "rationale": (
                f"Pinpoints the specific stage that still has a gap so the remediation plan and "
                f"scoring can target it directly against {citation}."
            ),
            "branch_rule_id": f"{slug}__partial__which_stage",
            "scoring_weight": 3,
        },
        {
            "question_id": f"{slug.upper()}_PART_TEAMS",
            "question": (
                f"Which teams are affected by the incomplete {noun} (and which are fully covered)?"
            ),
            "options": [
                "Only critical / important functions",
                "Critical functions + a subset of supporting teams",
                "Most teams except a known exclusion list",
                "All in-scope teams (uniform partial state)",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Identifies whether the partial state is scoped (e.g. only critical functions) or "
                "uniform — drives the remediation prioritisation."
            ),
            "branch_rule_id": f"{slug}__partial__teams",
            "scoring_weight": 2,
        },
        {
            "question_id": f"{slug.upper()}_PART_EVIDENCE",
            "question": (
                f"What evidence already exists for the parts of the {noun} that ARE complete?"
            ),
            "options": [
                "Operational log + management report",
                "Policy / procedure only",
                "Informal / SME knowledge",
                "Evidence not yet collated",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Even partial coverage scores meaningfully higher when the completed parts can be "
                "evidenced. Tests whether the partial path is auditable."
            ),
            "branch_rule_id": f"{slug}__partial__evidence",
            "scoring_weight": 2,
        },
    ]


def _not_implemented_specs(theme: str) -> List[Dict[str, Any]]:
    slug = _theme_slug(theme)
    noun = _THEME_NOUNS.get(theme, _THEME_NOUNS["General regulatory coverage"])
    citation = _THEME_CITATIONS.get(theme, _THEME_CITATIONS["General regulatory coverage"])
    return [
        {
            "question_id": f"{slug.upper()}_NOT_BLOCKER",
            "question": f"Why has implementation of the {noun} not started yet?",
            "options": [
                "Awaiting regulatory clarification",
                "No funded programme yet",
                "Awaiting executive sponsorship",
                "Dependency on a third party / vendor",
                "Resource / capability gap",
                "Lower priority than other workstreams",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                f"Identifies the root blocker preventing implementation against {citation} so the "
                f"recommendations engine can pair it with a targeted unblock action."
            ),
            "branch_rule_id": f"{slug}__not__blocker",
            "scoring_weight": 3,
        },
        {
            "question_id": f"{slug.upper()}_NOT_PROJECT",
            "question": f"Is a project to implement the {noun} planned?",
            "options": [
                "Yes — funded and scheduled",
                "Yes — proposed but not yet funded",
                "Under consideration",
                "No project planned",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Tests whether the gap has moved from awareness to a funded programme — this is the "
                "single biggest predictor of regulatory readiness within 12 months."
            ),
            "branch_rule_id": f"{slug}__not__project",
            "scoring_weight": 3,
        },
        {
            "question_id": f"{slug.upper()}_NOT_OWNERSHIP",
            "question": f"Has accountable ownership been assigned for delivering the {noun}?",
            "options": [
                "Named accountable owner",
                "Shared ownership",
                "Informal owner",
                "No owner assigned",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Without a named owner the project will not start; this question quantifies the "
                "governance gap and feeds Agent 4's recommendation severity."
            ),
            "branch_rule_id": f"{slug}__not__ownership",
            "scoring_weight": 3,
        },
        {
            "question_id": f"{slug.upper()}_NOT_SUPPORT",
            "question": (
                f"What external support or budget would be required to start implementing the {noun}?"
            ),
            "options": [
                "No external support required",
                "Education only",
                "Assessment support",
                "Implementation support",
                "Managed service support",
                "Budgeted opportunity",
                "Unknown",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Captures the support / budget signal so Agent 4 can pair the gap with the right "
                "delivery model (advisory vs implementation vs managed service)."
            ),
            "branch_rule_id": f"{slug}__not__support",
            "scoring_weight": 2,
        },
    ]


def _not_applicable_specs(theme: str) -> List[Dict[str, Any]]:
    slug = _theme_slug(theme)
    noun = _THEME_NOUNS.get(theme, _THEME_NOUNS["General regulatory coverage"])
    return [
        {
            "question_id": f"{slug.upper()}_NA_JUSTIFY",
            "question": (
                f"On what basis is the {noun} treated as Not Applicable for this scope? "
                f"(Selected answer will be recorded as a scope-out justification.)"
            ),
            "options": [
                "Out of scope per regulatory proportionality",
                "Outsourced to a third party with full control transfer",
                "Covered by a parent / group entity",
                "Pending scope decision",
                "Other — see comments",
            ],
            "question_type": "Single Select",
            "rationale": (
                "Records why the requirement is scoped out so the assessment remains auditable. "
                "Not Applicable answers are excluded from scoring."
            ),
            "branch_rule_id": f"{slug}__na__justify",
            "scoring_weight": 1,
        },
    ]


def _unknown_specs(theme: str) -> List[Dict[str, Any]]:
    slug = _theme_slug(theme)
    noun = _THEME_NOUNS.get(theme, _THEME_NOUNS["General regulatory coverage"])
    return [
        {
            "question_id": f"{slug.upper()}_UNK_WHO",
            "question": (
                f"Who is best placed to confirm the current state of the {noun} for this scope?"
            ),
            "options": _theme_who_options(theme),
            "question_type": "Single Select",
            "rationale": (
                "When the baseline status is Unknown, the engine routes to identifying who can "
                "authoritatively answer it before any deeper readiness questions are asked."
            ),
            "branch_rule_id": f"{slug}__unknown__who",
            "scoring_weight": 2,
        },
    ]


# Per-theme answer dictionaries kept small but materially different so the
# generated questions read like the example questionnaire rather than the
# generic signal-banded follow-ups.

def _theme_stage_options(theme: str) -> List[str]:
    return {
        "Incident reporting": [
            "Detection", "Classification", "Initial notification",
            "Intermediate notification", "Final report", "Lessons learned", "Unknown",
        ],
        "Resilience testing": [
            "Test plan", "Scenario design", "Execution", "Findings remediation",
            "Independent review", "Reporting to the management body", "Unknown",
        ],
        "Third-party risk": [
            "Vendor inventory", "Contract due diligence", "DORA clause coverage",
            "Exit strategy", "Sub-outsourcing oversight", "Concentration risk monitoring", "Unknown",
        ],
        "Security and access": [
            "Privileged access reviews", "Encryption coverage",
            "Vulnerability management", "SIEM event monitoring",
            "Identity governance", "Patching cadence", "Unknown",
        ],
        "Governance": [
            "Policy drafting", "Executive review", "Management-body approval",
            "Communication / training", "Periodic reaffirmation", "Unknown",
        ],
        "Data and evidence": [
            "Evidence dictionary", "Lineage mapping", "Owner assignment",
            "Retention schedule", "Quality controls", "Audit-trail wiring", "Unknown",
        ],
        "Reporting": [
            "KRI/KPI definition", "Data feeds", "Dashboard / pack build",
            "Management-body cadence", "Action tracking", "Unknown",
        ],
        "ICT risk management": [
            "Risk identification", "Risk assessment", "Treatment plan",
            "Risk acceptance / sign-off", "Re-assessment cadence", "Unknown",
        ],
    }.get(theme, [
        "Process design", "Documentation", "Operational rollout",
        "Evidence capture", "Periodic review", "Unknown",
    ])


def _theme_who_options(theme: str) -> List[str]:
    return {
        "Incident reporting": [
            "ICT Incident Response Lead", "Chief Information Security Officer",
            "Operational Resilience Lead", "Compliance Officer",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Resilience testing": [
            "Operational Resilience Lead", "Business Continuity Manager",
            "Head of Technology Risk", "Internal Audit",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Third-party risk": [
            "Head of Procurement / Vendor Management", "Legal counsel",
            "Operational Resilience Lead", "Chief Information Officer",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Security and access": [
            "Chief Information Security Officer", "Head of Identity & Access",
            "Head of Vulnerability Management", "Internal Audit",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Governance": [
            "Chief Risk Officer", "Chief Compliance Officer",
            "Head of Governance / Company Secretary", "Internal Audit",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Data and evidence": [
            "Chief Data Officer", "Data Governance Lead",
            "Operational Resilience Lead", "Internal Audit",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "Reporting": [
            "Head of Risk Reporting", "Operational Resilience Lead",
            "Chief Information Officer", "Company Secretary",
            "No clear owner — needs to be assigned", "Unknown",
        ],
        "ICT risk management": [
            "Head of Technology Risk", "Chief Information Security Officer",
            "Chief Information Officer", "Internal Audit",
            "No clear owner — needs to be assigned", "Unknown",
        ],
    }.get(theme, [
        "Business owner", "Technology owner", "Compliance owner", "Risk owner",
        "No clear owner — needs to be assigned", "Unknown",
    ])


# ---------------------------------------------------------------------------
# Canonical answer label aliases
# ---------------------------------------------------------------------------
#
# Older questionnaires (v10/v11) used the coverage option family
# ("Complete", "Mostly complete", "Partially complete", "Not started",
# "Not applicable") instead of the v13 canonical implementation-status family.
# We alias them so a saved package still routes correctly.

_LEGACY_ANSWER_ALIASES = {
    "Complete": "Fully Implemented",
    "Mostly complete": "Partially Implemented",
    "Partially complete": "Partially Implemented",
    "Partially": "Partially Implemented",
    "Not started": "Not Implemented",
    "Not applicable": "Not Applicable",
}


def _canonical_answer(label: str) -> str:
    return _LEGACY_ANSWER_ALIASES.get(label, label)


# ---------------------------------------------------------------------------
# BRANCH_LIBRARY — generated dynamically from the factories above
# ---------------------------------------------------------------------------

# Themes for which we register branches today. Adding a new theme is a
# one-line change here and (optionally) an entry in _THEME_NOUNS /
# _theme_stage_options / _theme_who_options for nicer wording.
_SUPPORTED_THEMES = (
    "Governance",
    "ICT risk management",
    "Incident reporting",
    "Resilience testing",
    "Third-party risk",
    "Security and access",
    "Data and evidence",
    "Reporting",
)


def _build_library() -> Dict[BranchKey, List[Dict[str, Any]]]:
    lib: Dict[BranchKey, List[Dict[str, Any]]] = {}
    for theme in _SUPPORTED_THEMES:
        lib[("DORA", theme, "coverage", "Fully Implemented")] = _fully_implemented_specs(theme)
        lib[("DORA", theme, "coverage", "Partially Implemented")] = _partially_implemented_specs(theme)
        lib[("DORA", theme, "coverage", "Not Implemented")] = _not_implemented_specs(theme)
        lib[("DORA", theme, "coverage", "Not Applicable")] = _not_applicable_specs(theme)
        lib[("DORA", theme, "coverage", "Unknown")] = _unknown_specs(theme)
    return lib


BRANCH_LIBRARY: Dict[BranchKey, List[Dict[str, Any]]] = _build_library()


# ---------------------------------------------------------------------------
# Lookup helpers (the engine only ever calls these)
# ---------------------------------------------------------------------------

def _norm(value: Optional[str]) -> str:
    return (value or "").strip()


def lookup_branch(
    regulation: Optional[str],
    theme: Optional[str],
    question_kind: Optional[str],
    answer_label: Optional[str],
) -> List[Dict[str, Any]]:
    """Return the branch follow-up specs for the (reg, theme, kind, answer) key.

    Returns an empty list when no specific branch is registered — callers
    should then fall back to the GenAI option-aware generator, and finally to
    the generic ``dynamic_followups`` family.
    """
    reg = _norm(regulation).upper() or "DORA"
    canonical_answer = _canonical_answer(_norm(answer_label))
    key: BranchKey = (reg, _norm(theme), _norm(question_kind), canonical_answer)
    specs = BRANCH_LIBRARY.get(key)
    if specs:
        return [dict(s) for s in specs]
    return []


def available_branch_keys(regulation: Optional[str] = None) -> List[BranchKey]:
    """List the registered branch keys (for diagnostics / UI / tests)."""
    reg = _norm(regulation).upper() if regulation else None
    if reg is None:
        return list(BRANCH_LIBRARY.keys())
    return [k for k in BRANCH_LIBRARY if k[0] == reg]


def has_branch(regulation: Optional[str], theme: Optional[str],
               question_kind: Optional[str], answer_label: Optional[str]) -> bool:
    return bool(lookup_branch(regulation, theme, question_kind, answer_label))


__all__ = [
    "BRANCH_LIBRARY",
    "BranchKey",
    "available_branch_keys",
    "has_branch",
    "lookup_branch",
]
