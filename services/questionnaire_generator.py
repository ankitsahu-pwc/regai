"""BRD/FRD to regulatory readiness questionnaire generator.

Refactored from ``generate_brd_questionnaire_streamlit_v11.py``. The dataclasses,
keyword taxonomies, deterministic question synthesis, deduplication, scoring
package and Excel writer are lifted verbatim. The DOCX-reading helper is
rewired to use :mod:`utils.docx_parser`, removing the duplicated body-iteration
trick from the original file.

Two new top-level entry points are added:

- :func:`build_questionnaire_package` accepts a DOCX path/bytes and returns the
  questionnaire package dict.
- :func:`build_package_from_report` accepts a Phase 4
  :class:`~services.brd_frd_generator.DoraDetailedBRD` in-memory model and
  builds the package without a DOCX round-trip. This is the closed loop that
  enables Page 1's "Generate BRD/FRD from regulation" path.

The package dict's shape is unchanged from the original v11 output, so
``utils.json_utils.validate_package_schema`` and the existing
``sample_data/dora_questionnaire_package_v10.json`` remain valid contracts.
"""

from __future__ import annotations

import os
import re as _re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from utils.docx_parser import (
    DocxSource,
    clean_text,
    iter_sectioned_tables,
    normalise_header,
)

# ---------------------------------------------------------------------------
# Tunables (env-overridable, identical defaults to v11)
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = int(os.getenv("QUESTION_CONFIDENCE_FLOOR", "90"))
# v13.1 — the package-confidence metric is now a **content-correctness** grade,
# so the historic 90 floor has been retired by default. Set
# OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR=90 in the environment to restore the
# legacy behaviour. Set PACKAGE_CONFIDENCE_MODE=structural to revert the
# whole metric to the v11/v12 structural-completeness grade.
OVERALL_CONFIDENCE_FLOOR = int(os.getenv("OVERALL_QUESTIONNAIRE_CONFIDENCE_FLOOR", "0"))
PACKAGE_CONFIDENCE_MODE = os.getenv("PACKAGE_CONFIDENCE_MODE", "content").strip().lower()
MAX_AREA_FUNCTION_PAIRS = int(os.getenv("MAX_AREA_FUNCTION_PAIRS", "40"))
MIN_FREE_TEXT = int(os.getenv("MIN_FREE_TEXT_QUESTIONS", "5"))
MAX_FREE_TEXT = int(os.getenv("MAX_FREE_TEXT_QUESTIONS", "10"))

# v13.1 — the keys that must be populated in every question's
# ``explainability`` bundle for the content-correctness grade to credit it
# as a fully-traceable question.
EXPLAINABILITY_REQUIRED_KEYS = (
    "regulation",
    "regulator",
    "article",
    "obligation_id",
    "brd_requirement_ids",
    "business_function",
    "control_objective",
    "reason",
    "expected_evidence",
    "risk_if_negative",
)

# ---------------------------------------------------------------------------
# Regulatory taxonomy (lifted verbatim)
# ---------------------------------------------------------------------------

REGULATORY_TAXONOMY = {
    "DORA": {
        "official_source": "Regulation (EU) 2022/2554 and related DORA RTS/ITS guidance",
        "pillars": [
            "Governance and organisation",
            "ICT risk management framework",
            "ICT systems, protocols and tools",
            "Identification, protection, prevention, detection, response and recovery",
            "Backup, restoration, recovery and communication",
            "Incident management, classification and reporting",
            "Digital operational resilience testing",
            "ICT third-party risk management",
            "Key contractual provisions and exit planning",
            "Information security, access control, data protection and auditability",
            "Management reporting and evidence traceability",
        ],
        "article_hints": {
            "governance": "DORA Article 5",
            "framework": "DORA Article 6",
            "identification": "DORA Article 8",
            "protection": "DORA Article 9",
            "detection": "DORA Article 10",
            "response": "DORA Article 11",
            "backup": "DORA Article 12",
            "incident": "DORA Articles 17-20",
            "testing": "DORA Articles 24-27",
            "third_party": "DORA Articles 28-30",
            "contracts": "DORA Article 30",
        },
    }
}

AREA_KEYWORDS = {
    "Front Office": ["front office", "client", "customer", "trading", "sales", "market"],
    "Middle Office": ["middle office", "risk control", "valuation", "trade support", "risk"],
    "Back Office": ["back office", "settlement", "reconciliation", "processing", "operations"],
    "Regulatory Reporting & Financial Reporting": ["report", "dashboard", "governance pack", "management body", "kpi", "kri"],
    "Business Structure & Functions": ["business function", "critical function", "important function", "service mapping", "institution"],
    "Firm Type / Client Type": ["tier", "entity", "financial services", "proportionality", "institution"],
    "Operating Model": ["operating model", "workflow", "role", "responsibil", "owner", "attestation"],
    "Risk & Controls framework": ["risk", "control", "framework", "vulnerability", "security", "risk acceptance"],
    "Governance Model": ["governance", "management body", "board", "approval", "decision"],
    "Internal Compliances": ["compliance", "audit", "policy", "procedure", "legal"],
    "Third Party Risk Management / Dependency": ["third-party", "third party", "vendor", "provider", "subcontract", "contract", "exit"],
    "Programme Maturity / Programme Ownership": ["programme", "maturity", "readiness", "roadmap", "phase", "implementation"],
    "Program Sponsorship / Budget Planning": ["budget", "sponsor", "resource", "funding", "buying", "support"],
    "People, Policies & Processes": ["training", "people", "policy", "process", "procedure", "lessons learned"],
    "IT, Systems & Technology": ["ict", "system", "application", "infrastructure", "cmdb", "itsm", "siem", "iam"],
    "Data Reporting & Governance": ["data", "metadata", "lineage", "quality", "evidence", "timestamp", "dictionary"],
    "IT Security / Cyber Security": ["security", "cyber", "access", "privileged", "vulnerability", "encryption", "intrusion"],
    "Legal (Contracts, Readiness & Agreements)": ["legal", "contract", "clause", "termination", "audit rights", "access rights", "jurisdiction"],
    "HR": ["hr", "training", "resource", "sme", "people"],
    "High Impact Pain Points": ["gap", "pain", "manual", "fragment", "incomplete", "overdue", "unclear", "inconsistent", "constraint"],
}

FUNCTION_KEYWORDS = {
    "Execution / Client Activity": ["front office", "client", "trading", "customer", "market"],
    "Risk Management": ["risk", "control", "risk appetite", "risk assessment", "vulnerability"],
    "Compliance & Legal": ["compliance", "legal", "policy", "regulatory", "contract", "audit"],
    "Technology / IT Operations": ["ict", "system", "application", "infrastructure", "cmdb", "itsm", "backup", "restore"],
    "Cyber Security": ["security", "cyber", "siem", "vulnerability", "access", "encryption"],
    "Business Continuity / Resilience": ["continuity", "recovery", "resilience", "testing", "backup", "restore"],
    "Incident Management": ["incident", "classification", "notification", "root cause", "response"],
    "Vendor / Third-Party Management": ["third-party", "third party", "vendor", "provider", "subcontract", "exit"],
    "Data Governance / Reporting": ["data", "metadata", "lineage", "dashboard", "report", "kpi", "kri"],
    "Internal Audit / Assurance": ["audit", "evidence", "traceability", "approval", "attestation"],
    "Operations / Settlement": ["operations", "settlement", "processing", "reconciliation", "back office"],
    "Programme Management": ["programme", "implementation", "roadmap", "budget", "sponsor", "maturity"],
    "Human Resources / Training": ["hr", "training", "people", "resource", "sme"],
}

DEFAULT_OPTIONS = {
    "maturity": ["Not started", "Ad hoc", "Defined", "Implemented", "Measured / Optimised", "Not applicable", "Unknown"],
    "yes_no_partial": ["Yes", "Partially", "No", "Not applicable", "Unknown"],
    "coverage": ["Complete", "Mostly complete", "Partially complete", "Not started", "Not applicable", "Unknown"],
    # v13 — canonical implementation-status family used as the L1 root for
    # every impact pair so the funnel matches the product brief's example:
    #   "Is the <theme> process implemented?"
    #   - Fully Implemented      -> evidence/validation branch (skips detail)
    #   - Partially Implemented  -> partial-implementation branch
    #   - Not Implemented        -> blocker/ownership branch
    #   - Not Applicable         -> scoped-out (excluded from scoring)
    "implementation_status": [
        "Fully Implemented",
        "Partially Implemented",
        "Not Implemented",
        "Not Applicable",
        "Unknown",
    ],
    "risk_level": ["Low", "Medium", "High", "Critical", "Unknown"],
    "ownership": ["Named accountable owner", "Shared ownership", "Informal owner", "No owner assigned", "Unknown"],
    "evidence": ["Policy / procedure", "Workflow record", "System report", "Dashboard", "Attestation", "Audit trail", "Contract evidence", "No evidence available"],
    "support": ["No external support required", "Education only", "Assessment support", "Implementation support", "Managed service support", "Budgeted opportunity", "Unknown"],
}

# Canonical answer labels for the implementation-status family. Used as
# branch_registry keys and as the trigger_answers for L2 questions.
IMPLEMENTATION_STATUS_LABELS = (
    "Fully Implemented",
    "Partially Implemented",
    "Not Implemented",
    "Not Applicable",
    "Unknown",
)

# Theme-specific option families used by the deep-dive questions injected into
# each impact pair's funnel. They intentionally include positive and negative
# signals already recognised by services.scoring_engine so the live cockpit can
# route follow-ups consistently.
THEME_OPTIONS = {
    "incident_reporting": [
        "Documented, tested, and meets the regulatory deadline",
        "Documented but not yet tested end-to-end",
        "Partial — manual workaround within deadline",
        "Partial — risk of missing the deadline",
        "Not implemented",
        "Not applicable",
        "Unknown",
    ],
    "third_party": [
        "All required clauses present and Legal-validated",
        "Most clauses present, awaiting Legal sign-off",
        "Some clauses missing or under negotiation",
        "Clauses not yet reviewed by Legal",
        "No contract evidence available",
        "Not applicable",
        "Unknown",
    ],
    "resilience_testing": [
        "Tested within the regulatory window with successful results",
        "Tested but with open findings being remediated",
        "Scheduled but not yet executed",
        "Not scheduled",
        "Not applicable",
        "Unknown",
    ],
    "security_access": [
        "Implemented with periodic review and SIEM coverage",
        "Implemented but reviews are ad hoc",
        "Partially implemented for in-scope systems",
        "Designed but not yet implemented",
        "Not implemented",
        "Not applicable",
        "Unknown",
    ],
    "governance": [
        "Approved by the management body with traceable evidence",
        "Approved at executive level, awaiting board ratification",
        "Drafted but not formally approved",
        "Not yet drafted",
        "Not applicable",
        "Unknown",
    ],
    "data_evidence": [
        "Evidence dictionary complete with owners and retention",
        "Evidence captured but no formal dictionary",
        "Evidence partial or manually compiled",
        "Evidence missing or inconsistent",
        "Not applicable",
        "Unknown",
    ],
    "reporting": [
        "Automated dashboard with KRIs/KPIs reviewed periodically",
        "Manual report produced on schedule",
        "Ad hoc reporting only",
        "No reporting in place",
        "Not applicable",
        "Unknown",
    ],
}

# Maps the `theme` field (output of :func:`infer_themes`) to its question key.
_THEME_TO_KEY = {
    "Incident reporting": "incident_reporting",
    "Third-party risk": "third_party",
    "Resilience testing": "resilience_testing",
    "Security and access": "security_access",
    "Governance": "governance",
    "Data and evidence": "data_evidence",
    "Reporting": "reporting",
}

# ---------------------------------------------------------------------------
# Option helpers — options may now be plain strings OR dicts of the shape:
#   {"label": str, "next_question_ids": [str], "skip_question_ids": [str],
#    "score_value": int|float, "impact_value": ..., "risk_value": ...,
#    "branch_rule_id": str}
# Every helper below accepts both shapes so the rest of the pipeline is
# fully backward-compatible with the legacy ``List[str]`` form.
# ---------------------------------------------------------------------------

def option_label(option: Any) -> str:
    """Return the human-facing label for either a string option or a dict option."""
    if isinstance(option, Mapping):
        return str(option.get("label", "")).strip()
    return str(option).strip()


def option_labels(options: Sequence[Any]) -> List[str]:
    return [option_label(o) for o in options or []]


def option_metadata(options: Sequence[Any], label: str) -> Dict[str, Any]:
    """Return the option dict matching ``label``, or an empty dict for plain options."""
    for opt in options or []:
        if isinstance(opt, Mapping) and option_label(opt) == label:
            return dict(opt)
    return {}


ANSWER_SCORES = {
    "Yes": 100, "Complete": 100, "Measured / Optimised": 100, "Implemented": 90, "Mostly complete": 85,
    "Defined": 75, "Partially": 55, "Partially complete": 55, "Ad hoc": 35, "Not started": 0,
    "No": 0, "No owner assigned": 0, "No evidence available": 0, "Critical": 15, "High": 35,
    "Medium": 65, "Low": 90, "Named accountable owner": 95, "Shared ownership": 70,
    "Informal owner": 40, "Not applicable": None, "Unknown": 25,
    # v13 — canonical implementation-status family (mirrors coverage scoring
    # so the readiness % stays comparable to historical assessments).
    "Fully Implemented": 100,
    "Partially Implemented": 55,
    "Not Implemented": 0,
    "Not Applicable": None,
}


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Requirement:
    source_section: str
    source_id: str
    normalized_id: str
    category: str
    requirement: str
    detail: str
    alignment: str
    priority: str
    acceptance: str
    confidence: int
    themes: List[str] = field(default_factory=list)
    #: Source citations attached to this requirement. Each entry is a plain
    #: dict with the ``SourceReference`` shape produced by
    #: :mod:`services.source_traceability`. Empty when no live source could
    #: be anchored to this requirement (UI surfaces this gap explicitly
    #: instead of fabricating a citation).
    source_references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ImpactPair:
    area: str
    function: str
    requirement_ids: List[str]
    regulatory_basis: str
    confidence: int


@dataclass
class Question:
    question_id: str
    area: str
    function: str
    question_type: str
    question: str
    options: List[Any]  # list of str OR list of dict (option metadata)
    mapped_requirement_ids: List[str]
    regulatory_basis: str
    confidence: int
    scoring_weight: int
    funnel_parent_id: str = ""
    trigger_answers: List[str] = field(default_factory=list)
    rationale: str = ""
    is_free_text: bool = False
    # --- v12 adaptive-branching extensions ------------------------------------
    # All new fields are optional and default-safe so older JSON packages still
    # load via ``Question(**q)`` (extra absent keys are fine; we never read them
    # without a default). The engine treats missing values as "use generic
    # routing" so behaviour is unchanged for legacy questions.
    branch_theme: str = ""          # e.g. "Incident reporting"
    branch_rule_id: str = ""        # set on dynamically-generated branch questions
    source_parent_id: str = ""      # the *original* base question that opened the branch
    dynamic_depth: int = 0          # 0 = base question, 1..N = dynamic depth
    # --- v13 explainability bundle -------------------------------------------
    # Structured traceability metadata so the UI and exports can answer
    # "Why am I being asked this?" without parsing free-form rationale text.
    # Schema (every key optional, missing keys treated as empty):
    #   regulation              str   (e.g. "DORA")
    #   regulator               str   (e.g. "ESMA / EBA")
    #   article                 str   (e.g. "DORA Article 17-20")
    #   obligation_id           str   (e.g. "OBL-014")
    #   brd_requirement_ids     [str] (e.g. ["BR-PRO-001", "BR-DAT-004"])
    #   rtm_trace_ids           [str] (e.g. ["TR-0014"])
    #   business_function       str   (e.g. "Incident Management")
    #   control_objective       str   (e.g. "Meet 24h initial notification window")
    #   reason                  str   (why this question exists)
    #   expected_evidence       str   (artefacts that would substantiate a positive answer)
    #   risk_if_negative        str   (consequence of a negative answer)
    explainability: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers (lifted verbatim)
# ---------------------------------------------------------------------------

def clamp_confidence(value, floor: int = CONFIDENCE_FLOOR) -> int:
    try:
        score = int(str(value).replace("%", "").strip())
    except Exception:
        score = 94
    return max(floor, min(100, score))


def infer_themes(text: str) -> List[str]:
    lower = text.lower()
    themes: List[str] = []
    checks = [
        ("Governance", ["governance", "board", "management body", "approval"]),
        ("ICT risk management", ["ict risk", "framework", "risk management", "control"]),
        ("Incident reporting", ["incident", "classification", "notification", "reporting"]),
        ("Resilience testing", ["resilience", "testing", "backup", "restore", "recovery", "continuity"]),
        ("Third-party risk", ["third-party", "third party", "vendor", "provider", "contract", "subcontract", "exit"]),
        ("Data and evidence", ["data", "inventory", "lineage", "metadata", "evidence", "timestamp", "quality"]),
        ("Security and access", ["security", "access", "vulnerability", "encryption", "siem", "iam"]),
        ("Reporting", ["dashboard", "report", "kpi", "kri", "governance pack"]),
    ]
    for name, keys in checks:
        if any(k in lower for k in keys):
            themes.append(name)
    return themes or ["General regulatory coverage"]


def _section_prefix(section_label: str, source_id: str) -> str:
    section_l = (section_label or "").lower()
    source_l = (source_id or "").lower()
    if "process" in section_l or source_l.startswith("br-pro"):
        return "BR-PRO"
    if "data" in section_l or source_l.startswith("br-dat"):
        return "BR-DAT"
    if "report" in section_l or source_l.startswith("br-rep"):
        return "BR-REP"
    if "non-functional" in section_l or source_l.startswith("nfr"):
        return "NFR"
    if "functional" in section_l or source_l.startswith("fr"):
        return "FR"
    return "REQ"


# ---------------------------------------------------------------------------
# Requirement extraction from DOCX (now uses utils.docx_parser)
# ---------------------------------------------------------------------------

_REQUIRED_HEADERS = ("id", "category", "requirement", "detailedrequirement",
                     "doraalignment", "priority", "acceptancecriteria")


def read_docx_requirements(source: DocxSource) -> List[Requirement]:
    """Extract Requirement records from a BRD/FRD DOCX file or bytes.

    Equivalent to the original v11 ``read_docx_requirements`` but the
    document-iteration boilerplate has been replaced by a call to
    :func:`utils.docx_parser.iter_sectioned_tables`.
    """
    requirements: List[Requirement] = []
    counters: Dict[str, int] = defaultdict(int)
    for section_label, rows in iter_sectioned_tables(source):
        if not rows:
            continue
        headers = [normalise_header(h) for h in rows[0]]
        if not all(h in headers for h in _REQUIRED_HEADERS):
            continue
        idx = {h: i for i, h in enumerate(headers)}
        conf_idx = idx.get("aiconfidence")
        for row in rows[1:]:
            if len(row) < len(headers):
                row = list(row) + [""] * (len(headers) - len(row))
            source_id = row[idx["id"]]
            title = row[idx["requirement"]]
            detail = row[idx["detailedrequirement"]]
            if not (source_id or title or detail):
                continue
            prefix = _section_prefix(section_label, source_id)
            counters[prefix] += 1
            combined = " ".join([
                section_label, source_id, title, detail,
                row[idx["doraalignment"]], row[idx["acceptancecriteria"]],
            ])
            requirements.append(Requirement(
                source_section=section_label,
                source_id=source_id,
                normalized_id=f"{prefix}-{counters[prefix]:03d}",
                category=row[idx["category"]],
                requirement=title,
                detail=detail,
                alignment=row[idx["doraalignment"]],
                priority=row[idx["priority"]],
                acceptance=row[idx["acceptancecriteria"]],
                confidence=clamp_confidence(row[conf_idx] if conf_idx is not None else 95),
                themes=infer_themes(combined),
            ))
    return requirements


def requirements_from_report(
    report: Any,
    source_refs_by_item: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Requirement]:
    """Convert a Phase-4 DoraDetailedBRD into the Requirement list used here.

    Accepts the in-memory Pydantic ``DoraDetailedBRD`` model so the closed loop
    GenAI/offline -> questionnaire works without a DOCX round-trip.

    ``source_refs_by_item`` is the optional ``{REQ:<id> -> [SourceReference
    dict]}`` map produced by :func:`services.source_traceability.attach_source_references`
    and carried on ``BRDArtifact.metadata``. When supplied each returned
    ``Requirement`` carries the citations of the BRD row it was derived from
    so downstream UI / questionnaire / scoring layers can surface
    traceability without re-running the matcher.
    """
    section_map: List[Tuple[str, str, Any]] = [
        ("7.1 Process Requirements", "BR-PRO", report.process_business_requirements),
        ("7.2 Data Requirements", "BR-DAT", report.data_business_requirements),
        ("7.3 Reporting Requirements", "BR-REP", report.reporting_business_requirements),
        ("8. Functional Requirements", "FR", report.functional_requirements),
        ("10. Non-Functional Requirements", "NFR", report.non_functional_requirements),
    ]
    refs_map = source_refs_by_item or {}
    requirements: List[Requirement] = []
    counters: Dict[str, int] = defaultdict(int)
    for section_label, prefix, req_section in section_map:
        for item in req_section.items:
            counters[prefix] += 1
            combined = " ".join([
                section_label, item.id, item.category, item.requirement,
                item.detailed_requirement, item.dora_alignment, item.acceptance_criteria,
            ])
            req_refs = list(refs_map.get(f"REQ:{item.id}", []))
            requirements.append(Requirement(
                source_section=section_label,
                source_id=item.id,
                normalized_id=f"{prefix}-{counters[prefix]:03d}",
                category=item.category,
                requirement=item.requirement,
                detail=item.detailed_requirement,
                alignment=item.dora_alignment,
                priority=item.priority,
                acceptance=item.acceptance_criteria,
                confidence=clamp_confidence(item.confidence_level),
                themes=infer_themes(combined),
                source_references=req_refs,
            ))
    return requirements


# ---------------------------------------------------------------------------
# Impact-pair derivation
# ---------------------------------------------------------------------------

def score_keywords(text: str, keyword_map: Dict[str, List[str]]) -> Counter:
    lower = text.lower()
    scores: Counter = Counter()
    for label, keys in keyword_map.items():
        for key in keys:
            if key in lower:
                scores[label] += 1
    return scores


def impacted_labels_for_requirement(req: Requirement, keyword_map: Dict[str, List[str]], default: str) -> List[str]:
    text = " ".join([req.source_section, req.category, req.requirement, req.detail, req.alignment, req.acceptance])
    scores = score_keywords(text, keyword_map)
    if not scores:
        return [default]
    max_score = max(scores.values())
    selected = [k for k, v in scores.items() if v >= max(1, max_score - 1)]
    return selected[:3]


def regulatory_basis_for(reqs: Sequence[Requirement], regulation: str) -> str:
    alignments = [r.alignment for r in reqs if r.alignment]
    if alignments:
        return " | ".join(list(dict.fromkeys(alignments))[:3])
    return REGULATORY_TAXONOMY.get(regulation.upper(), {}).get("official_source", regulation)


def derive_impact_pairs(requirements: Sequence[Requirement], regulation: str) -> List[ImpactPair]:
    pair_to_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    req_by_id = {r.normalized_id: r for r in requirements}
    for req in requirements:
        areas = impacted_labels_for_requirement(req, AREA_KEYWORDS, "Risk & Controls framework")
        functions = impacted_labels_for_requirement(req, FUNCTION_KEYWORDS, "Compliance & Legal")
        for area in areas:
            for function in functions:
                pair_to_ids[(area, function)].append(req.normalized_id)
    ranked = sorted(pair_to_ids.items(), key=lambda kv: (-len(set(kv[1])), kv[0][0], kv[0][1]))[:MAX_AREA_FUNCTION_PAIRS]
    pairs: List[ImpactPair] = []
    for (area, function), ids in ranked:
        unique_ids = list(dict.fromkeys(ids))
        reqs = [req_by_id[i] for i in unique_ids if i in req_by_id]
        avg_conf = round(sum(r.confidence for r in reqs) / max(1, len(reqs)))
        basis = regulatory_basis_for(reqs, regulation)
        pairs.append(ImpactPair(area, function, unique_ids, basis, clamp_confidence(avg_conf)))
    return pairs


# ---------------------------------------------------------------------------
# Question synthesis
# ---------------------------------------------------------------------------

def select_option_family(reqs: Sequence[Requirement], area: str, function: str) -> Tuple[str, List[str]]:
    text = " ".join([area, function] + [r.category + " " + r.requirement + " " + r.detail for r in reqs]).lower()
    if any(k in text for k in ["owner", "role", "responsibil", "attestation", "approval"]):
        return "Single Select", DEFAULT_OPTIONS["ownership"]
    if any(k in text for k in ["evidence", "audit", "traceability", "metadata"]):
        return "Multi Select", DEFAULT_OPTIONS["evidence"]
    if any(k in text for k in ["risk", "severity", "critical", "vulnerability", "incident"]):
        return "Single Select", DEFAULT_OPTIONS["risk_level"]
    if any(k in text for k in ["programme", "maturity", "readiness", "framework"]):
        return "Single Select", DEFAULT_OPTIONS["maturity"]
    if any(k in text for k in ["support", "budget", "sponsor"]):
        return "Single Select", DEFAULT_OPTIONS["support"]
    return "Single Select", DEFAULT_OPTIONS["coverage"]


def _short_requirement_focus(reqs: Sequence[Requirement]) -> str:
    if not reqs:
        return "the mapped regulatory requirement set"
    primary = reqs[0]
    title = clean_text(primary.requirement)
    if len(title) > 72:
        title = title[:69].rstrip() + "..."
    return f"{title} ({primary.normalized_id})"


# ---------------------------------------------------------------------------
# Specificity helpers — extract concrete anchors from a requirement so that
# the generated questions reference the *actual* behaviour, metric, evidence
# expectation and regulatory article instead of generic placeholders.
# ---------------------------------------------------------------------------

_ARTICLE_RE = _re.compile(
    r"\b(?:Article|Art\.?)\s*([0-9]{1,3}(?:\s*\([a-z0-9]+\))?(?:\s*\.\s*[0-9]+)?)\b",
    _re.IGNORECASE,
)
_REGULATION_RE = _re.compile(
    r"\b(DORA|MiFID(?:\s*II)?|GDPR|EMIR|SFTR|MAR|PSD2|EU\s*2022\s*/\s*2554|NIS\s*2)\b",
    _re.IGNORECASE,
)
_METRIC_RE = _re.compile(
    r"(\b\d+(?:\.\d+)?\s*(?:%|percent|hours?|hrs?|minutes?|mins?|days?|seconds?|secs?|years?|months?|business\s+days?)\b"
    r"|\b(?:RTO|RPO|SLA|MTTR|MTBF|TLPT|KRI|KPI)\b)",
    _re.IGNORECASE,
)
_VERB_HINTS = (
    "establish", "implement", "maintain", "monitor", "report", "classify",
    "notify", "escalate", "test", "review", "approve", "document",
    "encrypt", "backup", "restore", "recover", "validate", "attest",
    "log", "track", "remediate", "assess", "evidence", "ensure",
)

# Best-effort regulator lookup for the explainability bundle. Maps regulation
# codes to the supervisory authority that typically enforces it. Used only as
# a default — the field can be overridden by anything more specific picked up
# from the requirement text (Article reference, RTS reference, etc.).
_REGULATION_TO_REGULATOR = {
    "DORA": "European Supervisory Authorities (ESMA / EBA / EIOPA)",
    "MIFID II": "ESMA + National Competent Authorities",
    "MIFID": "ESMA + National Competent Authorities",
    "GDPR": "European Data Protection Board / National DPAs",
    "EMIR": "ESMA + National Competent Authorities",
    "SFTR": "ESMA",
    "MAR": "ESMA + National Competent Authorities",
    "PSD2": "EBA + National Competent Authorities",
    "NIS2": "ENISA + National CSIRTs",
}


def _resolve_regulator(regulation: str, requirement_text: str = "") -> str:
    reg = (regulation or "").strip().upper()
    if reg in _REGULATION_TO_REGULATOR:
        return _REGULATION_TO_REGULATOR[reg]
    haystack = (requirement_text or "").upper()
    for key, regulator in _REGULATION_TO_REGULATOR.items():
        if key in haystack:
            return regulator
    return f"{regulation} competent authority" if regulation else "Competent authority"

# Subject-verb stems that we strip from a behaviour clause so that what remains
# reads as a verb phrase (e.g. "monitor risks" rather than "The organization
# must monitor risks"). Order matters — longer phrases must come first.
_SUBJECT_STEMS = (
    "the organization must",
    "the organisation must",
    "the firm must",
    "the institution must",
    "the entity must",
    "the system must",
    "the system shall",
    "the firm shall",
    "the organization shall",
    "the organisation shall",
    "the firm should",
    "the organization should",
    "the institution shall",
    "the institution should",
    "management must",
    "the bank must",
    "the bank shall",
    "the company must",
    "the company shall",
)


def _strip_id_prefix(text: str) -> str:
    """Drop a leading 'BR-PRO-001 — ' / 'FR-002:' style identifier from a sentence."""
    return _re.sub(r"^\s*(?:BR-[A-Z]+-\d+|FR-\d+|NFR-\d+|REQ-\d+|R\d+)\s*[—\-:.\u2014\u2013]\s*", "", text or "")


def _strip_subject_stem(text: str) -> str:
    """Drop a leading subject-verb stem ("The organization must ...") so the
    remainder reads as an imperative verb phrase."""
    if not text:
        return text
    lower = text.lower().lstrip()
    for stem in _SUBJECT_STEMS:
        if lower.startswith(stem):
            trimmed = text.lstrip()[len(stem):].lstrip()
            if trimmed:
                return trimmed[0].lower() + trimmed[1:] if trimmed[0].isupper() else trimmed
    return text


def _short_clause(text: str, max_words: int = 18, strip_subject: bool = True) -> str:
    """Return the first sentence-ish fragment of ``text`` capped at ``max_words``."""
    if not text:
        return ""
    cleaned = _re.sub(r"\s+", " ", clean_text(text)).strip()
    cleaned = _strip_id_prefix(cleaned)
    if strip_subject:
        cleaned = _strip_subject_stem(cleaned)
    parts = _re.split(r"(?<=[.;\n])\s+", cleaned)
    first = (parts[0] if parts else cleaned).strip().rstrip(".,;:")
    words = first.split()
    if len(words) <= max_words:
        return first
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _extract_article(req: "Requirement") -> str:
    """Pull a regulatory article reference from the requirement, if available."""
    for source in (req.alignment, req.detail, req.requirement, req.acceptance):
        if not source:
            continue
        article = _ARTICLE_RE.search(source)
        if not article:
            continue
        regulation = _REGULATION_RE.search(source)
        prefix = "DORA" if not regulation else regulation.group(1).upper().replace("  ", " ")
        return f"{prefix} Article {article.group(1).strip()}"
    return ""


def _extract_metric(req: "Requirement") -> str:
    """Pull the most informative metric/threshold (e.g. ``4 hours``, ``RTO``)."""
    for source in (req.detail, req.acceptance, req.requirement):
        if not source:
            continue
        match = _METRIC_RE.search(source)
        if match:
            value = match.group(0).strip()
            return _re.sub(r"\s+", " ", value)
    return ""


def _behavioural_anchor(req: "Requirement") -> str:
    """Return a verb-led phrase describing the concrete behaviour the requirement demands."""
    sources = [req.detail, req.requirement, req.acceptance]
    for source in sources:
        clause = _short_clause(source, max_words=22)
        if not clause:
            continue
        lower = clause.lower()
        if any(verb in lower for verb in _VERB_HINTS):
            return clause
    for source in sources:
        clause = _short_clause(source, max_words=18)
        if clause:
            return clause
    return "deliver the control behaviour required by the regulation"


def _evidence_anchor(req: "Requirement") -> str:
    """Return a phrase describing the evidence/acceptance expectation."""
    if req.acceptance:
        clause = _short_clause(req.acceptance, max_words=20)
        if clause:
            return clause
    if req.detail:
        return _short_clause(req.detail, max_words=18)
    return "the documented control outcome"


def _format_req_label(req: "Requirement") -> str:
    """Compact, citation-style label used at the start of each question."""
    title = clean_text(req.requirement) or "Mapped regulatory requirement"
    if len(title) > 60:
        title = title[:57].rstrip() + "..."
    article = _extract_article(req)
    suffix = f", {article}" if article else ""
    return f"{req.normalized_id} — {title}{suffix}"


def _select_anchor_requirement(reqs: Sequence["Requirement"]) -> Optional["Requirement"]:
    """Pick the most informative mapped requirement to anchor the question on."""
    if not reqs:
        return None

    def score(r: "Requirement") -> int:
        s = 0
        if _extract_article(r):
            s += 5
        if _extract_metric(r):
            s += 3
        if r.priority and any(tag in r.priority.lower() for tag in ("must", "high", "critical")):
            s += 3
        if r.acceptance:
            s += 1
        s += min(3, len(r.detail or "") // 120)
        return s

    return max(reqs, key=score)


def _dominant_theme_key(reqs: Sequence["Requirement"]) -> Optional[str]:
    """Return the most frequent theme key across the mapped requirements."""
    if not reqs:
        return None
    counts: Counter = Counter()
    for r in reqs:
        for theme in r.themes:
            key = _THEME_TO_KEY.get(theme)
            if key:
                counts[key] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# v13 — Explainability bundle + canonical implementation-status root question
# ---------------------------------------------------------------------------

# Maps the canonical theme label (output of :func:`infer_themes`) to a
# concise control-objective phrase used in explainability metadata and as the
# subject of the L1 implementation-status root question.
_THEME_TO_CONTROL_OBJECTIVE = {
    "Governance": "Documented and approved governance/control framework",
    "ICT risk management": "Implemented ICT risk-management framework",
    "Incident reporting": "Operational major-ICT-incident classification and notification process",
    "Resilience testing": "Periodic resilience / recovery testing programme",
    "Third-party risk": "Critical ICT third-party contracts and exit planning",
    "Data and evidence": "Evidence dictionary with lineage, ownership and retention",
    "Security and access": "Privileged access, encryption, vulnerability and SIEM coverage",
    "Reporting": "Management reporting with KRIs/KPIs to the management body",
    "General regulatory coverage": "Regulatory control coverage for the mapped requirement",
}


_THEME_TO_RISK_IF_NEGATIVE = {
    "Governance": "No traceable management-body approval; assessment cannot be evidenced or audited.",
    "ICT risk management": "Risks are not identified, assessed or treated within the regulatory framework.",
    "Incident reporting": "Risk of missing the regulatory notification deadline and incurring sanction.",
    "Resilience testing": "Recovery objectives unproven; operational disruption may extend beyond regulatory tolerances.",
    "Third-party risk": "Critical contracts lack DORA clauses; the firm cannot demonstrate oversight or exit ability.",
    "Data and evidence": "Audit trail is incomplete; supervisory inspections cannot be substantiated.",
    "Security and access": "Material control gaps in privileged access / encryption increase breach exposure.",
    "Reporting": "Management body lacks situational awareness; weak governance attestation.",
    "General regulatory coverage": "Underlying regulatory obligation may not be demonstrably met at the next inspection.",
}


def _theme_label_from_requirements(reqs: Sequence[Requirement]) -> str:
    """Return the canonical theme label (not the registry key) for the anchor requirement set."""
    if not reqs:
        return "General regulatory coverage"
    counts: Counter = Counter()
    for r in reqs:
        for theme in r.themes:
            counts[theme] += 1
    if not counts:
        return "General regulatory coverage"
    return counts.most_common(1)[0][0]


def _build_explainability(
    *,
    regulation: str,
    pair: ImpactPair,
    anchor: Requirement,
    mapped_requirement_ids: Sequence[str],
    theme_label: str,
    article: str,
    reason: str,
    expected_evidence: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the structured explainability dict attached to every Question.

    Every field is best-effort: when the underlying BRD/RTM does not surface
    a value we emit a sensible default rather than ``""`` so the UI never
    shows blank rows. ``extra`` lets callers override individual fields
    (e.g. branch questions can override ``reason`` and ``risk_if_negative``).
    """
    requirement_text = " ".join([
        anchor.alignment or "",
        anchor.detail or "",
        anchor.requirement or "",
    ])
    obligation_id = ""
    if anchor.source_id:
        obligation_id = anchor.source_id
    elif mapped_requirement_ids:
        obligation_id = mapped_requirement_ids[0]
    # Carry the anchor requirement's source citations into the question so the
    # assessment UI's "Why am I being asked this?" panel can render the
    # underlying regulatory publications. We use the anchor requirement's
    # references rather than aggregating across all mapped IDs to keep the
    # citation focused on the dominant obligation the question is testing.
    source_references = list(getattr(anchor, "source_references", []) or [])
    bundle: Dict[str, Any] = {
        "regulation": regulation,
        "regulator": _resolve_regulator(regulation, requirement_text),
        "article": article or pair.regulatory_basis or f"{regulation} mapped clause",
        "obligation_id": obligation_id,
        "brd_requirement_ids": list(mapped_requirement_ids),
        "rtm_trace_ids": [f"TR-{rid}" for rid in mapped_requirement_ids[:5]],
        "business_function": pair.function,
        "business_area": pair.area,
        "control_objective": _THEME_TO_CONTROL_OBJECTIVE.get(
            theme_label, _THEME_TO_CONTROL_OBJECTIVE["General regulatory coverage"],
        ),
        "theme": theme_label,
        "reason": reason,
        "expected_evidence": expected_evidence,
        "risk_if_negative": _THEME_TO_RISK_IF_NEGATIVE.get(
            theme_label, _THEME_TO_RISK_IF_NEGATIVE["General regulatory coverage"],
        ),
        "source_references": source_references,
    }
    if extra:
        bundle.update({k: v for k, v in extra.items() if v not in (None, "", [], {})})
    return bundle


# Per-option metadata for the canonical implementation-status family. Each
# entry becomes a dict-shaped option carrying:
#   - label          : the user-facing answer
#   - score_value    : numeric score consumed by services.scoring_engine
#   - branch_rule_id : namespaced rule key looked up in branch_registry
#   - reason         : short explanation surfaced in the audit trail
def _implementation_status_options(theme_label: str) -> List[Dict[str, Any]]:
    theme_key = theme_label.replace(" ", "_").replace("-", "_").lower() or "general"
    base = [
        {
            "label": "Fully Implemented",
            "score_value": 100,
            "branch_rule_id": f"{theme_key}__status__fully_implemented",
            "reason": (
                "Trigger the validation/evidence branch — implementation is claimed complete; "
                "the next questions confirm evidence and last-test cadence rather than asking "
                "for implementation detail."
            ),
        },
        {
            "label": "Partially Implemented",
            "score_value": 55,
            "branch_rule_id": f"{theme_key}__status__partially_implemented",
            "reason": (
                "Trigger the partial-implementation branch — the next questions isolate which "
                "stage is incomplete, which teams are affected, and what evidence already exists."
            ),
        },
        {
            "label": "Not Implemented",
            "score_value": 0,
            "branch_rule_id": f"{theme_key}__status__not_implemented",
            "reason": (
                "Trigger the not-started branch — the next questions investigate blockers, "
                "ownership, planned start, and required funding/support."
            ),
        },
        {
            "label": "Not Applicable",
            "score_value": None,
            "branch_rule_id": f"{theme_key}__status__not_applicable",
            "reason": (
                "Mark scope-out — exclude from scoring and request a single justification "
                "follow-up so the N/A claim is auditable."
            ),
        },
        {
            "label": "Unknown",
            "score_value": 25,
            "branch_rule_id": f"{theme_key}__status__unknown",
            "reason": (
                "Trigger the discovery branch — identify who can authoritatively answer this "
                "before any deeper readiness questions are asked."
            ),
        },
    ]
    return base


def _theme_question_text(theme_key: str, anchor: Requirement, pair: ImpactPair) -> Tuple[str, str]:
    """Return ``(question_text, rationale)`` for the theme-specific deep dive.

    Questions are intentionally written in **plain, business-friendly
    English** so an SME can answer without decoding jargon or having to
    look up an article number. The technical anchors (article, metric,
    behaviour) are still surfaced in the rationale for auditability.
    """
    label = _format_req_label(anchor)
    article = _extract_article(anchor) or pair.regulatory_basis
    metric = _extract_metric(anchor)
    behaviour = _behavioural_anchor(anchor)
    if theme_key == "incident_reporting":
        text = (
            f"For {label}, does {pair.function} report major incidents to the "
            "regulator on time?"
        )
        rationale = (
            f"Tests classification, severity and notification-window "
            f"execution for {article}. Behaviour anchor: '{behaviour}'."
            + (f" Metric expected: {metric}." if metric else "")
        )
        return text, rationale
    if theme_key == "third_party":
        text = (
            f"For {label}, do {pair.area} vendor contracts cover the "
            "required audit, exit and data terms?"
        )
        rationale = (
            f"Probes contractual readiness against {article} for {pair.function}. "
            "Missing clauses are the biggest DORA third-party gap."
        )
        return text, rationale
    if theme_key == "resilience_testing":
        text = (
            f"For {label}, has {pair.function} run its resilience tests "
            "on schedule?"
        )
        rationale = (
            f"Validates that testing required by {article} is executed on time and "
            f"linked to '{behaviour}'."
            + (f" Metric expected: {metric}." if metric else "")
        )
        return text, rationale
    if theme_key == "security_access":
        text = (
            f"For {label}, are the security controls in {pair.area} "
            "working and reviewed regularly?"
        )
        rationale = (
            f"Tests design and operating effectiveness of the controls behind "
            f"{article}, including review cadence."
        )
        return text, rationale
    if theme_key == "governance":
        text = (
            f"For {label}, has the management body approved the policy "
            "with documented evidence?"
        )
        rationale = (
            f"Confirms governance approval is documented and traceable for "
            f"{article} in {pair.area} / {pair.function}."
        )
        return text, rationale
    if theme_key == "data_evidence":
        text = (
            f"For {label}, do we have an evidence trail linking each "
            f"{pair.function} control output to an owner and source?"
        )
        rationale = (
            f"Tests evidence-dictionary maturity so {article} outcomes can be "
            "audited and substantiated."
        )
        return text, rationale
    if theme_key == "reporting":
        text = (
            f"For {label}, is the {pair.area} management dashboard "
            "current and traceable?"
        )
        rationale = (
            f"Validates that reporting for {pair.function} carries live KRI/KPI values "
            f"traceable to '{behaviour}' as expected by {article}."
        )
        return text, rationale
    return "", ""


def build_closed_questions_for_pair(
    pair: ImpactPair,
    requirements: Sequence[Requirement],
    start_idx: int,
    regulation: str = "DORA",
) -> List[Question]:
    """Synthesise a requirement-specific, fully-funneled question set for one impact pair.

    v13 funnel tree (per pair)::

        L1  Implementation status  (root, asks "Is the <theme> process implemented?")
              |
              +-- Per-option branches:
              |     - Fully Implemented      -> validation/evidence branch (registry+AI)
              |     - Partially Implemented  -> partial-implementation branch
              |     - Not Implemented        -> blocker/ownership/planning branch
              |     - Not Applicable         -> single N/A justification follow-up
              |     - Unknown                -> discovery branch ("who can answer this?")
              |
              +-- L2  Ownership   (parent=L1, triggers=Partially/Not/Unknown)
              |       |
              |       +-- L3  Risk (shared)
              |
              +-- L2  Evidence    (parent=L1, triggers=Partially/Not/Unknown)
              |
              +-- L2  Risk        (parent=L1, triggers=Partially/Not/Unknown + weak ownership)
                       |
                       +-- L3  Remediation  (parent=Risk, triggers=Medium/High/Critical/Unknown)

        L2  Theme deep-dive       (parent=L1, triggers=Partially/Not/Unknown when a
                                   dominant theme is detected for the pair)

    Every non-root question carries a ``funnel_parent_id`` and a non-empty
    ``trigger_answers`` list, so the live cockpit can short-circuit branches when
    the parent answer is positive.

    The L1 root uses dict-shaped options carrying ``score_value`` and
    ``branch_rule_id`` so the scoring engine can route per-option to the
    branch registry / GenAI fallback without needing to re-classify the answer.
    Every question carries a structured :py:attr:`Question.explainability`
    bundle for traceability.
    """
    req_by_id = {r.normalized_id: r for r in requirements}
    mapped_reqs = [req_by_id[i] for i in pair.requirement_ids if i in req_by_id]
    anchor = _select_anchor_requirement(mapped_reqs) or (requirements[0] if requirements else None)
    if anchor is None:
        return []

    req_ids = pair.requirement_ids[:5]
    base_conf = clamp_confidence(min(pair.confidence + 1, 98))
    label = _format_req_label(anchor)
    behaviour = _behavioural_anchor(anchor)
    behaviour_short = _short_clause(behaviour, max_words=12) or behaviour
    evidence = _evidence_anchor(anchor)
    article = _extract_article(anchor) or pair.regulatory_basis
    metric = _extract_metric(anchor)
    metric_clause = f" (target: {metric})" if metric else ""

    theme_label = _theme_label_from_requirements(mapped_reqs)
    theme_key_local = _THEME_TO_KEY.get(theme_label)
    branch_theme_label = theme_label if theme_label != "General regulatory coverage" else ""
    control_objective = _THEME_TO_CONTROL_OBJECTIVE.get(
        theme_label, _THEME_TO_CONTROL_OBJECTIVE["General regulatory coverage"],
    )

    qid_status = f"Q-{start_idx:04d}"
    qid_ownership = f"Q-{start_idx + 1:04d}"
    qid_evidence = f"Q-{start_idx + 2:04d}"
    qid_risk = f"Q-{start_idx + 3:04d}"
    qid_remediation = f"Q-{start_idx + 4:04d}"

    negative_status_triggers = ["Partially Implemented", "Not Implemented", "Unknown"]
    weak_ownership_triggers = [
        "Shared ownership", "Informal owner", "No owner assigned", "Unknown",
    ]
    high_risk_triggers = ["Medium", "High", "Critical", "Unknown"]

    status_options = _implementation_status_options(theme_label)

    def explain(reason: str, expected_evidence: str, **extra: Any) -> Dict[str, Any]:
        return _build_explainability(
            regulation=regulation,
            pair=pair,
            anchor=anchor,
            mapped_requirement_ids=req_ids,
            theme_label=theme_label,
            article=article,
            reason=reason,
            expected_evidence=expected_evidence,
            extra=extra or None,
        )

    questions: List[Question] = [
        Question(
            question_id=qid_status,
            area=pair.area,
            function=pair.function,
            question_type="Single Select",
            question=(
                f"Is the {control_objective.lower()} in place for "
                f"{pair.area} / {pair.function}?"
            ),
            options=status_options,
            mapped_requirement_ids=req_ids,
            regulatory_basis=pair.regulatory_basis,
            confidence=base_conf,
            scoring_weight=3,
            funnel_parent_id="",
            trigger_answers=[],
            rationale=(
                f"Root screening question for the funnel. Establishes implementation status against "
                f"{article} and routes per-option to the matching adaptive branch. Mapped to "
                f"{', '.join(req_ids)}."
            ),
            branch_theme=branch_theme_label,
            explainability=explain(
                reason=(
                    f"Establishes the implementation baseline so the engine can route the user to the "
                    f"correct branch — Fully Implemented unlocks evidence/validation questions, "
                    f"Partially Implemented isolates which stage is incomplete, Not Implemented "
                    f"investigates blockers and ownership, Not Applicable is recorded with a single "
                    f"justification."
                ),
                expected_evidence=(
                    "Process documentation, control register entry, last test report, and a named "
                    "accountable owner for the control."
                ),
            ),
        ),
        Question(
            question_id=qid_ownership,
            area=pair.area,
            function=pair.function,
            question_type="Single Select",
            question=(
                f"Who owns the {control_objective.lower()} in "
                f"{pair.area} / {pair.function}?"
            ),
            options=DEFAULT_OPTIONS["ownership"],
            mapped_requirement_ids=req_ids,
            regulatory_basis=pair.regulatory_basis,
            confidence=base_conf,
            scoring_weight=3,
            funnel_parent_id=qid_status,
            trigger_answers=negative_status_triggers,
            rationale=(
                "Triggered when implementation status is below 'Fully Implemented'. Tests whether "
                "accountability is clear enough to support regulatory evidence and remediation."
            ),
            branch_theme=branch_theme_label,
            explainability=explain(
                reason=(
                    "Without a named accountable owner the control cannot be operated, audited or "
                    "remediated — this is the second-highest predictor of regulatory readiness."
                ),
                expected_evidence=(
                    "RACI entry, role assignment in the policy, signed attestation or governance "
                    "minute referencing the owner."
                ),
            ),
        ),
        Question(
            question_id=qid_evidence,
            area=pair.area,
            function=pair.function,
            question_type="Multi Select",
            question=(
                f"What evidence can {pair.function} show today for "
                f"{pair.area}?"
            ),
            options=DEFAULT_OPTIONS["evidence"],
            mapped_requirement_ids=req_ids,
            regulatory_basis=pair.regulatory_basis,
            confidence=clamp_confidence(base_conf - 2),
            scoring_weight=2,
            funnel_parent_id=qid_status,
            trigger_answers=negative_status_triggers,
            rationale=(
                "Triggered when implementation status is below 'Fully Implemented'. Validates that "
                "the answer can be evidenced through policies, workflow records, dashboards, audit "
                "trails or contractual artefacts referenced in the acceptance criteria."
            ),
            branch_theme=branch_theme_label,
            explainability=explain(
                reason=(
                    "Evidence is the bridge between a control claim and a regulator's inspection — "
                    "any partial answer must be backed by at least one auditable artefact."
                ),
                expected_evidence=evidence,
            ),
        ),
        Question(
            question_id=qid_risk,
            area=pair.area,
            function=pair.function,
            question_type="Single Select",
            question=(
                f"How much risk is still open for "
                f"{pair.area} / {pair.function}?"
            ),
            options=DEFAULT_OPTIONS["risk_level"],
            mapped_requirement_ids=req_ids,
            regulatory_basis=pair.regulatory_basis,
            confidence=clamp_confidence(base_conf - 2),
            scoring_weight=3,
            funnel_parent_id=qid_status,
            trigger_answers=negative_status_triggers + weak_ownership_triggers,
            rationale=(
                f"Triggered by Partially / Not Implemented / Unknown status, or by weak ownership. "
                f"Identifies whether the requirement is still exposed after current controls and "
                f"evidence. Mapped to {article}."
            ),
            branch_theme=branch_theme_label,
            explainability=explain(
                reason=(
                    "Converts an implementation-status gap into a quantified risk signal that feeds "
                    "the heatmap, Agent 4 prioritisation and the recommendations engine."
                ),
                expected_evidence=(
                    "Risk register entry, risk acceptance memo, scenario analysis or treatment plan."
                ),
            ),
        ),
        Question(
            question_id=qid_remediation,
            area=pair.area,
            function=pair.function,
            question_type="Single Select",
            question=(
                f"How mature is the remediation plan for "
                f"{pair.area} / {pair.function}?"
            ),
            options=DEFAULT_OPTIONS["maturity"],
            mapped_requirement_ids=req_ids,
            regulatory_basis=pair.regulatory_basis,
            confidence=clamp_confidence(base_conf - 2),
            scoring_weight=2,
            funnel_parent_id=qid_risk,
            trigger_answers=high_risk_triggers,
            rationale=(
                "Triggered only when residual risk is Medium, High, Critical or Unknown. Tests "
                "whether the client has moved from gap identification to a funded and measurable "
                "remediation plan."
            ),
            branch_theme=branch_theme_label,
            explainability=explain(
                reason=(
                    "Without a funded remediation plan the gap stays open at the next inspection — "
                    "this question tests programme maturity, not just intent."
                ),
                expected_evidence=(
                    "Project plan, funded line in the programme budget, accountable owner and "
                    "target milestones."
                ),
            ),
        ),
    ]

    theme_key = theme_key_local
    if theme_key:
        text, rationale = _theme_question_text(theme_key, anchor, pair)
        if text:
            questions.append(Question(
                question_id=f"Q-{start_idx + 5:04d}",
                area=pair.area,
                function=pair.function,
                question_type="Single Select",
                question=text,
                options=THEME_OPTIONS[theme_key],
                mapped_requirement_ids=req_ids,
                regulatory_basis=pair.regulatory_basis,
                confidence=clamp_confidence(base_conf - 1),
                scoring_weight=2,
                funnel_parent_id=qid_status,
                trigger_answers=negative_status_triggers,
                rationale=(
                    f"Theme-specific deep dive ({theme_key.replace('_', ' ')}). {rationale} "
                    f"Triggered when implementation status is below 'Fully Implemented'."
                ),
                branch_theme=branch_theme_label,
                explainability=explain(
                    reason=(
                        f"Theme deep-dive for {theme_label}. {rationale}"
                    ),
                    expected_evidence=evidence,
                ),
            ))

    return questions


_THEME_FREE_TEXT_PROMPTS = {
    "Incident reporting": (
        "How does {function} report major incidents on time for {focus}? "
        "Share any recent close calls."
    ),
    "Third-party risk": (
        "For {focus}, which key vendors support {function}, and what "
        "contract or exit gaps are still open?"
    ),
    "Resilience testing": (
        "What was the last resilience test {function} ran for {focus}, "
        "and what is still open?"
    ),
    "Security and access": (
        "Describe the key security controls in place for {function} "
        "covering {focus}, including known exceptions."
    ),
    "Governance": (
        "Who approved the policy for {focus}, and are any approvals "
        "still pending in {function}?"
    ),
    "Data and evidence": (
        "For {focus}, how does {function} prove the control works — "
        "which artefacts and any data gaps?"
    ),
    "Reporting": (
        "How does {function} report on {focus} to leadership, and "
        "what reporting gaps remain?"
    ),
    "ICT risk management": (
        "How does {function} manage risk for {focus} — accepted risks, "
        "register entries, and review cycle?"
    ),
}

_GENERIC_FREE_TEXT_PROMPTS = [
    "What are the biggest gaps that could block compliance with {ids}?",
    "Any assumptions on {ids} that Legal or Compliance should confirm?",
    "What data, evidence or reporting limits affect scoring for {ids}?",
    "Any budget, sponsorship or ownership constraints for {ids}?",
    "Any scope exclusions for {ids} — and why?",
    "Anything else the reviewer should know about {ids}?",
]


def build_free_text_questions(
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    start_idx: int,
) -> List[Question]:
    """Produce theme- and requirement-specific narrative questions.

    Each free-text prompt now references concrete requirement IDs and (where
    possible) the requirement focus phrase, instead of asking generic
    ``describe...`` questions. The pair list seeds the function context.
    """
    theme_counts = Counter(theme for r in requirements for theme in r.themes)
    top_themes = [t for t, _ in theme_counts.most_common(len(_THEME_FREE_TEXT_PROMPTS))]

    pair_by_theme: Dict[str, ImpactPair] = {}
    for pair in pairs:
        if not pair.requirement_ids:
            continue
        pair_reqs = [r for r in requirements if r.normalized_id in pair.requirement_ids]
        for theme in (t for r in pair_reqs for t in r.themes):
            pair_by_theme.setdefault(theme, pair)

    questions: List[Question] = []
    used_prompts: set[str] = set()
    qid_counter = start_idx

    def append(prompt: str, mapped: List[str], theme: str, function: str) -> None:
        nonlocal qid_counter
        key = prompt.lower().strip()
        if key in used_prompts:
            return
        used_prompts.add(key)
        questions.append(Question(
            question_id=f"Q-{qid_counter:04d}",
            area="Free Text / SME Narrative",
            function=function,
            question_type="Open Ended",
            question=prompt,
            options=["Free text response"],
            mapped_requirement_ids=mapped,
            regulatory_basis=theme,
            confidence=92,
            scoring_weight=1,
            funnel_parent_id="",
            trigger_answers=[],
            rationale=(
                "Captures qualitative context, evidence references and SME judgement that the closed-ended "
                f"questions cannot score. Anchored to mapped requirement(s): {', '.join(mapped)}."
            ),
            is_free_text=True,
        ))
        qid_counter += 1

    # Theme-anchored prompts, populated from the actual requirement set.
    for theme in top_themes:
        template = _THEME_FREE_TEXT_PROMPTS.get(theme)
        if not template:
            continue
        theme_reqs = [r for r in requirements if theme in r.themes][:3]
        if not theme_reqs:
            continue
        anchor = _select_anchor_requirement(theme_reqs) or theme_reqs[0]
        pair_ctx = pair_by_theme.get(theme)
        function = pair_ctx.function if pair_ctx else "Cross-functional"
        focus = _format_req_label(anchor)
        mapped_ids = [r.normalized_id for r in theme_reqs]
        prompt = template.format(focus=focus, function=function)
        append(prompt, mapped_ids, theme, function)

    # Pad with cross-cutting generic prompts that still reference real IDs.
    rotation = [r.normalized_id for r in requirements[:8]] or ["the mapped requirements"]
    for i, template in enumerate(_GENERIC_FREE_TEXT_PROMPTS):
        if len(questions) >= MAX_FREE_TEXT:
            break
        ids_subset = rotation[i % len(rotation): i % len(rotation) + 3] or rotation[:3]
        ids_text = ", ".join(ids_subset)
        mapped_ids = list(ids_subset)
        prompt = template.format(ids=ids_text)
        theme = top_themes[i % len(top_themes)] if top_themes else "General regulatory coverage"
        append(prompt, mapped_ids, theme, "Cross-functional")

    # Honour the configured minimum count if we have somehow generated fewer.
    while len(questions) < MIN_FREE_TEXT and len(questions) < MAX_FREE_TEXT:
        idx = len(questions)
        template = _GENERIC_FREE_TEXT_PROMPTS[idx % len(_GENERIC_FREE_TEXT_PROMPTS)]
        ids_subset = rotation[idx % len(rotation): idx % len(rotation) + 3] or rotation[:3]
        ids_text = ", ".join(ids_subset)
        append(template.format(ids=ids_text), list(ids_subset), "General regulatory coverage", "Cross-functional")

    return questions[:MAX_FREE_TEXT]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def question_kind(question: Question) -> str:
    text = question.question.lower()
    # v13 — the canonical L1 status question. The plain-English rewrite
    # uses "Is the ... in place for ..." so we accept both the legacy
    # "implemented" wording and the new "in place" phrasing.
    if text.startswith("is the ") and ("implemented" in text or "in place" in text):
        return "coverage"
    if "implementation coverage" in text or "current implementation" in text:
        return "coverage"
    if (
        "named accountable owner" in text
        or "accountable owner" in text
        or "ownership model" in text
        or text.startswith("who owns ")
    ):
        return "ownership"
    if (
        "substantiating evidence" in text
        or "which evidence" in text
        or "evidence artefact" in text
        or "what evidence can" in text
        or text.startswith("what evidence")
    ):
        return "evidence"
    if (
        "remediation maturity" in text
        or ("remediation" in text and "gap" in text)
        or text.startswith("how mature is the remediation")
    ):
        return "remediation"
    if (
        "residual dora" in text
        or ("residual" in text and "risk" in text)
        or text.startswith("how much risk is still open")
    ):
        return "risk"
    # Theme deep-dive markers — each yields a distinct kind so the dedupe
    # logic does not collapse a third-party question with a security or
    # incident question for the same area/requirement set.
    if "incident" in text and ("classif" in text or "notification deadline" in text
                                or "report major incidents" in text):
        return "theme_incident"
    if (
        "third-party" in text
        or "third party" in text
        or "vendor contracts" in text
    ):
        return "theme_third_party"
    if (
        "resilience test" in text
        or "threat-led penetration" in text
        or "tlpt" in text
        or "resilience tests" in text
    ):
        return "theme_resilience"
    if (
        "privileged access" in text
        or "siem" in text
        or "encryption" in text
        or "security controls" in text
    ):
        return "theme_security"
    if ("management body" in text and "approv" in text) or "approved the policy" in text:
        return "theme_governance"
    if (
        "evidence dictionary" in text
        or "data lineage" in text
        or "metadata" in text
        or "evidence trail" in text
    ):
        return "theme_data"
    if (
        "management reporting" in text
        or "dashboard" in text
        or "kri" in text
        or "kpi" in text
    ):
        return "theme_reporting"
    if question.is_free_text:
        return "free_text"
    return _re.sub(r"[^a-z0-9]+", " ", text).strip()[:80]


def overlap_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dedupe_impact_pairs(pairs: Sequence[ImpactPair]) -> List[ImpactPair]:
    kept: List[ImpactPair] = []
    for pair in pairs:
        duplicate = False
        for existing in kept:
            same_axis = pair.area == existing.area or pair.function == existing.function
            if same_axis and overlap_ratio(pair.requirement_ids, existing.requirement_ids) >= 0.80:
                duplicate = True
                break
        if not duplicate:
            kept.append(pair)
    return kept


def dedupe_and_resequence_questions(questions: Sequence[Question]) -> List[Question]:
    kept: List[Question] = []
    seen_text: set[str] = set()
    seen_requirement_kind: set[Tuple[str, Tuple[str, ...]]] = set()
    seen_area_kind_primary: set[Tuple[str, str, str]] = set()

    for q in questions:
        norm_text = _re.sub(r"\W+", " ", q.question.lower()).strip()
        req_key = tuple(sorted(set(q.mapped_requirement_ids)))
        primary_req = req_key[0] if req_key else "FREE"
        kind = question_kind(q)

        requirement_kind_key = (kind, req_key)
        area_kind_primary_key = (kind, q.area, primary_req)

        if norm_text in seen_text:
            continue
        if not q.is_free_text and requirement_kind_key in seen_requirement_kind:
            continue
        if not q.is_free_text and area_kind_primary_key in seen_area_kind_primary:
            continue

        near_duplicate = False
        for existing in kept:
            if existing.is_free_text or q.is_free_text:
                continue
            same_kind = question_kind(existing) == kind
            same_axis = existing.area == q.area or existing.function == q.function
            req_overlap = overlap_ratio(existing.mapped_requirement_ids, q.mapped_requirement_ids)
            if same_kind and req_overlap >= 0.70 and same_axis:
                near_duplicate = True
                break
        if near_duplicate:
            continue

        seen_text.add(norm_text)
        seen_requirement_kind.add(requirement_kind_key)
        seen_area_kind_primary.add(area_kind_primary_key)
        kept.append(q)

    old_to_new: Dict[str, str] = {}
    for idx, q in enumerate(kept, start=1):
        old_to_new[q.question_id] = f"Q-{idx:04d}"
    for idx, q in enumerate(kept, start=1):
        q.question_id = f"Q-{idx:04d}"
        if q.funnel_parent_id:
            q.funnel_parent_id = old_to_new.get(q.funnel_parent_id, "")
            if not q.funnel_parent_id:
                q.trigger_answers = []
    return kept


def generate_question_bank(
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    regulation: str = "DORA",
) -> List[Question]:
    questions: List[Question] = []
    idx = 1
    clean_pairs = dedupe_impact_pairs(pairs)
    for pair in clean_pairs:
        qs = build_closed_questions_for_pair(pair, requirements, idx, regulation=regulation)
        questions.extend(qs)
        idx += len(qs)
    questions.extend(build_free_text_questions(requirements, clean_pairs, idx))
    return dedupe_and_resequence_questions(questions)


# ---------------------------------------------------------------------------
# Validation, scoring, package, evaluation
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset((
    "the", "and", "for", "with", "from", "into", "this", "that", "must",
    "shall", "should", "will", "have", "has", "are", "was", "were", "been",
    "their", "there", "such", "any", "all", "only", "also", "more", "than",
    "may", "can", "via", "per", "incl", "including", "use", "used", "using",
    "based", "ensure", "ensures", "ensuring", "applicable", "relevant",
    "where", "which", "what", "when", "how", "etc", "ict",
))


def _token_bag(text: Optional[str]) -> set:
    """Lowercased lemma-ish word bag, stopword-filtered, length > 3."""
    if not text:
        return set()
    raw = _re.sub(r"[^a-zA-Z0-9\s]", " ", str(text)).lower().split()
    return {t for t in raw if len(t) > 3 and t not in _STOPWORDS}


def _cited_article_matches_requirement(question_article: str, mapped_reqs: Sequence[Requirement]) -> bool:
    """True iff the article cited by the question appears in at least one mapped requirement's alignment.

    Catches hallucinated citations — a question that says "DORA Article 30"
    when its mapped BRD row only talks about Article 6 fails this check.
    """
    if not question_article or not mapped_reqs:
        return False
    q_match = _ARTICLE_RE.search(question_article)
    if q_match:
        q_num = q_match.group(1).strip().lower()
        for req in mapped_reqs:
            for source in (req.alignment, req.detail, req.requirement, req.acceptance):
                if not source:
                    continue
                for r_match in _ARTICLE_RE.findall(source):
                    if r_match.strip().lower() == q_num:
                        return True
        return False
    # No article number in the question citation — fall back to literal substring match.
    needle = question_article.lower().strip()
    return any(needle in (req.alignment or "").lower() for req in mapped_reqs)


def _question_text_anchors_to_requirement(question_text: str, mapped_reqs: Sequence[Requirement]) -> bool:
    """True iff the question text shares at least 3 content tokens with the mapped requirement."""
    if not question_text or not mapped_reqs:
        return False
    q_tokens = _token_bag(question_text)
    if not q_tokens:
        return False
    for req in mapped_reqs:
        req_tokens = _token_bag(" ".join([
            req.requirement or "", req.detail or "", req.acceptance or "",
            req.category or "",
        ]))
        if len(q_tokens & req_tokens) >= 3:
            return True
    return False


def _evidence_anchors_to_acceptance(expected_evidence: str, mapped_reqs: Sequence[Requirement]) -> bool:
    """True iff the question's expected-evidence shares vocabulary with the mapped acceptance criteria."""
    if not expected_evidence or not mapped_reqs:
        return False
    ev_tokens = _token_bag(expected_evidence)
    if not ev_tokens:
        return False
    for req in mapped_reqs:
        accept_tokens = _token_bag(req.acceptance) | _token_bag(req.detail)
        if len(ev_tokens & accept_tokens) >= 2:
            return True
    return False


def _question_cites_specific_article(question_text: str, rationale: str) -> bool:
    """True iff the question or its rationale cites a specific article number (not just 'DORA')."""
    blob = " ".join([question_text or "", rationale or ""])
    return bool(_ARTICLE_RE.search(blob))


def _explainability_is_complete(explainability: Mapping[str, Any]) -> bool:
    """True iff every required explainability key is populated and non-empty."""
    if not explainability:
        return False
    for key in EXPLAINABILITY_REQUIRED_KEYS:
        value = explainability.get(key)
        if value is None or value == "" or value == [] or value == {}:
            return False
    return True


def _l1_option_is_grounded(q: Question) -> bool:
    """True iff an L1 root question's options carry per-option branch metadata.

    Returns False for non-root questions (they have a funnel parent) so this
    only counts L1 status questions in the denominator.
    """
    if q.funnel_parent_id or q.is_free_text:
        return False
    opts = q.options or []
    if not opts or not isinstance(opts[0], Mapping):
        return False
    dict_opts = [o for o in opts if isinstance(o, Mapping)]
    if not dict_opts:
        return False
    return all(
        o.get("branch_rule_id") and "score_value" in o
        for o in dict_opts
    )


def _validate_structural_completeness(
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    questions: Sequence[Question],
) -> Tuple[float, Dict[str, float]]:
    """Legacy v11 structural-completeness grade.

    Kept verbatim from v11/v12 so audit trails comparing v13.1 packages to
    historical ones remain like-for-like. Returns ``(overall_pct, breakdown)``.
    """
    req_ids = {r.normalized_id for r in requirements}
    mapped = {rid for q in questions for rid in q.mapped_requirement_ids}
    req_cov = len(req_ids & mapped) / max(1, len(req_ids))
    areas = {p.area for p in pairs}
    q_areas = {q.area for q in questions if not q.is_free_text}
    area_cov = len(areas & q_areas) / max(1, len(areas))
    functions = {p.function for p in pairs}
    q_functions = {q.function for q in questions if not q.is_free_text}
    fn_cov = len(functions & q_functions) / max(1, len(functions))
    avg_q_conf = sum(q.confidence for q in questions) / max(1, len(questions))
    free_text_count = sum(1 for q in questions if q.is_free_text)
    free_text_ok = 1 if MIN_FREE_TEXT <= free_text_count <= MAX_FREE_TEXT else 0
    pair_depth = min(1, (len([q for q in questions if not q.is_free_text]) / max(1, len(pairs) * 4)))
    overall = round(
        (0.30 * req_cov + 0.18 * area_cov + 0.18 * fn_cov + 0.14 * pair_depth + 0.10 * free_text_ok + 0.10 * (avg_q_conf / 100)) * 100,
        1,
    )
    breakdown = {
        "requirement_coverage_pct": round(req_cov * 100, 1),
        "area_coverage_pct": round(area_cov * 100, 1),
        "function_coverage_pct": round(fn_cov * 100, 1),
        "pair_depth_pct": round(pair_depth * 100, 1),
        "average_question_confidence_pct": round(avg_q_conf, 1),
        "free_text_question_count": free_text_count,
        "free_text_in_target_band": bool(free_text_ok),
    }
    return overall, breakdown


def _validate_content_correctness(
    requirements: Sequence[Requirement],
    questions: Sequence[Question],
) -> Tuple[float, Dict[str, float]]:
    """v13.1 content-correctness grade.

    Tests whether each generated question is **grounded** in its mapped BRD
    row(s):

    * cites the right article (no hallucinated citations),
    * shares vocabulary with the requirement detail / acceptance,
    * references the BRD's expected-evidence phrasing,
    * carries a complete explainability bundle,
    * traces back through ``mapped_requirement_ids`` + ``obligation_id``,
    * cites a specific article number (not just the regulation name),
    * (for L1 roots only) has dict-shaped options with per-option
      ``branch_rule_id`` + ``score_value`` so the adaptive funnel can route.

    Returns ``(overall_pct, breakdown)``. The overall is a weighted sum of
    seven per-question pass-rates; weights sum to 1.0.
    """
    closed = [q for q in questions if not q.is_free_text]
    if not closed:
        return 0.0, {
            "article_citation_match_pct": 0.0,
            "behaviour_anchoring_pct": 0.0,
            "evidence_anchoring_pct": 0.0,
            "traceability_completeness_pct": 0.0,
            "explainability_completeness_pct": 0.0,
            "specificity_pct": 0.0,
            "l1_option_grounding_pct": 0.0,
            "closed_question_count": 0,
            "l1_root_question_count": 0,
        }

    req_by_id = {r.normalized_id: r for r in requirements}
    article_hits = 0
    anchor_hits = 0
    evidence_hits = 0
    traceability_hits = 0
    explainability_hits = 0
    specificity_hits = 0
    l1_grounded_hits = 0
    l1_total = 0

    for q in closed:
        mapped_reqs = [req_by_id[rid] for rid in q.mapped_requirement_ids if rid in req_by_id]
        explainability = q.explainability or {}
        question_article = (
            str(explainability.get("article") or q.regulatory_basis or "")
        )
        if _cited_article_matches_requirement(question_article, mapped_reqs):
            article_hits += 1
        if _question_text_anchors_to_requirement(q.question, mapped_reqs):
            anchor_hits += 1
        expected_evidence = str(explainability.get("expected_evidence") or "")
        if _evidence_anchors_to_acceptance(expected_evidence, mapped_reqs):
            evidence_hits += 1
        if q.mapped_requirement_ids and explainability.get("obligation_id"):
            traceability_hits += 1
        if _explainability_is_complete(explainability):
            explainability_hits += 1
        if _question_cites_specific_article(q.question, q.rationale):
            specificity_hits += 1
        if not q.funnel_parent_id:
            l1_total += 1
            if _l1_option_is_grounded(q):
                l1_grounded_hits += 1

    n = len(closed)
    article_rate = article_hits / n
    anchor_rate = anchor_hits / n
    evidence_rate = evidence_hits / n
    traceability_rate = traceability_hits / n
    explainability_rate = explainability_hits / n
    specificity_rate = specificity_hits / n
    l1_rate = (l1_grounded_hits / l1_total) if l1_total else 1.0

    # Weights sum to 1.0
    overall = (
        0.25 * article_rate
        + 0.20 * explainability_rate
        + 0.15 * traceability_rate
        + 0.15 * anchor_rate
        + 0.10 * specificity_rate
        + 0.10 * evidence_rate
        + 0.05 * l1_rate
    ) * 100

    breakdown = {
        "article_citation_match_pct": round(article_rate * 100, 1),
        "behaviour_anchoring_pct": round(anchor_rate * 100, 1),
        "evidence_anchoring_pct": round(evidence_rate * 100, 1),
        "traceability_completeness_pct": round(traceability_rate * 100, 1),
        "explainability_completeness_pct": round(explainability_rate * 100, 1),
        "specificity_pct": round(specificity_rate * 100, 1),
        "l1_option_grounding_pct": round(l1_rate * 100, 1),
        "closed_question_count": n,
        "l1_root_question_count": l1_total,
    }
    return round(overall, 1), breakdown


def validate_and_score_package(
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    questions: Sequence[Question],
) -> Tuple[int, Dict[str, Any]]:
    """Compute the package-confidence headline + a full audit breakdown.

    The headline number returned (``overall``) is:

    * **Content-correctness grade** (default, ``PACKAGE_CONFIDENCE_MODE=content``)
      — seven groundedness signals weighted to test whether each question is
      anchored in its BRD row. See :func:`_validate_content_correctness`.
    * **Structural-completeness grade** (legacy, ``PACKAGE_CONFIDENCE_MODE=structural``)
      — the v11/v12 formula testing whether all BRD requirements, areas,
      functions and pairs are covered. See :func:`_validate_structural_completeness`.

    Both grades are always computed and surfaced in ``metrics`` so audits and
    the recommendations engine can read either.
    """
    content_pct, content_breakdown = _validate_content_correctness(requirements, questions)
    structural_pct, structural_breakdown = _validate_structural_completeness(
        requirements, pairs, questions,
    )

    if PACKAGE_CONFIDENCE_MODE == "structural":
        headline = structural_pct
        mode_used = "structural"
    else:
        headline = content_pct
        mode_used = "content"

    headline = max(OVERALL_CONFIDENCE_FLOOR, min(100, headline))
    metrics: Dict[str, Any] = {
        "package_confidence_mode": mode_used,
        "content_correctness_pct": round(content_pct, 1),
        "structural_completeness_pct": round(structural_pct, 1),
        "content_breakdown": content_breakdown,
        "structural_breakdown": structural_breakdown,
        # Back-compat keys read by older dashboards / exports.
        "requirement_coverage_pct": structural_breakdown["requirement_coverage_pct"],
        # ``coverage_pct`` is the headline "Coverage (closed)" KPI shown on
        # Page 3: what fraction of BRD requirements have at least one closed
        # (mapped) question. Same value as ``requirement_coverage_pct``;
        # kept as an explicit alias because the cockpit reads this key.
        "coverage_pct": structural_breakdown["requirement_coverage_pct"],
        "area_coverage_pct": structural_breakdown["area_coverage_pct"],
        "function_coverage_pct": structural_breakdown["function_coverage_pct"],
        "pair_depth_pct": structural_breakdown["pair_depth_pct"],
        "average_question_confidence_pct": structural_breakdown["average_question_confidence_pct"],
        "free_text_question_count": structural_breakdown["free_text_question_count"],
        "closed_question_count": content_breakdown["closed_question_count"],
        "question_count": len(questions),
        "requirement_count": len(requirements),
        "impact_pair_count": len(pairs),
    }
    return int(round(headline)), metrics


def evaluate_responses(questions: Sequence[Question], responses: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic batch evaluation (kept here for backwards compatibility).

    The live Streamlit cockpit uses an adaptive variant exposed through
    :mod:`services.scoring_engine` in Phase 6. The two implementations share
    :data:`ANSWER_SCORES`.
    """
    numerator = 0.0
    denominator = 0.0
    by_area: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    details: List[Dict[str, Any]] = []
    for q in questions:
        if q.is_free_text:
            continue
        raw = responses.get(q.question_id)
        values = raw if isinstance(raw, list) else [raw]
        scores: List[float] = []
        for value in values:
            if value in ANSWER_SCORES and ANSWER_SCORES[value] is not None:
                scores.append(float(ANSWER_SCORES[value]))  # type: ignore[arg-type]
        if not scores:
            score = 25.0 if raw not in (None, "", []) else 0.0
        else:
            score = sum(scores) / len(scores)
        weighted = score * q.scoring_weight * (q.confidence / 100)
        max_weighted = 100 * q.scoring_weight * (q.confidence / 100)
        numerator += weighted
        denominator += max_weighted
        by_area[q.area][0] += weighted
        by_area[q.area][1] += max_weighted
        details.append({
            "question_id": q.question_id,
            "score": round(score, 1),
            "weighted_score": round(weighted, 1),
            "max_weighted": round(max_weighted, 1),
        })
    compliance = round((numerator / denominator) * 100, 1) if denominator else 0.0
    confidence = round(min(99, max(90, sum(q.confidence for q in questions) / max(1, len(questions)) - (5 if not responses else 0))), 1)
    return {
        "compliance_score_pct": compliance,
        "evaluation_confidence_pct": confidence,
        "area_scores": {area: round(vals[0] / vals[1] * 100, 1) if vals[1] else 0 for area, vals in by_area.items()},
        "details": details,
        "explanation": (
            "Score is weighted by regulatory priority/scoring weight and discounted by per-question confidence. "
            "Not applicable responses are excluded where possible; unknown/unanswered responses reduce confidence and score."
        ),
    }


def package_dict(
    regulation: str,
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    questions: Sequence[Question],
    overall_confidence: int,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    mode = (metrics or {}).get("package_confidence_mode", PACKAGE_CONFIDENCE_MODE)
    note = (
        "Content-correctness grade: every question is tested for article-citation match, "
        "behaviour/evidence anchoring against the BRD, traceability completeness, "
        "explainability bundle completeness, specificity, and (for L1 roots) "
        "per-option branch grounding. Deterministic; not legal advice."
        if mode == "content"
        else
        "Structural-completeness grade (legacy): tests whether all BRD requirements, "
        "impacted areas/functions and pairs have questions. Deterministic; not legal advice."
    )
    # Promote a handful of headline metrics from ``metrics`` to the top of
    # ``metadata`` so the cockpit can read them directly without diving into
    # the nested ``metrics`` blob. ``overall_confidence_pct`` was always
    # surfaced this way; ``coverage_pct`` is the headline "Coverage (closed)"
    # tile on Page 3 (fraction of BRD requirements covered by at least one
    # closed/mapped question).
    coverage_pct = (
        metrics.get("coverage_pct")
        if metrics
        else None
    )
    if coverage_pct is None and metrics:
        coverage_pct = metrics.get("requirement_coverage_pct", 0)
    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "regulation": regulation,
            "overall_confidence_pct": overall_confidence,
            "coverage_pct": coverage_pct if coverage_pct is not None else 0,
            "confidence_note": note,
            "metrics": metrics,
            "regulatory_taxonomy": REGULATORY_TAXONOMY.get(regulation.upper(), {}),
        },
        "requirements": [asdict(r) for r in requirements],
        "impact_pairs": [asdict(p) for p in pairs],
        "questions": [asdict(q) for q in questions],
        "answer_scores": ANSWER_SCORES,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------

def _build_package(requirements: List[Requirement], regulation: str) -> Dict[str, Any]:
    pairs = dedupe_impact_pairs(derive_impact_pairs(requirements, regulation))
    questions = generate_question_bank(requirements, pairs, regulation=regulation)
    overall, metrics = validate_and_score_package(requirements, pairs, questions)
    return package_dict(regulation, requirements, pairs, questions, overall, metrics)


def build_questionnaire_package(
    source: DocxSource,
    regulation: str = "DORA",
) -> Dict[str, Any]:
    """Parse a BRD/FRD DOCX and return the complete questionnaire package."""
    requirements = read_docx_requirements(source)
    if not requirements:
        raise ValueError(
            "No BRD/FRD requirement tables were found. Ensure the DOCX contains tables with "
            "the columns: ID, Category, Requirement, Detailed Requirement, DORA Alignment, "
            "Priority, Acceptance Criteria (AI Confidence optional)."
        )
    return _build_package(requirements, regulation)


def build_package_from_report(
    report: Any,
    regulation: str = "DORA",
    source_refs_by_item: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Build a questionnaire package directly from a Phase-4 DoraDetailedBRD model.

    This is the closed-loop path used when the Streamlit Page 1 "Generate BRD/FRD
    from regulation" option produces an in-memory report.

    ``source_refs_by_item`` is forwarded to :func:`requirements_from_report`
    so every question inherits the citations of the BRD requirement it was
    derived from. Pass the ``source_references_by_item`` map carried on
    ``BRDArtifact.metadata`` to enable end-to-end traceability into the
    questionnaire.
    """
    requirements = requirements_from_report(report, source_refs_by_item or {})
    if not requirements:
        raise ValueError("Report has no requirements; cannot build questionnaire package.")
    return _build_package(requirements, regulation)


# ---------------------------------------------------------------------------
# Excel writer (lifted verbatim from v11)
# ---------------------------------------------------------------------------

def write_excel(
    path: str,
    requirements: Sequence[Requirement],
    pairs: Sequence[ImpactPair],
    questions: Sequence[Question],
    overall_confidence: int,
    metrics: Dict[str, float],
) -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    impact = wb.create_sheet("Impacted Functions Areas")
    qsheet = wb.create_sheet("Questionnaire")
    free = wb.create_sheet("Free Text Questions")
    funnel = wb.create_sheet("Funnel Logic")
    scoring = wb.create_sheet("Scoring Rubric")
    trace = wb.create_sheet("Requirement Traceability")

    header_fill = PatternFill("solid", fgColor="1F4E78")
    white = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_sheet(sheet):
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = white
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in sheet.iter_rows():
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

    ws.append(["Metric", "Value"])
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            ws.append([f"-- {key} --", ""])
            for sub_key, sub_value in value.items():
                ws.append([f"   {sub_key}", sub_value])
        else:
            ws.append([key, value])
    ws.append(["Overall Confidence", f"{overall_confidence}%"])
    ws.append(["Important note",
               "Confidence is a model/rule assurance indicator and not a legal opinion. "
               "Compliance and Legal should validate before client use."])
    style_sheet(ws)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 80

    impact.append(["Impact Area", "Function", "Mapped Requirement IDs", "Regulatory Basis", "Confidence"])
    for p in pairs:
        impact.append([p.area, p.function, " | ".join(p.requirement_ids), p.regulatory_basis, p.confidence])
    style_sheet(impact)
    for col, width in enumerate([36, 32, 42, 58, 14], 1):
        impact.column_dimensions[get_column_letter(col)].width = width

    headers = [
        "Question ID", "Impact Area", "Function", "Type", "Question", "Options",
        "Mapped Requirement IDs", "Regulatory Basis", "Confidence", "Scoring Weight",
        "Funnel Parent", "Trigger Answers", "Rationale",
    ]
    qsheet.append(headers)
    free.append(headers)
    for q in questions:
        rendered_options = " | ".join(option_label(o) for o in q.options)
        row = [q.question_id, q.area, q.function, q.question_type, q.question, rendered_options,
               " | ".join(q.mapped_requirement_ids), q.regulatory_basis, q.confidence, q.scoring_weight,
               q.funnel_parent_id, " | ".join(q.trigger_answers), q.rationale]
        (free if q.is_free_text else qsheet).append(row)
    style_sheet(qsheet)
    style_sheet(free)
    widths = [14, 32, 28, 14, 70, 48, 38, 46, 14, 14, 14, 36, 70]
    for sheet in [qsheet, free]:
        for col, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(col)].width = width
        for r in range(2, sheet.max_row + 1):
            sheet.row_dimensions[r].height = 58

    funnel.append(["Question ID", "Parent Question ID", "Trigger Answers", "Funnel Rule"])
    for q in questions:
        rule = "Always visible" if not q.funnel_parent_id else f"Show when {q.funnel_parent_id} in {', '.join(q.trigger_answers)}"
        funnel.append([q.question_id, q.funnel_parent_id, " | ".join(q.trigger_answers), rule])
    style_sheet(funnel)
    for col, width in enumerate([16, 20, 44, 80], 1):
        funnel.column_dimensions[get_column_letter(col)].width = width

    scoring.append(["Answer", "Score", "Interpretation"])
    for answer, score in ANSWER_SCORES.items():
        scoring.append([answer, "Excluded" if score is None else score,
                        "Mapped deterministic score used by Streamlit evaluation"])
    style_sheet(scoring)
    scoring.column_dimensions["A"].width = 34
    scoring.column_dimensions["B"].width = 14
    scoring.column_dimensions["C"].width = 60

    trace.append(["Requirement ID", "Source ID", "Section", "Category", "Requirement",
                  "DORA Alignment", "Priority", "Mapped Question IDs", "Mapped Question Count"])
    for r in requirements:
        qids = [q.question_id for q in questions if r.normalized_id in q.mapped_requirement_ids]
        trace.append([r.normalized_id, r.source_id, r.source_section, r.category, r.requirement,
                      r.alignment, r.priority, " | ".join(qids), len(qids)])
    style_sheet(trace)
    for col, width in enumerate([18, 18, 32, 24, 54, 42, 12, 62, 18], 1):
        trace.column_dimensions[get_column_letter(col)].width = width

    if qsheet.max_row > 1:
        dv = DataValidation(
            type="list",
            formula1='"Yes,Partially,No,Complete,Mostly complete,Partially complete,Not started,Unknown,Not applicable"',
            allow_blank=True,
        )
        qsheet.add_data_validation(dv)
    wb.save(path)
    return os.path.abspath(path)


def _filter_dataclass_kwargs(cls: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop unknown keys so packages serialised by a future/older version still load."""
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in payload.items() if k in allowed}


def write_excel_from_package(path: str, package: Mapping[str, Any]) -> str:
    """Convenience writer that hydrates the package dict back into dataclasses first."""
    requirements = [Requirement(**_filter_dataclass_kwargs(Requirement, r)) for r in package["requirements"]]
    pairs = [ImpactPair(**_filter_dataclass_kwargs(ImpactPair, p)) for p in package["impact_pairs"]]
    questions = [Question(**_filter_dataclass_kwargs(Question, q)) for q in package["questions"]]
    meta = package.get("metadata", {})
    overall = int(meta.get("overall_confidence_pct", OVERALL_CONFIDENCE_FLOOR))
    metrics = dict(meta.get("metrics", {}))
    return write_excel(path, requirements, pairs, questions, overall, metrics)


__all__ = [
    "ANSWER_SCORES",
    "AREA_KEYWORDS",
    "CONFIDENCE_FLOOR",
    "DEFAULT_OPTIONS",
    "EXPLAINABILITY_REQUIRED_KEYS",
    "FUNCTION_KEYWORDS",
    "ImpactPair",
    "option_label",
    "option_labels",
    "option_metadata",
    "MAX_AREA_FUNCTION_PAIRS",
    "MAX_FREE_TEXT",
    "MIN_FREE_TEXT",
    "OVERALL_CONFIDENCE_FLOOR",
    "PACKAGE_CONFIDENCE_MODE",
    "Question",
    "REGULATORY_TAXONOMY",
    "Requirement",
    "build_closed_questions_for_pair",
    "build_free_text_questions",
    "build_package_from_report",
    "build_questionnaire_package",
    "clamp_confidence",
    "dedupe_and_resequence_questions",
    "dedupe_impact_pairs",
    "derive_impact_pairs",
    "evaluate_responses",
    "generate_question_bank",
    "impacted_labels_for_requirement",
    "infer_themes",
    "package_dict",
    "question_kind",
    "read_docx_requirements",
    "regulatory_basis_for",
    "requirements_from_report",
    "score_keywords",
    "select_option_family",
    "validate_and_score_package",
    "write_excel",
    "write_excel_from_package",
]
