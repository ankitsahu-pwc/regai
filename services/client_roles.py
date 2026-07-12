"""Client Role–Aware Regulatory Interpretation catalog + engine.

This module is the single source of truth for the *Institution Type / Client
Type* dimension of the pipeline. A single regulation does not apply equally
to every financial institution, so before any regulatory analysis begins the
platform now knows **which type(s) of client** it is assessing.

The module ships three things:

1. :data:`INSTITUTION_TYPES` — the canonical catalog of Banking & Capital
   Markets institution types the UI dropdown offers. Each entry carries a
   short business-domain description plus the keyword bag that the offline /
   deterministic applicability engine uses to reason about how the regulation
   maps onto that role.
2. :func:`normalize_client_roles` — validation + canonicalisation helper that
   maps free-form inputs (from the UI multi-select, from persistence, or from
   downstream agents) onto known catalog names.
3. :func:`derive_role_applicability` — the deterministic role-aware
   interpretation engine. Given the regulation text (or a summary), the list
   of selected client roles, and a candidate obligation, it returns a
   :class:`RoleApplicability` record for every role — explaining **why**
   the obligation applies (or does not) to that specific institution type.

The engine is intentionally grounded: every applicability decision quotes the
matched keywords from the regulation / obligation text and the role's own
business domain. When there is not enough signal to justify applicability the
engine returns ``uncertain`` rather than inventing a mapping. Downstream
agents can then decide whether to include the obligation, mark it out of
scope, or route it for SME review.

When a real GenAI client is available the surrounding Agent 1 layer may
overlay an LLM-produced interpretation on top of this heuristic; the
heuristic still runs so the pipeline degrades gracefully when the LLM is
offline (and so the interpretation is always traceable to the regulation
text, per the auditability requirement).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Institution catalog
# ---------------------------------------------------------------------------
#
# The canonical catalog of institution types **is not hardcoded in this
# module**. It lives in ``services/data/institution_types.json`` and is
# loaded once at module import time. This makes the catalog editable
# without changing Python code — a new Banking or Capital Markets
# institution type can be onboarded by adding one JSON entry.
#
# Scope: the catalog is restricted to **Banking & Capital Markets** only
# (Insurance, Payments-only, Consumer-finance-only and FinTech / RegTech
# vendors are intentionally out of scope). The JSON file explicitly
# declares its allowed categories via ``categories`` so drift can be
# detected at load time.
#
# Each entry captures:
#
#   * ``name``      — the canonical label rendered in the UI dropdown.
#   * ``category``  — coarse grouping (``"Banking"`` or
#                     ``"Capital Markets"``).
#   * ``summary``   — one-liner describing what the institution actually does.
#                     Used by the LLM prompt so it knows the business model.
#   * ``domains``   — the regulatory / business surface areas the role
#                     habitually touches (used as anchor keywords when we
#                     score how strongly the regulation applies to the role).
#   * ``keywords``  — additional free-form tokens looked up in the regulation
#                     text and obligation body. Presence + density drives the
#                     deterministic applicability score.
#   * ``typical_obligations`` — cheat-sheet of the obligation themes the role
#                     usually cares about; used by the offline engine to bias
#                     ambiguous cases toward the correct answer.
#
# Override precedence for finding the JSON file at import time:
#   1. ``$REGAI_INSTITUTION_TYPES_PATH`` environment variable (absolute path).
#   2. ``services/data/institution_types.json`` alongside this module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstitutionType:
    """One canonical institution / client type."""

    name: str
    category: str
    summary: str
    domains: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    typical_obligations: Tuple[str, ...] = ()


_INSTITUTION_TYPES_JSON_FILENAME = "institution_types.json"


def _resolve_institution_types_path() -> Path:
    """Return the path to the ``institution_types.json`` catalog.

    Honours the ``REGAI_INSTITUTION_TYPES_PATH`` environment variable so
    deployments can ship an operator-approved override without editing
    the source tree. When the override is missing / unreadable we fall
    back to the shipped default alongside this module.
    """
    override = os.environ.get("REGAI_INSTITUTION_TYPES_PATH")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate
        logger.warning(
            "REGAI_INSTITUTION_TYPES_PATH points to '%s' but the file was "
            "not found; falling back to the shipped default.", override,
        )
    return Path(__file__).with_name("data") / _INSTITUTION_TYPES_JSON_FILENAME


def _load_institution_catalog(path: Path) -> Tuple[InstitutionType, ...]:
    """Load and validate the JSON catalog.

    The loader is strict: every entry MUST declare ``name``, ``category``
    and ``summary``. Optional list fields default to empty tuples. Any
    row whose category is not in the file's declared ``categories``
    allow-list is dropped (with a warning) so the "Banking & Capital
    Markets only" invariant cannot silently regress.

    Raises :class:`FileNotFoundError` when the file is missing entirely,
    since a missing catalog would render the Page 1 dropdown empty and
    the applicability engine useless. Any other JSON / schema error is
    logged and re-raised for the same reason.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Institution catalog not found at {path}. Ship a valid JSON "
            f"file or set REGAI_INSTITUTION_TYPES_PATH."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed_categories = {str(c).strip() for c in (raw.get("categories") or []) if c}
    entries = raw.get("institution_types") or []
    if not isinstance(entries, list):
        raise ValueError(
            f"Institution catalog {path} does not contain an "
            f"'institution_types' list."
        )
    out: List[InstitutionType] = []
    seen_names: set = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip()
        category = str(entry.get("category") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if not name or not category or not summary:
            logger.warning(
                "Skipping institution entry with missing required fields: %r",
                entry,
            )
            continue
        if allowed_categories and category not in allowed_categories:
            logger.warning(
                "Skipping institution '%s' (category %r not in allowed "
                "categories %s).", name, category, sorted(allowed_categories),
            )
            continue
        if name in seen_names:
            logger.warning(
                "Duplicate institution name '%s' in catalog — keeping the "
                "first occurrence.", name,
            )
            continue
        seen_names.add(name)
        out.append(InstitutionType(
            name=name,
            category=category,
            summary=summary,
            domains=tuple(str(d) for d in (entry.get("domains") or [])),
            keywords=tuple(str(k) for k in (entry.get("keywords") or [])),
            typical_obligations=tuple(
                str(o) for o in (entry.get("typical_obligations") or [])
            ),
        ))
    if not out:
        raise ValueError(
            f"Institution catalog at {path} produced zero valid entries — "
            f"the UI dropdown would be empty."
        )
    return tuple(out)


# The catalog is loaded once at module import time from
# ``services/data/institution_types.json`` (or the file pointed to
# by ``$REGAI_INSTITUTION_TYPES_PATH``). This is the SINGLE SOURCE
# OF TRUTH for the Client Type / Institution Type dropdown and for
# every downstream applicability decision. New Banking or Capital
# Markets institution types can be onboarded by editing the JSON
# file — no Python edit needed.
INSTITUTION_TYPES: Tuple[InstitutionType, ...] = _load_institution_catalog(
    _resolve_institution_types_path(),
)


# Convenience: an ``{name -> InstitutionType}`` map used by lookups.
INSTITUTION_TYPES_BY_NAME: Dict[str, InstitutionType] = {
    row.name: row for row in INSTITUTION_TYPES
}


INSTITUTION_TYPE_NAMES: Tuple[str, ...] = tuple(row.name for row in INSTITUTION_TYPES)


def list_institution_types() -> List[Dict[str, Any]]:
    """Return the catalog as a list of dicts (for JSON / UI serialisation)."""
    return [
        {
            "name": t.name,
            "category": t.category,
            "summary": t.summary,
            "domains": list(t.domains),
            "typical_obligations": list(t.typical_obligations),
        }
        for t in INSTITUTION_TYPES
    ]


def get_institution_type(name: str) -> Optional[InstitutionType]:
    """Return the canonical entry for ``name`` (case-insensitive), else ``None``."""
    if not name:
        return None
    exact = INSTITUTION_TYPES_BY_NAME.get(name)
    if exact is not None:
        return exact
    lower = name.strip().lower()
    for t in INSTITUTION_TYPES:
        if t.name.lower() == lower:
            return t
    return None


def normalize_client_roles(
    roles: Optional[Sequence[str]],
    *,
    default: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return a de-duplicated list of canonical institution names.

    Unknown roles are dropped (they still appear in the raw session state, but
    the interpretation engine only reasons about roles it has business context
    for). When the resulting list is empty and ``default`` is supplied the
    default set is returned instead.
    """
    seen: List[str] = []
    seen_set: set = set()
    for raw in roles or []:
        if not raw:
            continue
        t = get_institution_type(str(raw))
        if t is None:
            continue
        if t.name not in seen_set:
            seen.append(t.name)
            seen_set.add(t.name)
    if not seen and default:
        return normalize_client_roles(default)
    return seen


# ---------------------------------------------------------------------------
# Role-aware interpretation output types
# ---------------------------------------------------------------------------


APPLICABILITY_APPLICABLE = "Applicable"
APPLICABILITY_PARTIAL = "Partially Applicable"
APPLICABILITY_NOT_APPLICABLE = "Not Applicable"
APPLICABILITY_UNCERTAIN = "Uncertain"

APPLICABILITY_ORDER: Tuple[str, ...] = (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_PARTIAL,
    APPLICABILITY_UNCERTAIN,
    APPLICABILITY_NOT_APPLICABLE,
)


@dataclass
class RoleApplicability:
    """Per-role applicability decision for a single obligation / section.

    ``applicability`` is one of :data:`APPLICABILITY_APPLICABLE`,
    :data:`APPLICABILITY_PARTIAL`, :data:`APPLICABILITY_NOT_APPLICABLE` or
    :data:`APPLICABILITY_UNCERTAIN`. ``confidence`` (0-100) reflects how
    strongly the underlying signals support the decision. ``matched_terms``
    and ``rationale`` provide the audit trail.
    """

    role: str
    applicability: str = APPLICABILITY_UNCERTAIN
    confidence: int = 50
    rationale: str = ""
    matched_terms: List[str] = field(default_factory=list)
    obligations_for_role: List[str] = field(default_factory=list)
    business_impacts: List[str] = field(default_factory=list)
    operational_impacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "applicability": self.applicability,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "matched_terms": list(self.matched_terms),
            "obligations_for_role": list(self.obligations_for_role),
            "business_impacts": list(self.business_impacts),
            "operational_impacts": list(self.operational_impacts),
        }


@dataclass
class RoleAwareInterpretation:
    """Bundle produced by the client role–aware interpretation engine.

    Attached to :class:`~models.workflow_models.RegulatoryAnalysis` so every
    downstream stage (Agent 2 RTM, Agent 3 questionnaire, Agent 4
    recommendations, dashboard, exports) can consult it instead of
    reinterpreting the regulation independently.
    """

    regulation: str
    client_roles: List[str]
    per_role_summary: Dict[str, str] = field(default_factory=dict)
    per_role_scope: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_obligation_applicability: Dict[str, List[RoleApplicability]] = field(
        default_factory=dict,
    )
    role_specific_obligations: Dict[str, List[str]] = field(default_factory=dict)
    compliance_expectations: Dict[str, List[str]] = field(default_factory=dict)
    business_impacts: Dict[str, List[str]] = field(default_factory=dict)
    operational_impacts: Dict[str, List[str]] = field(default_factory=dict)
    generated_by_ai: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regulation": self.regulation,
            "client_roles": list(self.client_roles),
            "per_role_summary": dict(self.per_role_summary),
            "per_role_scope": {
                role: dict(scope) for role, scope in self.per_role_scope.items()
            },
            "per_obligation_applicability": {
                key: [rr.to_dict() for rr in rows]
                for key, rows in self.per_obligation_applicability.items()
            },
            "role_specific_obligations": {
                role: list(items) for role, items
                in self.role_specific_obligations.items()
            },
            "compliance_expectations": {
                role: list(items) for role, items
                in self.compliance_expectations.items()
            },
            "business_impacts": {
                role: list(items) for role, items
                in self.business_impacts.items()
            },
            "operational_impacts": {
                role: list(items) for role, items
                in self.operational_impacts.items()
            },
            "generated_by_ai": self.generated_by_ai,
            "notes": list(self.notes),
        }

    def roles_for_obligation(self, obligation_id: str) -> List[str]:
        """Return the roles the obligation is Applicable / Partial for."""
        rows = self.per_obligation_applicability.get(obligation_id) or []
        applicable: List[str] = []
        for row in rows:
            if row.applicability in (APPLICABILITY_APPLICABLE,
                                     APPLICABILITY_PARTIAL):
                applicable.append(row.role)
        return applicable

    def is_obligation_in_scope(self, obligation_id: str) -> bool:
        """True when at least one selected role treats the obligation as
        Applicable, Partial, or Uncertain (Uncertain is kept in scope by
        default so the SME reviewer can adjudicate)."""
        rows = self.per_obligation_applicability.get(obligation_id) or []
        if not rows:
            return True
        return any(
            row.applicability != APPLICABILITY_NOT_APPLICABLE
            for row in rows
        )


# ---------------------------------------------------------------------------
# Deterministic role-aware interpretation engine
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-/]{2,}")


def _tokenize(text: str) -> List[str]:
    """Return a case-folded token bag suitable for keyword overlap scoring."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _keyword_hits(needles: Sequence[str], haystack_text: str) -> List[str]:
    """Return the subset of ``needles`` that appear inside ``haystack_text``."""
    if not haystack_text:
        return []
    haystack = haystack_text.lower()
    return sorted({
        needle for needle in needles
        if needle and needle.lower() in haystack
    })


def _score_applicability(
    role: InstitutionType,
    *,
    regulation_context: str,
    obligation_text: str,
) -> Tuple[str, int, List[str], str]:
    """Deterministic per-role applicability scorer.

    Returns ``(applicability, confidence, matched_terms, rationale)``.

    The score aggregates:

    * how many role-specific keywords appear in the *regulation context*
      (evidence that the regulation actually talks about this role); and
    * how many role-specific keywords appear in the *obligation text*
      (evidence that this particular obligation touches the role's business
      surface).

    When neither signal is available the engine returns ``Uncertain`` at low
    confidence so downstream agents can route the obligation for SME review
    instead of inventing an applicability decision (per the "no hallucination"
    requirement).
    """
    all_terms = tuple(role.keywords) + tuple(role.domains) + tuple(role.typical_obligations)
    obligation_hits = _keyword_hits(all_terms, obligation_text)
    context_hits = _keyword_hits(all_terms, regulation_context)

    obligation_score = min(3, len(obligation_hits))
    context_score = min(2, len(context_hits))
    total = obligation_score * 2 + context_score

    matched = sorted(set(obligation_hits) | set(context_hits))

    if total >= 5:
        applicability = APPLICABILITY_APPLICABLE
        confidence = min(95, 70 + total * 3)
        rationale = (
            f"Regulation text and obligation both reference {role.name}-"
            f"specific concepts ({', '.join(matched[:4])}). "
            f"Obligation is treated as directly applicable."
        )
    elif total >= 3:
        applicability = APPLICABILITY_PARTIAL
        confidence = 60 + total * 3
        rationale = (
            f"Obligation partly aligns with {role.name}'s regulatory "
            f"surface via {', '.join(matched[:4])}. Some elements may "
            f"require proportionality adjustments."
        )
    elif total >= 1:
        applicability = APPLICABILITY_PARTIAL
        confidence = 55
        rationale = (
            f"Weak but non-zero overlap with {role.name} "
            f"({', '.join(matched[:3])}). Applicability should be "
            f"validated by an SME."
        )
    else:
        applicability = APPLICABILITY_UNCERTAIN
        confidence = 45
        rationale = (
            f"No strong keyword or domain overlap between the obligation "
            f"and {role.name}'s regulatory surface. Cannot determine "
            f"applicability from the regulation text alone; flagging for "
            f"SME review rather than inventing a mapping."
        )
    return applicability, confidence, matched, rationale


def _obligation_text(obligation: Any) -> str:
    """Return the concatenated searchable text for one obligation.

    Works on both :class:`~models.workflow_models.Obligation` dataclasses and
    plain dicts (the engine is used by Agent 1 before the dataclass has been
    fully populated, and by exports that only carry dicts).
    """
    def _get(name: str) -> str:
        if isinstance(obligation, dict):
            return str(obligation.get(name) or "")
        return str(getattr(obligation, name, "") or "")

    return " ".join([
        _get("title"),
        _get("theme"),
        _get("compliance_requirement"),
        _get("impacted_area"),
        _get("impacted_function"),
        _get("regulatory_basis"),
        " ".join(_get("control_expectations") if not isinstance(obligation, dict)
                 else obligation.get("control_expectations") or []),
    ])


def derive_role_applicability(
    obligation: Any,
    roles: Sequence[str],
    *,
    regulation_context: str = "",
) -> List[RoleApplicability]:
    """Return one :class:`RoleApplicability` per selected role.

    The engine grounds every decision in either the regulation context or the
    obligation text itself. It never fabricates matches — when there is no
    signal, the returned applicability is ``Uncertain`` (not "Applicable" and
    not "Not Applicable") so downstream agents can flag the item for SME
    review rather than dropping or inventing coverage.
    """
    obligation_text = _obligation_text(obligation)
    out: List[RoleApplicability] = []
    for role_name in roles:
        role = get_institution_type(role_name)
        if role is None:
            out.append(RoleApplicability(
                role=role_name,
                applicability=APPLICABILITY_UNCERTAIN,
                confidence=40,
                rationale=(
                    "Selected role is not in the institution catalog. "
                    "Falling back to Uncertain so the item is not silently "
                    "excluded."
                ),
            ))
            continue

        applicability, confidence, matched, rationale = _score_applicability(
            role,
            regulation_context=regulation_context,
            obligation_text=obligation_text,
        )
        out.append(RoleApplicability(
            role=role.name,
            applicability=applicability,
            confidence=confidence,
            rationale=rationale,
            matched_terms=matched,
        ))
    return out


# ---------------------------------------------------------------------------
# High-level engine used by Agent 1
# ---------------------------------------------------------------------------


def _executive_summary_for_role(
    role: InstitutionType,
    regulation: str,
    *,
    applicable_count: int,
    partial_count: int,
    not_applicable_count: int,
    uncertain_count: int,
) -> str:
    """Per-role executive summary that is *actually* per-role.

    Weaves in ``role.summary`` and the role's single most-distinctive
    typical obligation so two roles never emit an identical summary —
    even when their applicability tallies happen to match.
    """
    total = applicable_count + partial_count + not_applicable_count + uncertain_count
    if total == 0:
        return (
            f"No obligations were evaluated against {role.name} "
            f"({role.category}). Regenerate the analysis with regulation "
            f"content attached to produce a role-specific interpretation."
        )
    parts: List[str] = []
    if applicable_count:
        parts.append(f"{applicable_count} directly applicable")
    if partial_count:
        parts.append(f"{partial_count} partially applicable")
    if uncertain_count:
        parts.append(f"{uncertain_count} uncertain / SME review")
    if not_applicable_count:
        parts.append(f"{not_applicable_count} out of scope")
    coverage = ", ".join(parts)

    distinctive = (
        role.typical_obligations[0]
        if role.typical_obligations else "supervisory expectations"
    )
    surfaces = ", ".join(role.domains[:3]) if role.domains else "its operating model"
    article = _indefinite_article(role.name)
    return (
        f"For {article} {role.name} ({role.category}) — {role.summary} — "
        f"{regulation} produces {coverage} obligations. The interpretation "
        f"is scoped to this institution's specific operating model "
        f"({surfaces}); the most distinctive obligation cluster for "
        f"{article} {role.name} is '{distinctive}'."
    )


# ---------------------------------------------------------------------------
# Angle rotation for per-role bullets
# ---------------------------------------------------------------------------
#
# Business, operational and compliance bullets are built by *rotating*
# through the angle lists below. Each angle contributes a different focal
# lens (Process / Controls / Data / Governance / …) so the four bullets
# generated for one role never read as clones with a single keyword
# swapped. Every bullet is also prefixed with the role name + category,
# so two roles that share a domain still emit role-distinguishable text.

_BUSINESS_IMPACT_ANGLES: Tuple[Tuple[str, str], ...] = (
    ("Process",         "reshape end-to-end processes around"),
    ("Controls",        "redesign preventive and detective controls covering"),
    ("Data & Reporting", "extend data lineage, evidence capture and management/regulator reporting for"),
    ("Technology",      "invest in platform capabilities, tooling and integrations that support"),
    ("Governance",      "raise board / committee oversight, RACI and accountability for"),
    ("Third-party",     "tighten third-party lifecycle, exit strategies and concentration risk around"),
    ("People",          "re-skill first- and second-line teams and update role mandates for"),
    ("Client-facing",   "adjust onboarding, disclosures and contractual terms tied to"),
)

_OPERATIONAL_ANGLES: Tuple[str, ...] = (
    "Playbook & runbook uplift for {theme}: trigger criteria, ownership, evidence retention.",
    "Detection & monitoring build-out for {theme}: KRIs, alerts and independent second-line challenge.",
    "Testing regime for {theme}: scheduled control tests, independent assurance and remediation SLAs.",
    "Reporting cadence for {theme}: management dashboards, board escalation and regulator submissions.",
    "Recovery capability for {theme}: response-time targets, dependency mapping and tabletop cadence.",
    "Training & awareness for {theme}: role-based curricula, attestations and refresher cycles.",
)

_COMPLIANCE_ANGLES: Tuple[str, ...] = (
    "Documented policies and standards for {domain}, refreshed on a defined cadence.",
    "Risk taxonomy for {domain} mapped to {regulation}'s articles and to control library IDs.",
    "Independent second-line challenge of {domain} with committee-level reporting.",
    "Regulator-facing evidence pack for {domain} kept current with sign-off trail.",
    "Metrics, thresholds and SLAs for {domain} with breach handling and escalation.",
    "Annual attestation and management-owned assurance plan for {domain}.",
)


def _role_flavour_pool(role: InstitutionType) -> List[str]:
    """Return a *rotating* pool of role-specific framing phrases.

    Rather than emitting one identical flavour tail on every bullet (which
    made bullets read as clones with one word swapped), we now cycle
    through a pool built from three sources:

    * the fixed role signature (``_role_flavour``);
    * each entry in ``role.typical_obligations`` — wrapped so it reads as
      a framing clause; and
    * the role's own one-liner summary.

    Any duplicates (case-insensitive) are dropped, so if a role happens
    to have a very short typical_obligations list the pool still returns
    unique entries.
    """
    seen: set = set()
    pool: List[str] = []

    def _add(item: str) -> None:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            pool.append(item)

    _add(_role_flavour(role))

    for ob in role.typical_obligations[:8]:
        _add(f"anchored to the {ob} obligation cluster")

    for domain in role.domains[:4]:
        _add(f"framed around the {domain} regulatory surface")

    if role.summary:
        _add(
            "consistent with the "
            + role.name
            + " operating model ("
            + role.summary.rstrip(".").lower()
            + ")"
        )

    _add(f"proportionate to {role.name.lower()} supervisory expectations")

    return pool


def _indefinite_article(word: str) -> str:
    """Return ``"an"`` when ``word`` starts with a vowel sound, else ``"a"``.

    Simple orthographic rule — the input is always an English institution
    name from the catalog, so we don't need to handle exotic edge cases
    (like "hour" vs "house"). Case-insensitive.
    """
    if not word:
        return "a"
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _role_flavour(role: InstitutionType) -> str:
    """Return a short role-specific modifier appended to bullets.

    The modifier is derived from tokens in the role name so two roles that
    share a domain still produce visibly different sentences. Broker Dealer
    (Small) → "proportionality-based"; Broker Dealer (Large) → "systemic
    footprint-scaled"; Digital Bank → "digital-native"; and so on.
    """
    name = role.name.lower()
    if "(small)" in name:
        return "scaled proportionally to a small operating footprint"
    if "(mid" in name:
        return "sized for a mid-scale operating footprint"
    if "(large)" in name:
        return "matched to a systemic / cross-asset footprint"
    if "digital" in name or "neo" in name or "challenger" in name:
        return "delivered through digital-native channels and cloud-hosted platforms"
    if "central bank" in name:
        return "aligned to the institution's monetary-authority and oversight mandate"
    if "custodian" in name:
        return "anchored on asset-servicing and safekeeping obligations"
    if "clearing" in name or "ccp" in name or "csd" in name:
        return "calibrated to systemically important market-infrastructure duties"
    if "insurance" in name or "reinsurance" in name:
        return "framed through Solvency II policyholder-protection lenses"
    if "fintech" in name or "regtech" in name:
        return "delivered through a technology-first operating model"
    if "asset manager" in name or "hedge fund" in name or "wealth" in name or "mutual fund" in name:
        return "aligned to UCITS / AIFMD investor-protection duties"
    if "payment" in name or "emi" in name or "electronic money" in name:
        return "aligned to PSD2 / EMD safeguarding and SCA duties"
    if "cooperative" in name or "credit union" in name or "savings" in name:
        return "aligned to member / depositor protection principles"
    if "corporate bank" in name:
        return "framed around wholesale-client credit and trade-finance risk"
    if "private bank" in name:
        return "framed around HNW suitability and cross-border tax duties"
    if "retail bank" in name:
        return "framed around consumer-protection and conduct-risk duties"
    if "commercial bank" in name or "universal bank" in name:
        return "framed around prudential (CRR/CRD) and consolidated-supervision duties"
    if "investment bank" in name:
        return "framed around MiFID trading, market-abuse and best-execution duties"
    if "housing finance" in name or "mortgage" in name:
        return "framed around responsible-lending and mortgage-conduct duties"
    if "leasing" in name or "consumer finance" in name or "nbfc" in name:
        return "framed around non-bank consumer-credit duties"
    return f"scoped to a {role.category.lower()} operating model"


def build_role_aware_interpretation(
    *,
    regulation: str,
    client_roles: Sequence[str],
    regulation_context: str = "",
    obligations: Optional[Sequence[Any]] = None,
) -> RoleAwareInterpretation:
    """Build a :class:`RoleAwareInterpretation` for the selected client roles.

    This is the deterministic offline engine. Agent 1 may overlay an LLM
    interpretation on top of the returned bundle when a GenAI client is
    available; when the client is offline the returned bundle is used
    verbatim so the platform still produces role-specific outputs.
    """
    roles = normalize_client_roles(client_roles) or []
    interp = RoleAwareInterpretation(
        regulation=regulation,
        client_roles=list(roles),
    )

    role_records = [get_institution_type(r) for r in roles]
    role_records = [r for r in role_records if r is not None]

    # Per-role scope block — populated from the catalog metadata.
    for role in role_records:
        interp.per_role_scope[role.name] = {
            "category": role.category,
            "summary": role.summary,
            "domains": list(role.domains),
            "typical_obligations": list(role.typical_obligations),
        }
        interp.role_specific_obligations.setdefault(role.name, [])
        interp.business_impacts.setdefault(role.name, [])
        interp.operational_impacts.setdefault(role.name, [])
        interp.compliance_expectations.setdefault(role.name, [])

    obligations = list(obligations or [])

    # Per-role tallies for the executive summary line.
    tallies: Dict[str, Dict[str, int]] = {
        role.name: {
            APPLICABILITY_APPLICABLE: 0,
            APPLICABILITY_PARTIAL: 0,
            APPLICABILITY_NOT_APPLICABLE: 0,
            APPLICABILITY_UNCERTAIN: 0,
        } for role in role_records
    }

    for obligation in obligations:
        obligation_id = (
            obligation.get("obligation_id") if isinstance(obligation, dict)
            else getattr(obligation, "obligation_id", None)
        )
        if not obligation_id:
            continue

        per_role = derive_role_applicability(
            obligation, [r.name for r in role_records],
            regulation_context=regulation_context,
        )
        interp.per_obligation_applicability[obligation_id] = per_role

        for row in per_role:
            tallies.setdefault(row.role, {}).setdefault(row.applicability, 0)
            tallies[row.role][row.applicability] += 1
            if row.applicability in (APPLICABILITY_APPLICABLE, APPLICABILITY_PARTIAL):
                title = (
                    obligation.get("title") if isinstance(obligation, dict)
                    else getattr(obligation, "title", "")
                )
                if title and title not in interp.role_specific_obligations[row.role]:
                    interp.role_specific_obligations[row.role].append(str(title))

    for role in role_records:
        counts = tallies.get(role.name, {})
        summary = _executive_summary_for_role(
            role,
            regulation,
            applicable_count=counts.get(APPLICABILITY_APPLICABLE, 0),
            partial_count=counts.get(APPLICABILITY_PARTIAL, 0),
            not_applicable_count=counts.get(APPLICABILITY_NOT_APPLICABLE, 0),
            uncertain_count=counts.get(APPLICABILITY_UNCERTAIN, 0),
        )
        interp.per_role_summary[role.name] = summary

        # Build per-role business / operational / compliance bullets by
        # rotating through the angle catalogs so each bullet within one
        # role uses a *different* focal lens (Process, Controls, Data,
        # Governance, …), a *different* framing tail (from the flavour
        # pool), and — where possible — a different domain / theme.
        # This prevents both intra-role duplication ("same points in one
        # role") and cross-role duplication ("all similar for every
        # role").
        flavour_pool = _role_flavour_pool(role)
        domain_pool = list(role.domains) or ["its regulatory surface"]
        theme_pool = list(role.typical_obligations) or [
            "supervisory expectations",
        ]
        article = _indefinite_article(role.name)

        # Business impacts — one bullet per angle, cycling domains AND
        # flavour tails so both the middle and the end of the sentence
        # change bullet-to-bullet. Bullet count = min(6, angles).
        bi: List[str] = []
        seen_bi: set = set()
        bullet_count = min(6, max(len(domain_pool), len(_BUSINESS_IMPACT_ANGLES)))
        for idx in range(min(bullet_count, len(_BUSINESS_IMPACT_ANGLES))):
            angle_label, angle_verb = _BUSINESS_IMPACT_ANGLES[idx]
            domain = domain_pool[idx % len(domain_pool)]
            flavour = flavour_pool[idx % len(flavour_pool)]
            impact = (
                f"[{angle_label}] As {article} {role.name} ({role.category}), "
                f"{regulation} obligations will {angle_verb} '{domain}' — "
                f"{flavour}."
            )
            if impact.lower() not in seen_bi:
                bi.append(impact)
                seen_bi.add(impact.lower())
        interp.business_impacts[role.name] = bi

        # Operational impacts — one bullet per operational angle, cycled
        # through the role's obligation themes AND a rotating flavour
        # tail so successive bullets vary in phrasing, focus and framing.
        op_bullets: List[str] = []
        seen_op: set = set()
        for idx in range(min(6, len(_OPERATIONAL_ANGLES))):
            theme = theme_pool[idx % len(theme_pool)]
            base = _OPERATIONAL_ANGLES[idx]
            # Rotate through the flavour pool starting at a different
            # offset than the business-impacts loop, so an operational
            # bullet with the same index doesn't share its tail with
            # the corresponding business-impact bullet.
            flavour = flavour_pool[(idx + 2) % len(flavour_pool)]
            op = (
                f"{role.name}: {base.format(theme=theme)} — {flavour}."
            )
            if op.lower() not in seen_op:
                op_bullets.append(op)
                seen_op.add(op.lower())
        interp.operational_impacts[role.name] = op_bullets

        # Compliance expectations — one bullet per compliance angle,
        # cycled through domains AND a rotating flavour tail so the six
        # accountability lenses (policy → taxonomy → 2LOD → evidence →
        # metrics → attestation) each read differently even for the
        # same role.
        ce_bullets: List[str] = []
        seen_ce: set = set()
        for idx in range(min(6, len(_COMPLIANCE_ANGLES))):
            domain = domain_pool[idx % len(domain_pool)]
            base = _COMPLIANCE_ANGLES[idx]
            flavour = flavour_pool[(idx + 4) % len(flavour_pool)]
            ce = (
                f"{role.name} ({role.category}): "
                + base.format(domain=domain, regulation=regulation)
                + f" ({flavour})"
            )
            if ce.lower() not in seen_ce:
                ce_bullets.append(ce)
                seen_ce.add(ce.lower())
        interp.compliance_expectations[role.name] = ce_bullets

    if not role_records:
        interp.notes.append(
            "No client role was selected. The pipeline will fall back to a "
            "generic interpretation, but this reduces auditability. Select "
            "one or more institution types on Page 1 to enable role-aware "
            "reasoning."
        )

    return interp


__all__ = [
    "APPLICABILITY_APPLICABLE",
    "APPLICABILITY_NOT_APPLICABLE",
    "APPLICABILITY_ORDER",
    "APPLICABILITY_PARTIAL",
    "APPLICABILITY_UNCERTAIN",
    "INSTITUTION_TYPES",
    "INSTITUTION_TYPES_BY_NAME",
    "INSTITUTION_TYPE_NAMES",
    "InstitutionType",
    "RoleApplicability",
    "RoleAwareInterpretation",
    "build_role_aware_interpretation",
    "derive_role_applicability",
    "get_institution_type",
    "list_institution_types",
    "normalize_client_roles",
]
