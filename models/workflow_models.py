"""Intermediate workflow artefacts exchanged between agents.

These dataclasses give every step of the Regulatory Impact & Readiness pipeline
a clearly-named contract so the orchestrator can wire agents together without
each agent reaching into another agent's internals.

Pipeline reminder (see ``orchestrator.py`` for the full diagram):

    Upload Regulation
        -> ParsedDocument
    Agent 1 (Regulatory Analysis)
        -> RegulatoryAnalysis (carries Obligation list)
    Agent 2 (BRD + RTM)
        -> BRDArtifact + RTMArtifact
    Agent 3 (Questionnaire Generation)
        -> QuestionnairePackage
    User Responses
        -> AssessmentResponse
    Python Rules Engine
        -> ScoringResult
    Agent 4 (Recommendations)
        -> RecommendationResult

All models are simple dataclasses with JSON-friendly field types so they can be
persisted via :mod:`services.persistence` or serialised for export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Stage 1 — input
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """Output of the Document Parser stage.

    Wraps the text extracted from an uploaded regulation/BRD/FRD file so the
    rest of the pipeline can stay agnostic of file format.
    """

    name: str
    kind: str  # 'regulation' | 'brd' | 'frd' | 'other'
    text: str
    source_path: Optional[str] = None
    page_count: Optional[int] = None
    mime: Optional[str] = None
    warning_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.text or "").strip()


# ---------------------------------------------------------------------------
# Stage 2 — Agent 1 outputs (Regulatory Analysis + Obligations)
# ---------------------------------------------------------------------------

@dataclass
class Obligation:
    """One discrete regulatory obligation extracted by Agent 1.

    Fields are intentionally aligned with how downstream agents reason about
    BRDs, RTMs and assessment questions:

    - ``impacted_area``/``impacted_function`` feed Agent 2 (RTM) and Agent 3.
    - ``control_expectations``/``evidence_needs`` feed Agent 2's RTM and the
      questionnaire's evidence prompts.
    - ``source_requirement_id`` carries back-traceability to the BRD row that
      generated the obligation (or to an upstream taxonomy item if no BRD row
      exists yet).
    - ``applicable_roles``/``partial_roles``/``not_applicable_roles``/
      ``uncertain_roles`` capture the Client Role-Aware interpretation
      produced by Agent 1 for every selected institution type. Downstream
      agents (BRD, RTM, questionnaire, recommendations) consult these lists
      instead of reinterpreting the regulation independently.
    - ``role_applicability`` carries the per-role explainability record
      (matched regulation terms, rationale) so BRD / RTM / questionnaire
      renderers can quote **why** the obligation applies to a role.
    """

    obligation_id: str
    title: str
    theme: str
    compliance_requirement: str
    impacted_area: str
    impacted_function: str
    deadline: Optional[str] = None
    control_expectations: List[str] = field(default_factory=list)
    evidence_needs: List[str] = field(default_factory=list)
    risk_implication: str = ""
    source_requirement_id: str = ""
    regulatory_basis: str = ""
    priority: str = "Should"
    #: Canonical regulatory obligation verb detected in the source text:
    #: ``"Must" | "Shall" | "Should" | "May" | "Can" | ""``. Populated by
    #: :mod:`services.obligation_verb` from the obligation's title +
    #: compliance requirement + regulatory basis. Empty string means the
    #: classifier could not find a canonical verb in the source; downstream
    #: code should treat that as "unknown / SME review" rather than as
    #: "not mandatory". Kept distinct from ``priority`` (MoSCoW) because
    #: the two describe different things: ``obligation_verb`` is what the
    #: regulator said; ``priority`` is how the delivery team sequences it.
    obligation_verb: str = ""
    #: Per-obligation baseline confidence. The AI Assessment Intelligence
    #: service produces the authoritative :class:`ConfidenceAssessment` that
    #: is surfaced to the user; this attribute stays for BRD interop.
    confidence: int = 92
    #: Citations attached to this obligation, copied from the BRD
    #: requirement that produced it. Each entry is a plain dict with the
    #: ``SourceReference`` shape (source_url, title, regulator,
    #: publication_date, regulation_reference, source_type, ...). When the
    #: list is empty the upstream BRD requirement was not anchored to a
    #: retrieved publication; the downstream UI surfaces this gap explicitly
    #: instead of fabricating a citation.
    source_references: List[Dict[str, Any]] = field(default_factory=list)
    #: Client roles for which this obligation is Applicable (canonical
    #: institution names from :mod:`services.client_roles`). Empty when no
    #: client roles were selected; downstream code then treats the obligation
    #: as generic.
    applicable_roles: List[str] = field(default_factory=list)
    #: Client roles for which the obligation is Partially Applicable.
    partial_roles: List[str] = field(default_factory=list)
    #: Client roles for which the obligation is Not Applicable (explicitly
    #: out-of-scope). Retained on the obligation so exports can surface the
    #: "out of scope for X but in scope for Y" narrative.
    not_applicable_roles: List[str] = field(default_factory=list)
    #: Client roles for which applicability could not be determined from the
    #: regulation text alone. Flagged for SME review instead of dropped.
    uncertain_roles: List[str] = field(default_factory=list)
    #: Per-role explainability records. Each dict has the
    #: :class:`~services.client_roles.RoleApplicability` shape (role,
    #: applicability, confidence, rationale, matched_terms, ...).
    role_applicability: List[Dict[str, Any]] = field(default_factory=list)
    #: Free-form summary of how the obligation is interpreted for the
    #: selected client roles. Populated by Agent 1's role-aware interpretation
    #: engine.
    role_interpretation: str = ""

    @property
    def in_scope_roles(self) -> List[str]:
        """Roles for which the obligation is Applicable, Partial, or Uncertain.

        Uncertain roles are kept in scope by default so the SME reviewer can
        adjudicate rather than silently dropping the obligation.
        """
        merged: List[str] = []
        for bucket in (self.applicable_roles, self.partial_roles, self.uncertain_roles):
            for role in bucket:
                if role and role not in merged:
                    merged.append(role)
        return merged

    def is_applicable_for(self, roles: Optional[Sequence[str]]) -> bool:
        """Return True if any of ``roles`` is Applicable / Partial / Uncertain.

        Returns True when ``roles`` is empty (no client-role filter set) so
        the pipeline degrades gracefully to the pre-role-aware behaviour.
        """
        if not roles:
            return True
        role_set = {r for r in roles if r}
        if not role_set:
            return True
        return bool(role_set.intersection(self.in_scope_roles))


@dataclass
class RegulatoryAnalysis:
    """Bundle returned by Agent 1.

    Carries the full obligation list, an executive summary of impacted areas
    and themes, and a back-reference to the generated BRD (``brd_report``) so
    Agent 2 does not have to regenerate it. ``used_genai`` and ``metadata``
    surface how the analysis was produced (GenAI vs deterministic fallback).

    ``client_roles`` records the institution types the analysis was scoped
    to (e.g. ``["Commercial Bank", "Broker Dealer (Small)"]``). Downstream
    agents consult ``role_interpretation`` and the per-obligation
    ``applicable_roles`` fields — they must not reinterpret the regulation
    independently.
    """

    regulation: str
    tier: str
    summary: str
    impacted_areas: List[str]
    obligation_themes: List[str]
    obligations: List[Obligation] = field(default_factory=list)
    used_genai: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    brd_report: Any = None  # services.brd_frd_generator.DoraDetailedBRD or None
    #: Canonical institution names the analysis was scoped to (see
    #: :mod:`services.client_roles`). Empty list = generic (pre-role-aware)
    #: interpretation.
    client_roles: List[str] = field(default_factory=list)
    #: Client Role-Aware Regulatory Interpretation payload produced by the
    #: engine in :mod:`services.client_roles` (and optionally refined by
    #: the LLM). Serialised as a plain dict so it can be persisted /
    #: exported without pulling the engine's dataclass into every caller.
    role_interpretation: Dict[str, Any] = field(default_factory=dict)
    #: Client Profile keyword bundle collected on Page 1 (see
    #: :mod:`services.client_profile`). Keys: ``organization_profile``,
    #: ``business_lines``, ``products_in_scope``, ``countries_of_operation``,
    #: ``legal_entities``, ``vendor_third_parties``. Each value is a list
    #: of keywords (curated + free-form). Empty dict when the user has not
    #: populated the profile.
    client_profile: Dict[str, List[str]] = field(default_factory=dict)

    def obligations_for_roles(
        self, roles: Optional[Sequence[str]] = None,
    ) -> List[Obligation]:
        """Return only the obligations that are in scope for ``roles``.

        Falls back to ``self.client_roles`` when the caller does not supply
        an explicit filter. When neither list has values the full obligation
        list is returned (pre-role-aware behaviour).
        """
        filter_roles = list(roles) if roles is not None else list(self.client_roles)
        if not filter_roles:
            return list(self.obligations)
        return [o for o in self.obligations if o.is_applicable_for(filter_roles)]


# ---------------------------------------------------------------------------
# Stage 3 — Agent 2 outputs (BRD + RTM)
# ---------------------------------------------------------------------------

@dataclass
class BRDArtifact:
    """In-memory BRD/FRD report plus its export metadata.

    Wraps the ``DoraDetailedBRD`` Pydantic model produced by
    :mod:`services.brd_frd_generator` so downstream stages do not need to
    import the Pydantic class directly.
    """

    report: Any  # services.brd_frd_generator.DoraDetailedBRD
    metadata: Dict[str, Any] = field(default_factory=dict)
    docx_path: Optional[str] = None
    source: str = "generated"  # 'generated' | 'uploaded' | 'sample'


@dataclass
class RTMEntry:
    """One row of the Requirements Traceability Matrix.

    Every requirement in the matrix carries traceability to:

    * the regulation section (``regulatory_basis``),
    * the underlying regulatory requirement (``business_requirement``),
    * the applicable institution type(s) (``applicable_roles``),
    * the business interpretation (``business_interpretation``),
    * the functional impact (``functional_requirement``), and
    * the business justification (``business_justification``).

    Requirements that do not apply to the selected institution type(s) are
    still emitted (so the "out of scope for X" narrative is preserved) but
    flagged via ``out_of_scope``.
    """

    traceability_id: str
    obligation_id: str
    business_requirement_id: str
    functional_requirement_id: Optional[str]
    business_requirement: str
    functional_requirement: str
    impacted_area: str
    impacted_function: str
    system_process_impact: str
    evidence_required: str
    regulatory_basis: str = ""
    priority: str = "Should"
    #: Canonical regulatory verb (``Must / Shall / Should / May / Can``)
    #: copied from the source obligation. Kept on the RTM entry so exports
    #: are self-contained without needing to join back to the obligation.
    #: Empty string when the classifier could not detect a verb in the
    #: source text.
    obligation_verb: str = ""
    #: Citations attached to this RTM row. Mirrors the obligation's source
    #: list so the matrix stays self-contained when exported as CSV/JSON
    #: (downstream consumers don't need to join back to the obligation).
    source_references: List[Dict[str, Any]] = field(default_factory=list)
    #: Canonical institution names for which this requirement is Applicable
    #: or Partially Applicable. Empty when no client roles were selected.
    applicable_roles: List[str] = field(default_factory=list)
    #: Roles for which the requirement is explicitly Not Applicable.
    not_applicable_roles: List[str] = field(default_factory=list)
    #: When True, the row was preserved for audit ("out of scope for the
    #: selected client(s)") but should not drive downstream questionnaire
    #: or scoring content.
    out_of_scope: bool = False
    #: Business interpretation of the requirement for the selected roles.
    business_interpretation: str = ""
    #: Why the requirement matters for the selected roles (business
    #: justification / risk-implication summary).
    business_justification: str = ""
    #: Per-role rationale strings (``{role -> reason}``) so the exported
    #: matrix can quote why each institution type is / is not in scope.
    role_rationale: Dict[str, str] = field(default_factory=dict)


@dataclass
class RTMArtifact:
    """Collection of RTM entries plus quick lookup metadata."""

    entries: List[RTMEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 4 — Agent 3 output (Questionnaire)
# ---------------------------------------------------------------------------

@dataclass
class QuestionnairePackage:
    """Wrapper around the existing questionnaire-package dict.

    The package dict's schema is the contract validated by
    :mod:`utils.json_utils` and produced by
    :mod:`services.questionnaire_generator` — we keep it verbatim so existing
    sample files / exports / Excel writer continue to work unchanged.
    """

    package: Dict[str, Any]
    source: str = "generated_brd"  # 'generated_brd' | 'uploaded_brd' | 'uploaded_json' | 'db'
    questionnaire_id: Optional[int] = None
    name: Optional[str] = None

    @property
    def question_count(self) -> int:
        return len(self.package.get("questions") or [])

    @property
    def requirement_count(self) -> int:
        return len(self.package.get("requirements") or [])


# ---------------------------------------------------------------------------
# Stage 5 — User Responses / Scoring inputs
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResponse:
    """Single user response to a question."""

    question_id: str
    answer: Any
    comments: Optional[str] = None
    display_sequence: Optional[int] = None


# ---------------------------------------------------------------------------
# Stage 6 — Rules Engine output
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceAssessment:
    """AI-generated confidence / accuracy score with explanation.

    Replaces the hard-coded 90-100 confidence clamping used across the pipeline.
    The confidence score reflects four evidence-driven signals:

    * ``completeness_score`` — how thoroughly the regulation analysis captured
      the surface area of the source text.
    * ``quality_score``      — quality and consistency of extracted
      requirements.
    * ``evidence_score``     — availability of supporting evidence
      (citations, RTS/ITS, article references).
    * ``clarity_score``      — clarity of mapping between the regulation and
      generated outputs.

    Each sub-score is a float in ``[0, 100]``. ``overall_score`` is the
    weighted composite. ``reasoning`` explains, in plain English, why the
    model landed on that score.
    """

    overall_score: float
    completeness_score: float
    quality_score: float
    evidence_score: float
    clarity_score: float
    reasoning: str = ""
    generated_by_ai: bool = False
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactDimension:
    """One dimension of an impact assessment (e.g. Systems, Data, Controls).

    Consulting-grade impact assessments identify affected areas across
    multiple lenses. Every dimension captures the discrete items affected,
    the severity of the impact, and the reasoning that justifies it.
    """

    dimension: str  # 'business_functions' | 'processes' | 'systems' | ...
    items: List[str] = field(default_factory=list)
    severity: str = "Medium"  # 'Critical' | 'High' | 'Medium' | 'Low'
    severity_score: float = 50.0  # 0-100
    rationale: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class ImpactAssessment:
    """AI-generated regulatory impact assessment, consulting-grade.

    Rather than reducing "impact" to ``100 - readiness``, the impact
    assessment identifies:

    * affected business functions
    * affected processes
    * affected systems and applications
    * affected data
    * affected controls
    * affected stakeholders

    ...and quantifies severity per dimension with supporting reasoning.
    """

    regulation: str = ""
    executive_summary: str = ""
    overall_severity: str = "Medium"
    overall_severity_score: float = 50.0
    business_functions: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="business_functions"),
    )
    processes: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="processes"),
    )
    systems: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="systems"),
    )
    data: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="data"),
    )
    controls: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="controls"),
    )
    stakeholders: ImpactDimension = field(
        default_factory=lambda: ImpactDimension(dimension="stakeholders"),
    )
    generated_by_ai: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dimensions(self) -> List[ImpactDimension]:
        return [
            self.business_functions,
            self.processes,
            self.systems,
            self.data,
            self.controls,
            self.stakeholders,
        ]


@dataclass
class ReadinessDimension:
    """One dimension of a readiness assessment.

    Enterprise readiness reviews evaluate multiple facets — existing
    controls, process maturity, policy coverage, technology, documentation,
    gaps, and overall preparedness. Each is captured as a discrete
    dimension with a maturity score and rationale.
    """

    dimension: str
    maturity_level: str = "Developing"  # 'Optimised' | 'Managed' | ...
    score: float = 50.0  # 0-100
    rationale: str = ""
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class ReadinessAssessment:
    """AI-generated regulatory readiness assessment, consulting-grade.

    Reports on the seven consulting-standard readiness dimensions plus an
    overall preparedness score and executive summary.
    """

    regulation: str = ""
    executive_summary: str = ""
    overall_score: float = 50.0
    overall_level: str = "Developing"
    existing_controls: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="existing_controls"),
    )
    process_maturity: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="process_maturity"),
    )
    policy_coverage: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="policy_coverage"),
    )
    technology_readiness: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="technology_readiness"),
    )
    documentation_completeness: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="documentation_completeness"),
    )
    implementation_gaps: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="implementation_gaps"),
    )
    organizational_preparedness: ReadinessDimension = field(
        default_factory=lambda: ReadinessDimension(dimension="organizational_preparedness"),
    )
    key_gaps: List[str] = field(default_factory=list)
    key_strengths: List[str] = field(default_factory=list)
    generated_by_ai: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dimensions(self) -> List[ReadinessDimension]:
        return [
            self.existing_controls,
            self.process_maturity,
            self.policy_coverage,
            self.technology_readiness,
            self.documentation_completeness,
            self.implementation_gaps,
            self.organizational_preparedness,
        ]


@dataclass
class ScoringResult:
    """Deterministic Python rules-engine output.

    ``evaluation`` holds the rich dict produced by
    :func:`services.scoring_engine.evaluate` (pair scores, area summary,
    function summary, requirement scores, etc.). ``top_gaps`` is a convenience
    cache used by Agent 4.
    """

    evaluation: Dict[str, Any]
    top_gaps: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[ConfidenceAssessment] = None
    impact: Optional[ImpactAssessment] = None
    readiness: Optional[ReadinessAssessment] = None

    @property
    def compliance_score_pct(self) -> float:
        return float(self.evaluation.get("compliance_score_pct") or 0.0)

    @property
    def evaluation_confidence_pct(self) -> float:
        if self.confidence is not None:
            return float(self.confidence.overall_score)
        return float(self.evaluation.get("evaluation_confidence_pct") or 0.0)


# ---------------------------------------------------------------------------
# Stage 7 — Agent 4 output (Recommendations)
# ---------------------------------------------------------------------------

@dataclass
class RichRecommendation:
    """Consulting-grade actionable recommendation with full context.

    Every rich recommendation contains the six elements a regulatory
    consultant would expect on a partner-signed deliverable: **what** needs
    to be done, **why** it is important, **how** to implement it, its
    **priority**, the **expected outcome**, and any **dependencies** that
    need to be resolved first. Recommendations are keyed to the specific
    impacted area, function and regulatory requirements so no two look
    generic.
    """

    recommendation_id: str
    title: str
    area: str
    function: str
    priority: str  # 'High' | 'Medium' | 'Low'
    severity: str = "Watch"
    horizon: str = "Short-term"
    what: str = ""
    why: str = ""
    how: str = ""
    expected_outcome: str = ""
    dependencies: List[str] = field(default_factory=list)
    owner: str = ""
    mapped_requirement_ids: List[str] = field(default_factory=list)
    mapped_obligation_ids: List[str] = field(default_factory=list)
    identified_gap: str = ""
    regulatory_rationale: str = ""
    business_impact: str = ""
    implementation_steps: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    short_term_actions: List[str] = field(default_factory=list)
    long_term_actions: List[str] = field(default_factory=list)
    quick_wins: List[str] = field(default_factory=list)
    generated_by_ai: bool = False


@dataclass
class RecommendationResult:
    """Recommendations bundle returned by Agent 4.

    ``recommendations`` is a list of either ``Recommendation`` dataclasses
    (from :mod:`services.recommendation_service`) or plain dicts when restored
    from persistence — the dashboard handles both shapes.

    ``rich_recommendations`` is the new consulting-grade structured
    recommendation list; when present the UI prefers it over the legacy
    ``recommendations`` list.
    """

    recommendations: List[Any] = field(default_factory=list)
    severity_filter: str = "Watch"
    top_n_requirements: int = 10
    used_genai: bool = False
    rich_recommendations: List[RichRecommendation] = field(default_factory=list)
