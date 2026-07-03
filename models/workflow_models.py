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
from typing import Any, Dict, List, Optional


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
    confidence: int = 92
    #: Citations attached to this obligation, copied from the BRD
    #: requirement that produced it. Each entry is a plain dict with the
    #: ``SourceReference`` shape (source_url, title, regulator,
    #: publication_date, regulation_reference, source_type, ...). When the
    #: list is empty the upstream BRD requirement was not anchored to a
    #: retrieved publication; the downstream UI surfaces this gap explicitly
    #: instead of fabricating a citation.
    source_references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RegulatoryAnalysis:
    """Bundle returned by Agent 1.

    Carries the full obligation list, an executive summary of impacted areas
    and themes, and a back-reference to the generated BRD (``brd_report``) so
    Agent 2 does not have to regenerate it. ``used_genai`` and ``metadata``
    surface how the analysis was produced (GenAI vs deterministic fallback).
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
    """One row of the Requirements Traceability Matrix."""

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
    #: Citations attached to this RTM row. Mirrors the obligation's source
    #: list so the matrix stays self-contained when exported as CSV/JSON
    #: (downstream consumers don't need to join back to the obligation).
    source_references: List[Dict[str, Any]] = field(default_factory=list)


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
class ScoringResult:
    """Deterministic Python rules-engine output.

    ``evaluation`` holds the rich dict produced by
    :func:`services.scoring_engine.evaluate` (pair scores, area summary,
    function summary, requirement scores, etc.). ``top_gaps`` is a convenience
    cache used by Agent 4.
    """

    evaluation: Dict[str, Any]
    top_gaps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def compliance_score_pct(self) -> float:
        return float(self.evaluation.get("compliance_score_pct") or 0.0)

    @property
    def evaluation_confidence_pct(self) -> float:
        return float(self.evaluation.get("evaluation_confidence_pct") or 0.0)


# ---------------------------------------------------------------------------
# Stage 7 — Agent 4 output (Recommendations)
# ---------------------------------------------------------------------------

@dataclass
class RecommendationResult:
    """Recommendations bundle returned by Agent 4.

    ``recommendations`` is a list of either ``Recommendation`` dataclasses
    (from :mod:`services.recommendation_service`) or plain dicts when restored
    from persistence — the dashboard handles both shapes.
    """

    recommendations: List[Any] = field(default_factory=list)
    severity_filter: str = "Watch"
    top_n_requirements: int = 10
    used_genai: bool = False
