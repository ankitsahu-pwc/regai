"""End-to-end orchestrator for the Regulatory Impact & Readiness pipeline.

Pipeline
--------
::

    Upload Regulation
        |
        v
    Document Parser            (services.document_parser.parse_document)
        |  -> ParsedDocument
        v
    Agent 1: Regulatory Analysis
        |  -> RegulatoryAnalysis (includes Obligation[])
        v
    Agent 2: BRD + RTM
        |  -> BRDArtifact + RTMArtifact
        v
    Agent 3: Questionnaire Generation
        |  -> QuestionnairePackage
        v
    User Responses                    (collected by the Streamlit UI)
        |
        v
    Python Rules Engine        (services.scoring_engine.evaluate)
        |  -> ScoringResult
        v
    Agent 4: Recommendations
        |  -> RecommendationResult
        v
    Dashboard                          (rendered by app.py)

Design goals
------------
* ``app.py`` should call orchestrator methods and never reach into individual
  agents or services to keep the workflow stages explicit.
* Every stage is idempotent and can be run in isolation — the orchestrator
  does not enforce a fixed order beyond data dependencies. The UI runs the
  stages incrementally as the user progresses through the cockpit pages.
* GenAI access is optional: agents are instantiated with an optional
  :class:`~services.genai_service.GenAIClient`. When the client is ``None``
  the deterministic offline fallbacks (already in :mod:`services`) are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from agents import (
    BRDRTMAgent,
    QuestionnaireAgent,
    RecommendationAgent,
    RegulatoryAnalysisAgent,
)
from models.workflow_models import (
    BRDArtifact,
    ParsedDocument,
    QuestionnairePackage,
    ReadinessAssessment,
    RecommendationResult,
    RegulatoryAnalysis,
    RTMArtifact,
    ScoringResult,
)
from services import document_parser
from services.genai_service import GenAIClient
from services.regulatory_intelligence_service import (
    RegulatoryIntelligencePackage,
    gather_regulatory_intelligence,
)
from services.scoring_engine import AssessmentState, applicable_base_questions, evaluate

StatusCallback = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


class RegulatoryWorkflowOrchestrator:
    """Single coordination object held by ``app.py``.

    Responsibilities
    ----------------
    * Owns the agent instances (so callers can share one configured GenAI
      client across them).
    * Exposes one method per pipeline stage. Each method is independently
      testable.
    * Provides a single ``run_full_pipeline`` helper that chains stages 1-3
      end-to-end for non-UI callers (smoke tests, batch runs).
    """

    def __init__(self, *, client: Optional[GenAIClient] = None) -> None:
        self.client = client
        self.regulatory_analysis_agent = RegulatoryAnalysisAgent(client=client)
        self.brd_rtm_agent = BRDRTMAgent()
        self.questionnaire_agent = QuestionnaireAgent()
        self.recommendation_agent = RecommendationAgent(client=client)

    # ------------------------------------------------------------------
    # Stage: Document Parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_document(path: Path, *, kind: str = "regulation") -> ParsedDocument:
        """Read a PDF/DOCX from disk into a :class:`ParsedDocument`."""
        return document_parser.parse_document(path, kind=kind)

    # ------------------------------------------------------------------
    # Stage: Agent 1 — Regulatory Analysis
    # ------------------------------------------------------------------

    def run_regulatory_analysis(
        self,
        *,
        parsed_document: Optional[ParsedDocument] = None,
        regulation: str = "DORA",
        tier: str = "Tier-2",
        status: StatusCallback = _noop,
        regulator_selection: Optional[Sequence[str]] = None,
        consulting_selection: Optional[Sequence[str]] = None,
        include_consulting_guidance: bool = True,
        intelligence_package: Optional[RegulatoryIntelligencePackage] = None,
        client_roles: Optional[Sequence[str]] = None,
        client_profile: Optional[Mapping[str, Any]] = None,
    ) -> RegulatoryAnalysis:
        return self.regulatory_analysis_agent.analyze(
            parsed_document=parsed_document,
            regulation=regulation,
            tier=tier,
            status=status,
            regulator_selection=regulator_selection,
            consulting_selection=consulting_selection,
            include_consulting_guidance=include_consulting_guidance,
            intelligence_package=intelligence_package,
            client_roles=client_roles,
            client_profile=dict(client_profile) if client_profile else None,
        )

    # ------------------------------------------------------------------
    # Stage: Regulatory Intelligence Pipeline (Stage 1 + Stage 2)
    # ------------------------------------------------------------------

    @staticmethod
    def gather_regulatory_intelligence(
        regulation: str,
        *,
        regulator_selection: Optional[Sequence[str]] = None,
        consulting_selection: Optional[Sequence[str]] = None,
        include_consulting: bool = True,
        status: StatusCallback = _noop,
    ) -> RegulatoryIntelligencePackage:
        """Run the hierarchical regulator + consulting search.

        Exposed on the orchestrator so the UI can preview sources on Page 1
        without going through Agent 1's full BRD generation pipeline.
        """
        return gather_regulatory_intelligence(
            regulation,
            regulator_selection=regulator_selection,
            consulting_selection=consulting_selection,
            include_consulting=include_consulting,
            status=status,
        )

    # ------------------------------------------------------------------
    # Stage: Agent 2 — BRD + RTM
    # ------------------------------------------------------------------

    def run_brd_rtm(
        self,
        analysis: RegulatoryAnalysis,
        *,
        docx_export_path: Optional[Path] = None,
        tier: Optional[str] = None,
    ) -> dict:
        """Return ``{"brd": BRDArtifact, "rtm": RTMArtifact}``."""
        return self.brd_rtm_agent.build(
            analysis,
            docx_export_path=docx_export_path,
            tier=tier,
        )

    # ------------------------------------------------------------------
    # Stage: Agent 3 — Questionnaire Generation
    # ------------------------------------------------------------------

    def run_questionnaire_from_report(
        self, brd: BRDArtifact, *, regulation: str = "DORA",
        name: Optional[str] = None,
        impact: Optional[Any] = None,
        readiness: Optional[ReadinessAssessment] = None,
        analysis: Optional[RegulatoryAnalysis] = None,
        rtm: Optional[RTMArtifact] = None,
        client_roles: Optional[Sequence[str]] = None,
        client_profile: Optional[Mapping[str, Any]] = None,
    ) -> QuestionnairePackage:
        """Generate the AI-driven questionnaire from a generated BRD.

        Passes the full regulatory-analysis context (obligations), RTM
        control mappings, impact + readiness assessments, selected client
        roles and client profile through to the AI questionnaire agent so
        the questions:

        * ask the most-required info to assess **impact** — grounded in
          the specific affected items from the ImpactAssessment;
        * ask the most-required info to assess **readiness** — targeted
          at the client's weakest dimensions from the ReadinessAssessment;
        * are scoped to the selected institution type(s).

        The shared :class:`GenAIClient` held on the orchestrator is also
        forwarded — when ``None``, the AI agent falls back to
        manual-review placeholders (no hardcoded templates).
        """
        return self.questionnaire_agent.from_report(
            brd, regulation=regulation, name=name,
            impact=impact, readiness=readiness,
            analysis=analysis, rtm=rtm,
            client_roles=client_roles,
            client_profile=dict(client_profile) if client_profile else None,
            client=self.client,
        )

    def run_questionnaire_from_docx(
        self, path: Path, *, regulation: str = "DORA",
        name: Optional[str] = None,
        impact: Optional[Any] = None,
        readiness: Optional[ReadinessAssessment] = None,
        analysis: Optional[RegulatoryAnalysis] = None,
        rtm: Optional[RTMArtifact] = None,
        client_roles: Optional[Sequence[str]] = None,
        client_profile: Optional[Mapping[str, Any]] = None,
    ) -> QuestionnairePackage:
        """Generate the AI-driven questionnaire from an uploaded BRD DOCX.

        See :meth:`run_questionnaire_from_report` for how the ``impact``
        and ``readiness`` assessments shape the AI generator's output.
        """
        return self.questionnaire_agent.from_docx(
            path, regulation=regulation, name=name,
            impact=impact, readiness=readiness,
            analysis=analysis, rtm=rtm,
            client_roles=client_roles,
            client_profile=dict(client_profile) if client_profile else None,
            client=self.client,
        )

    def load_questionnaire_package(
        self, package: Mapping[str, Any], *, source: str = "uploaded_json",
        name: Optional[str] = None,
        analysis: Optional[RegulatoryAnalysis] = None,
        client_roles: Optional[Sequence[str]] = None,
    ) -> QuestionnairePackage:
        return self.questionnaire_agent.from_package(
            package, source=source, name=name,
            analysis=analysis, client_roles=client_roles,
        )

    # ------------------------------------------------------------------
    # Stage: Python Rules Engine
    # ------------------------------------------------------------------

    @staticmethod
    def run_rules_engine(
        questionnaire: QuestionnairePackage,
        state: AssessmentState,
    ) -> ScoringResult:
        """Deterministic readiness/impact scoring with top-gap precomputation."""
        package = questionnaire.package
        base_questions = list(package.get("questions") or [])
        active = applicable_base_questions(state, base_questions) + list(state.dynamic_queue)
        evaluation = evaluate(active, state)

        req_scores = evaluation.get("requirement_scores") or {}
        top_gaps = [
            {"requirement_id": rid, "compliance_pct": round(score, 1)}
            for rid, score in sorted(req_scores.items(), key=lambda kv: kv[1])[:10]
        ]
        return ScoringResult(evaluation=evaluation, top_gaps=top_gaps)

    # ------------------------------------------------------------------
    # Stage: Agent 4 — Recommendations
    # ------------------------------------------------------------------

    def run_recommendations(
        self,
        questionnaire: QuestionnairePackage,
        scoring: ScoringResult,
        *,
        min_severity: str = "Watch",
        top_n_requirements: int = 10,
        enrich_with_genai: bool = False,
        branch_log: Optional[Any] = None,
        analysis: Optional[RegulatoryAnalysis] = None,
        client_roles: Optional[Sequence[str]] = None,
    ) -> RecommendationResult:
        return self.recommendation_agent.recommend(
            questionnaire,
            scoring,
            min_severity=min_severity,
            top_n_requirements=top_n_requirements,
            enrich_with_genai=enrich_with_genai,
            branch_log=branch_log,
            analysis=analysis,
            client_roles=client_roles,
        )

    # ------------------------------------------------------------------
    # Stage: AI Assessment Intelligence
    # ------------------------------------------------------------------

    def assess_confidence_intelligence(
        self,
        analysis: Optional[RegulatoryAnalysis],
        *,
        scoring_evaluation: Optional[Any] = None,
        questionnaire_package: Optional[Any] = None,
    ) -> Any:
        from services.ai_assessment_intelligence import assess_confidence

        return assess_confidence(
            analysis,
            scoring_evaluation=scoring_evaluation,
            questionnaire_package=questionnaire_package,
            client=self.client,
        )

    def assess_impact_intelligence(
        self,
        analysis: Optional[RegulatoryAnalysis],
    ) -> Any:
        from services.ai_assessment_intelligence import assess_impact

        return assess_impact(analysis, client=self.client)

    def assess_readiness_intelligence(
        self,
        scoring_evaluation: Any,
        *,
        analysis: Optional[RegulatoryAnalysis] = None,
        questionnaire_package: Optional[Any] = None,
        responses: Optional[Any] = None,
    ) -> Any:
        from services.ai_assessment_intelligence import assess_readiness

        return assess_readiness(
            scoring_evaluation,
            analysis=analysis,
            questionnaire_package=questionnaire_package,
            responses=responses,
            client=self.client,
        )

    # ------------------------------------------------------------------
    # Convenience: Agents 1 -> 3 in one call (smoke / batch path)
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        *,
        parsed_document: Optional[ParsedDocument] = None,
        regulation: str = "DORA",
        tier: str = "Tier-2",
        docx_export_path: Optional[Path] = None,
        status: StatusCallback = _noop,
        regulator_selection: Optional[Sequence[str]] = None,
        consulting_selection: Optional[Sequence[str]] = None,
        include_consulting_guidance: bool = True,
        intelligence_package: Optional[RegulatoryIntelligencePackage] = None,
        client_roles: Optional[Sequence[str]] = None,
        client_profile: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Run Agents 1, 2 and 3 in sequence.

        Returns a dict containing ``analysis``, ``brd``, ``rtm`` and
        ``questionnaire`` keys. User responses, scoring and recommendations
        are handled separately because they need user interaction.
        """
        analysis = self.run_regulatory_analysis(
            parsed_document=parsed_document,
            regulation=regulation,
            tier=tier,
            status=status,
            regulator_selection=regulator_selection,
            consulting_selection=consulting_selection,
            include_consulting_guidance=include_consulting_guidance,
            intelligence_package=intelligence_package,
            client_roles=client_roles,
            client_profile=client_profile,
        )
        bundle = self.run_brd_rtm(
            analysis,
            docx_export_path=docx_export_path,
            tier=tier,
        )
        questionnaire = self.run_questionnaire_from_report(
            bundle["brd"], regulation=regulation,
            analysis=analysis,
            rtm=bundle.get("rtm"),
            client_roles=client_roles,
            client_profile=client_profile,
        )
        return {
            "analysis": analysis,
            "brd": bundle["brd"],
            "rtm": bundle["rtm"],
            "questionnaire": questionnaire,
        }


__all__ = [
    "RegulatoryWorkflowOrchestrator",
    "StatusCallback",
]
