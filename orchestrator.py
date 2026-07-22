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

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@contextmanager
def _stage_timer(stage: str, **context: Any) -> Iterator[Callable[[], float]]:
    """Log start/finish/elapsed for a pipeline stage.

    Yields a callable that returns the elapsed seconds so far, which the
    caller can use to include timing in its own completion log line.
    """
    ctx = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
    logger.info("[TIMER] %s START %s", stage, ctx)
    t0 = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - t0
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.info("[TIMER] %s FAILED elapsed=%.3fs %s", stage, elapsed, ctx)
        raise
    else:
        elapsed = time.perf_counter() - t0
        logger.info("[TIMER] %s DONE elapsed=%.3fs %s", stage, elapsed, ctx)

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
        logger.info(
            "Orchestrator initialised. genai_client=%s",
            "configured" if client is not None else "offline",
        )

    # ------------------------------------------------------------------
    # Stage: Document Parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_document(path: Path, *, kind: str = "regulation") -> ParsedDocument:
        """Read a PDF/DOCX from disk into a :class:`ParsedDocument`."""
        logger.info("Parsing document. path=%s kind=%s", path, kind)
        with _stage_timer("DocumentParser", path=path, kind=kind):
            try:
                parsed = document_parser.parse_document(path, kind=kind)
            except Exception:
                logger.exception("Document parse failed. path=%s kind=%s", path, kind)
                raise
        logger.info(
            "Document parsed. path=%s pages=%s chars=%s",
            path,
            getattr(parsed, "page_count", None),
            len(getattr(parsed, "text", "") or ""),
        )
        return parsed

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
        logger.info(
            "Agent 1 (Regulatory Analysis) invoked. regulation=%s tier=%s regulators=%s client_roles=%s parsed_doc=%s",
            regulation, tier, list(regulator_selection or []) or None,
            list(client_roles or []) or None,
            "yes" if parsed_document is not None else "no",
        )
        with _stage_timer("Agent1_RegulatoryAnalysis", regulation=regulation, tier=tier) as elapsed:
            try:
                result = self.regulatory_analysis_agent.analyze(
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
            except Exception:
                logger.exception("Agent 1 (Regulatory Analysis) crashed. regulation=%s", regulation)
                raise
            logger.info(
                "Agent 1 completed. obligations=%d requirements=%d elapsed=%.3fs",
                len(getattr(result, "obligations", []) or []),
                len(getattr(result, "requirements", []) or []),
                elapsed(),
            )
        return result

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
        exhaustive: bool = False,
        status: StatusCallback = _noop,
    ) -> RegulatoryIntelligencePackage:
        """Run the hierarchical regulator + consulting search.

        Exposed on the orchestrator so the UI can preview sources on Page 1
        without going through Agent 1's full BRD generation pipeline.

        ``exhaustive=True`` forwards to Stage 1 to enable the wide
        multi-variant sweep used after a regulation document has been
        uploaded.
        """
        return gather_regulatory_intelligence(
            regulation,
            regulator_selection=regulator_selection,
            consulting_selection=consulting_selection,
            include_consulting=include_consulting,
            exhaustive=exhaustive,
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
        logger.info("Agent 2 (BRD + RTM) invoked. tier=%s docx=%s", tier, docx_export_path)
        with _stage_timer("Agent2_BRD_RTM", tier=tier) as elapsed:
            try:
                bundle = self.brd_rtm_agent.build(
                    analysis,
                    docx_export_path=docx_export_path,
                    tier=tier,
                )
            except Exception:
                logger.exception("Agent 2 (BRD + RTM) crashed. tier=%s", tier)
                raise
            brd = bundle.get("brd") if isinstance(bundle, dict) else None
            rtm = bundle.get("rtm") if isinstance(bundle, dict) else None
            logger.info(
                "Agent 2 completed. brd_requirements=%d rtm_rows=%d elapsed=%.3fs",
                len(getattr(brd, "requirements", []) or []) if brd else 0,
                len(getattr(rtm, "entries", []) or []) if rtm else 0,
                elapsed(),
            )
        return bundle

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
        logger.info(
            "Agent 3 (Questionnaire) invoked from BRD report. regulation=%s client_roles=%s",
            regulation, list(client_roles or []) or None,
        )
        with _stage_timer("Agent3_Questionnaire_fromReport", regulation=regulation) as elapsed:
            try:
                pkg = self.questionnaire_agent.from_report(
                    brd, regulation=regulation, name=name,
                    impact=impact, readiness=readiness,
                    analysis=analysis, rtm=rtm,
                    client_roles=client_roles,
                    client_profile=dict(client_profile) if client_profile else None,
                    client=self.client,
                )
            except Exception:
                logger.exception("Agent 3 (from report) crashed. regulation=%s", regulation)
                raise
            logger.info(
                "Agent 3 completed. questions=%d elapsed=%.3fs",
                len((pkg.package or {}).get("questions") or []),
                elapsed(),
            )
        return pkg

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
        logger.info(
            "Agent 3 (Questionnaire) invoked from uploaded DOCX. path=%s regulation=%s",
            path, regulation,
        )
        with _stage_timer("Agent3_Questionnaire_fromDOCX", path=path, regulation=regulation) as elapsed:
            try:
                pkg = self.questionnaire_agent.from_docx(
                    path, regulation=regulation, name=name,
                    impact=impact, readiness=readiness,
                    analysis=analysis, rtm=rtm,
                    client_roles=client_roles,
                    client_profile=dict(client_profile) if client_profile else None,
                    client=self.client,
                )
            except Exception:
                logger.exception("Agent 3 (from DOCX) crashed. path=%s regulation=%s", path, regulation)
                raise
            logger.info(
                "Agent 3 (from DOCX) completed. questions=%d elapsed=%.3fs",
                len((pkg.package or {}).get("questions") or []),
                elapsed(),
            )
        return pkg

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
        logger.debug(
            "Rules engine invoked. base_questions=%d active=%d responses=%d dynamic_queue=%d",
            len(base_questions), len(active), len(state.responses or {}), len(state.dynamic_queue or []),
        )
        with _stage_timer("RulesEngine", active_questions=len(active)) as elapsed:
            try:
                evaluation = evaluate(active, state)
            except Exception:
                logger.exception("Rules engine evaluate() crashed. active_questions=%d", len(active))
                raise

            req_scores = evaluation.get("requirement_scores") or {}
            top_gaps = [
                {"requirement_id": rid, "compliance_pct": round(score, 1)}
                for rid, score in sorted(req_scores.items(), key=lambda kv: kv[1])[:10]
            ]
            logger.info(
                "Rules engine done. compliance=%.1f%% top_gap=%s elapsed=%.3fs",
                float(evaluation.get("compliance_score_pct") or 0.0),
                top_gaps[0]["requirement_id"] if top_gaps else None,
                elapsed(),
            )
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
        weighted_impact: Optional[Any] = None,
    ) -> RecommendationResult:
        logger.info(
            "Agent 4 (Recommendations) invoked. min_severity=%s top_n=%d enrich=%s",
            min_severity, top_n_requirements, enrich_with_genai,
        )
        with _stage_timer(
            "Agent4_Recommendations",
            min_severity=min_severity, top_n=top_n_requirements, enrich=enrich_with_genai,
        ) as elapsed:
            try:
                result = self.recommendation_agent.recommend(
                    questionnaire,
                    scoring,
                    min_severity=min_severity,
                    top_n_requirements=top_n_requirements,
                    enrich_with_genai=enrich_with_genai,
                    branch_log=branch_log,
                    analysis=analysis,
                    client_roles=client_roles,
                    weighted_impact=weighted_impact,
                )
            except Exception:
                logger.exception("Agent 4 (Recommendations) crashed.")
                raise
            logger.info(
                "Agent 4 completed. recommendations=%d elapsed=%.3fs",
                len(getattr(result, "recommendations", []) or []),
                elapsed(),
            )
        return result

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
        with _stage_timer("FullPipeline_Agents1to3", regulation=regulation, tier=tier):
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
