"""Agent 4 — Recommendations.

Role in the pipeline
--------------------
Takes the :class:`~models.workflow_models.ScoringResult` produced by the
Python Rules Engine (plus the questionnaire package, so recommendations can
quote requirement IDs and titles) and returns a
:class:`~models.workflow_models.RecommendationResult`.

Recommendations are tied to:

* gaps (lowest-scoring requirements / area-function pairs)
* obligations (via mapped requirement IDs)
* impacted functions (used to pick a suggested owner)
* remediation priority (severity-driven horizon and action wording)

Implementation strategy
-----------------------
* Legacy compact recommendations come from
  :func:`services.recommendation_service.generate_recommendations` — the
  same engine the existing dashboard uses, so backwards compatibility is
  preserved.
* Consulting-grade rich recommendations (What / Why / How / Priority /
  Expected outcome / Dependencies) come from
  :func:`services.rich_recommendation_service.build_rich_recommendations`.
  These are the primary artefacts surfaced on the dashboard.
* Both paths gracefully fall back to deterministic composition when the
  GenAI Shared Service is unavailable.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _legacy_enrich_enabled() -> bool:
    """Return True when the legacy compact-recommendation LLM rewrite is on.

    The dashboard, exports and every UI surface prefer the rich
    consulting-grade recommendations produced by
    :func:`services.rich_recommendation_service.build_rich_recommendations`.
    The legacy per-rec rewrite is therefore duplicated LLM work that the
    UI never displays. Historically it cost ~120s per Agent 4 run because
    it looped sequentially over ~22 recommendations. It is now OFF by
    default and can be re-enabled by setting
    ``LEGACY_RECOMMENDATION_ENRICH_ENABLED=1`` in the environment for
    users who consume the legacy JSON/Excel exports directly.
    """
    raw = str(os.getenv("LEGACY_RECOMMENDATION_ENRICH_ENABLED", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}

from models.workflow_models import (
    QuestionnairePackage,
    RecommendationResult,
    RegulatoryAnalysis,
    ScoringResult,
)
from services.client_roles import normalize_client_roles
from services.genai_service import GenAIClient
from services.recommendation_evaluator import attach_evaluations
from services.recommendation_service import (
    enrich_recommendations_with_genai,
    generate_recommendations,
)
from services.rich_recommendation_service import build_rich_recommendations


class RecommendationAgent:
    """Agent 4 implementation."""

    def __init__(self, *, client: Optional[GenAIClient] = None) -> None:
        self.client = client

    def recommend(
        self,
        questionnaire: QuestionnairePackage,
        scoring: ScoringResult,
        *,
        min_severity: str = "Watch",
        top_n_requirements: int = 10,
        enrich_with_genai: bool = False,
        branch_log: Optional[Sequence[Mapping[str, Any]]] = None,
        analysis: Optional[RegulatoryAnalysis] = None,
        client_roles: Optional[Sequence[str]] = None,
        weighted_impact: Optional[Any] = None,
    ) -> RecommendationResult:
        """Produce a structured recommendation bundle.

        Parameters
        ----------
        questionnaire
            The :class:`QuestionnairePackage` produced by Agent 3. The
            underlying dict is what the deterministic engine reads.
        scoring
            The :class:`ScoringResult` produced by the Python Rules Engine.
        min_severity
            Lowest severity to include (``Critical``, ``At risk``, ``Watch``,
            ``Ready``).
        top_n_requirements
            Cap on requirement-level recommendations appended to the bundle.
        enrich_with_genai
            When ``True`` and a configured client is available, rewrite the
            ``suggested_action`` field via the PwC GenAI Shared Service, and
            also GenAI-enrich the rich recommendations.
            Fails soft.
        analysis
            Optional :class:`RegulatoryAnalysis` from Agent 1. When
            supplied, rich recommendations quote real obligation IDs.
        client_roles
            Institution types the recommendations should be scoped to.
            When supplied (or when ``analysis.client_roles`` is populated)
            only obligations Applicable / Partially Applicable / Uncertain
            for the selected roles feed into the recommendation generator,
            keeping the recommendations tightly aligned with the client
            role–aware interpretation produced by Agent 1.
        """
        roles = normalize_client_roles(client_roles) if client_roles else []
        if not roles and analysis is not None:
            roles = normalize_client_roles(getattr(analysis, "client_roles", None) or [])
        logger.info(
            "Agent 4 recommend() start. min_severity=%s top_n=%d enrich=%s roles=%s",
            min_severity, top_n_requirements, enrich_with_genai, roles or None,
        )

        # Scope the analysis to in-scope obligations without mutating the
        # caller's copy (persistence + UI still keep the full obligation
        # list). ``obligations_for_roles`` returns the unfiltered list when
        # ``roles`` is empty so this is a no-op in the generic case.
        scoped_analysis = analysis
        if analysis is not None and roles:
            in_scope = analysis.obligations_for_roles(roles)
            if in_scope and len(in_scope) != len(analysis.obligations):
                scoped_analysis = copy.copy(analysis)
                scoped_analysis.obligations = list(in_scope)

        recs = generate_recommendations(
            questionnaire.package,
            scoring.evaluation,
            min_severity=min_severity,
            top_n_requirements=top_n_requirements,
            branch_log=branch_log,
        )
        if roles:
            recs = _annotate_recommendations_with_roles(recs, roles)

        used_genai = False
        if enrich_with_genai and self.client is not None and _legacy_enrich_enabled():
            try:
                recs = enrich_recommendations_with_genai(
                    recs, questionnaire.package, client=self.client,
                )
                used_genai = True
            except Exception:
                logger.exception(
                    "Agent 4: GenAI enrichment of recommendations FAILED (using deterministic drafts).",
                )
                used_genai = False
        elif enrich_with_genai and self.client is not None:
            # Legacy per-rec rewrite is intentionally skipped by default -
            # the dashboard displays rich_recommendations, so this loop
            # is duplicated LLM work. Set
            # LEGACY_RECOMMENDATION_ENRICH_ENABLED=1 to re-enable.
            logger.info(
                "Agent 4: skipping legacy compact-rec GenAI rewrite "
                "(LEGACY_RECOMMENDATION_ENRICH_ENABLED not set); rich "
                "recommendations will still be LLM-refined.",
            )
            used_genai = True  # rich enrichment below still uses GenAI

        rich = build_rich_recommendations(
            analysis=scoped_analysis,
            scoring_evaluation=scoring.evaluation,
            top_gaps=scoring.top_gaps,
            package=questionnaire.package,
            impact=scoring.impact,
            readiness=scoring.readiness,
            weighted_impact=weighted_impact,
            client=self.client if enrich_with_genai else None,
            enrich_with_genai=bool(enrich_with_genai and self.client is not None),
        )
        if roles:
            rich = _annotate_rich_recommendations_with_roles(rich, roles)

        # Mentor #6 — deterministic recommendation-quality evaluation loop.
        # Score every recommendation across (coverage, specificity,
        # actionability, grounding), attach the result to
        # ``metadata["_eval"]`` so the Dashboard, Gap page and exports can
        # surface it. Runs on every session because it is cheap (token
        # Jaccard + heuristics) and offline.
        regulation = str(getattr(scoped_analysis, "regulation", "") or "")
        obligations_for_eval = list(getattr(scoped_analysis, "obligations", []) or [])
        try:
            attach_evaluations(
                rich,
                obligations=obligations_for_eval,
                scoring_evaluation=scoring.evaluation,
                regulation=regulation,
            )
            attach_evaluations(
                recs,
                obligations=obligations_for_eval,
                scoring_evaluation=scoring.evaluation,
                regulation=regulation,
            )
        except Exception:  # pragma: no cover - never block recs on eval bug
            logger.exception(
                "Agent 4: recommendation evaluation loop FAILED (non-fatal).",
            )

        return RecommendationResult(
            recommendations=list(recs),
            severity_filter=min_severity,
            top_n_requirements=top_n_requirements,
            used_genai=used_genai,
            rich_recommendations=rich,
        )


def _annotate_recommendations_with_roles(recs: Sequence[Any], roles: Sequence[str]) -> List[Any]:
    """Attach ``client_roles`` metadata to legacy recommendation dataclasses.

    Best-effort: the legacy :class:`~services.recommendation_service.Recommendation`
    is a dataclass without a slot for institution types, so we mutate the
    ``suggested_action`` header to include the role scope, which flows into
    JSON / Excel exports without a schema change. Callers that want the raw
    role list can consume ``rich_recommendations`` instead.
    """
    if not roles:
        return list(recs)
    scope_label = f"[Scope: {', '.join(roles)}] "
    out: List[Any] = []
    for rec in recs:
        try:
            current_action = getattr(rec, "suggested_action", "") or ""
            if not current_action.startswith("[Scope:"):
                rec.suggested_action = scope_label + current_action
        except Exception:
            pass
        out.append(rec)
    return out


def _annotate_rich_recommendations_with_roles(
    rich: Sequence[Any], roles: Sequence[str],
) -> List[Any]:
    """Attach role scope to rich recommendations.

    Mutates the recommendation's ``why`` / ``regulatory_rationale`` text so
    the exported deliverable clearly states the applicable institution
    types. Falls back gracefully for dict-shaped recommendations.
    """
    if not roles:
        return list(rich)
    role_label = ", ".join(roles)
    out: List[Any] = []
    for rec in rich:
        try:
            existing_why = getattr(rec, "why", "") or ""
            if role_label not in existing_why:
                rec.why = (
                    f"{existing_why}\n\nApplicability: this recommendation "
                    f"addresses obligations that are applicable to {role_label}."
                ).strip()
        except Exception:
            pass
        out.append(rec)
    return out


__all__ = ["RecommendationAgent"]
