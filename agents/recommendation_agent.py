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
* Deterministic path delegates to
  :func:`services.recommendation_service.generate_recommendations` — the
  same engine the existing dashboard uses, so behaviour is preserved.
* Optional GenAI enrichment is delegated to
  :func:`services.recommendation_service.enrich_recommendations_with_genai`
  and falls back silently when the client is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from models.workflow_models import (
    QuestionnairePackage,
    RecommendationResult,
    ScoringResult,
)
from services.genai_service import GenAIClient
from services.recommendation_service import (
    enrich_recommendations_with_genai,
    generate_recommendations,
)


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
            ``suggested_action`` field via the PwC GenAI Shared Service.
            Fails soft.
        """
        recs = generate_recommendations(
            questionnaire.package,
            scoring.evaluation,
            min_severity=min_severity,
            top_n_requirements=top_n_requirements,
            branch_log=branch_log,
        )
        used_genai = False
        if enrich_with_genai and self.client is not None:
            try:
                recs = enrich_recommendations_with_genai(
                    recs, questionnaire.package, client=self.client,
                )
                used_genai = True
            except Exception:
                used_genai = False

        return RecommendationResult(
            recommendations=list(recs),
            severity_filter=min_severity,
            top_n_requirements=top_n_requirements,
            used_genai=used_genai,
        )


__all__ = ["RecommendationAgent"]
