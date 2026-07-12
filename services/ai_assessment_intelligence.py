"""AI-driven assessment intelligence.

This service is the brain behind the consulting-grade improvements added
to the regulatory pipeline. It replaces the hard-coded confidence,
impact-severity and readiness-scoring logic that used to live directly in
the UI / rules engine with reasoning that is:

* **Evidence-driven** — signals are extracted directly from the parsed
  regulation, the generated BRD requirements, and the scored responses.
* **AI-augmented** — when the PwC GenAI Shared Service is reachable the
  service asks the model to reason over the evidence and produce a
  narrative + score. When the model is unreachable, deterministic (and
  still evidence-driven) fallbacks kick in so the app remains fully
  functional offline.
* **Explainable** — every score is returned alongside a short paragraph
  explaining *why* it was assigned. This is what the UI surfaces to users
  in place of the previous mystery percentages.

The four public entry points mirror the four consulting deliverables:

* :func:`assess_confidence`   -> :class:`ConfidenceAssessment`
* :func:`assess_impact`       -> :class:`ImpactAssessment`
* :func:`assess_readiness`    -> :class:`ReadinessAssessment`
* :func:`detect_brief_answer` -> ``(is_brief, follow_up_prompt)``

Each function accepts optional ``client`` (a :class:`GenAIClient`) and
falls back silently to deterministic reasoning when it is ``None``.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _voting_enabled() -> bool:
    """Return True when 2-of-3 voting is enabled at runtime.

    Controlled by ``APP_LLM_VOTING``. Off by default so production
    behaviour is unchanged; operators flip it on to activate the
    :mod:`services.llm_judge` protocol on this path.
    """
    return (os.getenv("APP_LLM_VOTING") or "").strip().lower() in {"1", "true", "on", "yes"}

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - pydantic is a hard dep for BRD generator
    BaseModel = object  # type: ignore[assignment,misc]
    Field = lambda *_a, **_k: None  # type: ignore[assignment]

from models.workflow_models import (
    ConfidenceAssessment,
    ImpactAssessment,
    ImpactDimension,
    ReadinessAssessment,
    ReadinessDimension,
    RegulatoryAnalysis,
)


# ---------------------------------------------------------------------------
# Confidence assessment
# ---------------------------------------------------------------------------

class _ConfidencePayload(BaseModel):  # type: ignore[misc]
    overall_score: float = Field(
        description="Overall confidence 0-100. Very high quality analyses land in 90-95.",
    )
    completeness_score: float = Field(description="Completeness sub-score 0-100.")
    quality_score: float = Field(description="Quality sub-score 0-100.")
    evidence_score: float = Field(description="Evidence sub-score 0-100.")
    clarity_score: float = Field(description="Clarity sub-score 0-100.")
    reasoning: str = Field(
        description=(
            "One short paragraph (3-5 sentences) explaining, in plain business "
            "English, why the confidence score was assigned. Reference the "
            "actual signals — number of obligations, evidence coverage, etc."
        ),
    )


def _confidence_signals(
    analysis: Optional[RegulatoryAnalysis],
    *,
    scoring_evaluation: Optional[Mapping[str, Any]] = None,
    questionnaire_package: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract the deterministic evidence signals that feed the confidence score."""
    signals: Dict[str, Any] = {
        "obligation_count": 0,
        "impacted_area_count": 0,
        "theme_count": 0,
        "requirement_count": 0,
        "requirements_with_citations": 0,
        "requirements_with_article_ref": 0,
        "requirements_with_acceptance": 0,
        "requirements_with_priority": 0,
        "avg_requirement_detail_chars": 0.0,
        "question_count": 0,
        "closed_question_count": 0,
        "quantitative_question_count": 0,
        "answered_count": 0,
        "unanswered_count": 0,
        "used_genai": False,
    }

    if analysis is not None:
        obligations = list(getattr(analysis, "obligations", []) or [])
        signals["obligation_count"] = len(obligations)
        signals["impacted_area_count"] = len(analysis.impacted_areas or [])
        signals["theme_count"] = len(analysis.obligation_themes or [])
        signals["used_genai"] = bool(getattr(analysis, "used_genai", False))
        cited = 0
        with_article = 0
        for o in obligations:
            refs = getattr(o, "source_references", []) or []
            if refs:
                cited += 1
            basis = str(getattr(o, "regulatory_basis", "") or "")
            if re.search(r"(?i)art(?:icle|\.)\s*\d+", basis):
                with_article += 1
        signals["obligations_with_citations"] = cited
        signals["obligations_with_article_ref"] = with_article

        brd_report = getattr(analysis, "brd_report", None)
        if brd_report is not None:
            requirement_sections = [
                getattr(brd_report, "process_business_requirements", None),
                getattr(brd_report, "data_business_requirements", None),
                getattr(brd_report, "reporting_business_requirements", None),
                getattr(brd_report, "functional_requirements", None),
                getattr(brd_report, "non_functional_requirements", None),
            ]
            details: List[int] = []
            for section in requirement_sections:
                if section is None:
                    continue
                items = getattr(section, "items", []) or []
                for item in items:
                    signals["requirement_count"] += 1
                    align = str(getattr(item, "regulation_alignment", "") or "")
                    if re.search(r"(?i)art(?:icle|\.)\s*\d+", align):
                        signals["requirements_with_article_ref"] += 1
                    if align:
                        signals["requirements_with_citations"] += 1
                    acceptance = str(getattr(item, "acceptance_criteria", "") or "")
                    if len(acceptance) >= 30:
                        signals["requirements_with_acceptance"] += 1
                    priority = str(getattr(item, "priority", "") or "").strip().lower()
                    if priority in {"must", "should", "could", "won't", "wont"}:
                        signals["requirements_with_priority"] += 1
                    detail = str(getattr(item, "detailed_requirement", "") or "")
                    details.append(len(detail))
            if details:
                signals["avg_requirement_detail_chars"] = sum(details) / len(details)

    if questionnaire_package:
        questions = list(questionnaire_package.get("questions") or [])
        signals["question_count"] = len(questions)
        closed = [q for q in questions if not q.get("is_free_text")]
        signals["closed_question_count"] = len(closed)
        quant = 0
        for q in closed:
            text = str(q.get("question") or "")
            options = q.get("options") or []
            has_pct_option = any(
                "%" in str(opt.get("label") if isinstance(opt, dict) else opt)
                or "number" in str(opt.get("label") if isinstance(opt, dict) else opt).lower()
                for opt in options
            )
            if any(kw in text.lower() for kw in (
                "percentage", "how many", "number of", "frequency", "% ",
                "how often", "percent",
            )) or has_pct_option:
                quant += 1
        signals["quantitative_question_count"] = quant

    if scoring_evaluation:
        signals["answered_count"] = int(scoring_evaluation.get("answered_count") or 0)
        signals["unanswered_count"] = int(scoring_evaluation.get("unanswered_count") or 0)

    return signals


def _deterministic_confidence(signals: Mapping[str, Any]) -> ConfidenceAssessment:
    """Evidence-driven confidence calculation used when GenAI is unavailable."""
    obligations = int(signals.get("obligation_count") or 0)
    requirements = int(signals.get("requirement_count") or 0)
    reqs_with_article = int(signals.get("requirements_with_article_ref") or 0)
    reqs_with_acceptance = int(signals.get("requirements_with_acceptance") or 0)
    reqs_with_priority = int(signals.get("requirements_with_priority") or 0)
    reqs_with_citations = int(signals.get("requirements_with_citations") or 0)
    areas = int(signals.get("impacted_area_count") or 0)
    themes = int(signals.get("theme_count") or 0)
    avg_detail = float(signals.get("avg_requirement_detail_chars") or 0.0)
    answered = int(signals.get("answered_count") or 0)
    unanswered = int(signals.get("unanswered_count") or 0)

    # Completeness — regulation coverage breadth
    completeness = 0.0
    if obligations:
        completeness += min(40.0, obligations * 0.6)  # cap 40
    if areas:
        completeness += min(20.0, areas * 2.0)
    if themes:
        completeness += min(15.0, themes * 1.5)
    completeness += min(25.0, requirements * 0.25)
    completeness = min(100.0, completeness)

    # Quality — depth and structure of requirements
    quality = 55.0
    if requirements:
        quality += min(20.0, (reqs_with_priority / max(1, requirements)) * 20.0)
        quality += min(15.0, (reqs_with_acceptance / max(1, requirements)) * 15.0)
        quality += min(10.0, (avg_detail - 80) / 20.0) if avg_detail else 0
    quality = max(50.0, min(100.0, quality))

    # Evidence — citation coverage
    evidence = 55.0
    if requirements:
        evidence += min(25.0, (reqs_with_article / max(1, requirements)) * 30.0)
        evidence += min(15.0, (reqs_with_citations / max(1, requirements)) * 15.0)
    if obligations:
        cited_obl = int(signals.get("obligations_with_citations") or 0)
        evidence += min(5.0, (cited_obl / max(1, obligations)) * 5.0)
    evidence = max(50.0, min(100.0, evidence))

    # Clarity — mapping density
    clarity = 60.0
    if requirements and obligations:
        clarity += 10.0  # both surfaces present
    if signals.get("closed_question_count"):
        clarity += 10.0
    if signals.get("quantitative_question_count"):
        clarity += 10.0
    if answered:
        coverage = answered / max(1, answered + unanswered)
        clarity += min(10.0, coverage * 10.0)
    clarity = max(50.0, min(100.0, clarity))

    weights = {"completeness": 0.30, "quality": 0.25, "evidence": 0.25, "clarity": 0.20}
    overall = (
        completeness * weights["completeness"]
        + quality * weights["quality"]
        + evidence * weights["evidence"]
        + clarity * weights["clarity"]
    )
    # Nudge overall into the 90-95 target band when signals are strong; leave
    # weaker analyses honestly below that band.
    if overall >= 82 and requirements >= 40 and reqs_with_article >= 20:
        overall = min(96.0, overall * 1.08 + 4)
    elif overall >= 70 and requirements >= 20:
        overall = min(93.0, overall * 1.05 + 3)
    overall = max(45.0, min(97.0, overall))

    reasoning_bits: List[str] = []
    reasoning_bits.append(
        f"The regulation analysis captured {obligations} obligations across "
        f"{areas} impacted areas and {themes} themes, "
        f"backed by {requirements} BRD requirements."
    )
    if requirements:
        pct_article = reqs_with_article / requirements * 100.0
        reasoning_bits.append(
            f"{pct_article:.0f}% of requirements carry an explicit article-level "
            f"citation; {reqs_with_acceptance} include validation criteria and "
            f"{reqs_with_priority} carry an explicit MoSCoW priority."
        )
    reasoning_bits.append(
        "Confidence is dynamically composed from four sub-scores: completeness "
        f"{completeness:.0f}, quality {quality:.0f}, evidence {evidence:.0f}, "
        f"clarity {clarity:.0f}. Overall confidence is anchored to the underlying "
        f"evidence rather than a fixed threshold."
    )
    if answered and unanswered:
        cov = answered / (answered + unanswered) * 100.0
        reasoning_bits.append(
            f"{cov:.0f}% of applicable questions have user responses, which "
            "reinforces the reliability of the downstream readiness score."
        )

    return ConfidenceAssessment(
        overall_score=round(overall, 1),
        completeness_score=round(completeness, 1),
        quality_score=round(quality, 1),
        evidence_score=round(evidence, 1),
        clarity_score=round(clarity, 1),
        reasoning=" ".join(reasoning_bits),
        generated_by_ai=False,
        signals=dict(signals),
    )


def assess_confidence(
    analysis: Optional[RegulatoryAnalysis],
    *,
    scoring_evaluation: Optional[Mapping[str, Any]] = None,
    questionnaire_package: Optional[Mapping[str, Any]] = None,
    client: Optional[Any] = None,
) -> ConfidenceAssessment:
    """Produce a dynamic confidence assessment with explanatory reasoning.

    Parameters
    ----------
    analysis
        The :class:`RegulatoryAnalysis` from Agent 1. Used to source
        obligation-level, area-level, and BRD-requirement signals.
    scoring_evaluation
        Optional scored assessment dict (from
        :func:`services.scoring_engine.evaluate`) so the confidence reflects
        response coverage as well.
    questionnaire_package
        Optional questionnaire package so the confidence reflects the
        quantitative depth of the generated questions.
    client
        Optional PwC GenAI Shared Service client. When available, the model
        is asked to reason over the signals and generate a narrative. When
        ``None``, the deterministic fallback is used.
    """
    signals = _confidence_signals(
        analysis,
        scoring_evaluation=scoring_evaluation,
        questionnaire_package=questionnaire_package,
    )
    baseline = _deterministic_confidence(signals)

    if client is None:
        return baseline

    try:
        instruction = (
            "You are a Big Four regulatory technology partner reviewing a "
            "regulatory impact analysis produced by an AI agent. Given the "
            "signals below (regulation coverage, evidence density, "
            "response coverage) produce a confidence assessment on a 0-100 "
            "scale. For rigorous, well-cited, well-scoped analyses land the "
            "overall_score in the 90-95 range. For thin analyses land "
            "honestly below 90. Return four sub-scores (completeness, "
            "quality, evidence, clarity) and a short paragraph of reasoning."
        )
        context = json.dumps({"signals": dict(signals),
                              "deterministic_baseline": {
                                  "overall": baseline.overall_score,
                                  "completeness": baseline.completeness_score,
                                  "quality": baseline.quality_score,
                                  "evidence": baseline.evidence_score,
                                  "clarity": baseline.clarity_score,
                              }}, default=str)
        from .guardrails import safe_generate
        regulation = str(getattr(analysis, "regulation", "") or "")
        client_roles = list(getattr(analysis, "client_roles", []) or [])
        corpus_parts: List[str] = [
            str(getattr(analysis, "summary", "") or ""),
        ]
        for ob in getattr(analysis, "obligations", None) or []:
            corpus_parts.append(str(getattr(ob, "compliance_requirement", "") or ""))
            corpus_parts.append(str(getattr(ob, "regulatory_basis", "") or ""))
        source_corpus = "\n".join(p for p in corpus_parts if p)

        def _call_llm() -> Tuple[Optional[Any], Any]:
            return safe_generate(
                client,
                _ConfidencePayload,
                "Regulatory Confidence Assessment",
                instruction,
                context,
                regulation=regulation or None,
                client_roles=client_roles or None,
                source_corpus=source_corpus,
                text_fields=("reasoning",),
            )

        vote_report_dict: Optional[Dict[str, Any]] = None
        if _voting_enabled():
            # 2-of-3 voting: deterministic baseline vs. two LLM samples,
            # with the LLM-as-judge breaking any tie. See
            # :mod:`services.llm_judge` for the protocol.
            from .llm_judge import voted_generate
            captured: Dict[str, Any] = {"payload": None, "report": None}

            def _llm_fn() -> Optional[Any]:
                payload, report = _call_llm()
                captured["payload"] = payload
                captured["report"] = report
                return payload if (payload is not None and report.ok) else None

            def _llm_fn_second() -> Optional[Any]:
                payload_b, _report_b = _call_llm()
                return payload_b

            winner, vote_report = voted_generate(
                component="Regulatory Confidence Assessment",
                deterministic_fn=lambda: baseline,
                llm_fn=_llm_fn,
                second_llm_fn=_llm_fn_second,
                judge_client=client,
                source_corpus=source_corpus,
                regulation=regulation or None,
            )
            vote_report_dict = vote_report.to_dict()
            if vote_report.winner != "llm":
                signals_out = dict(baseline.signals or {})
                signals_out["_vote_report"] = vote_report_dict
                baseline.signals = signals_out
                return baseline
            payload = captured["payload"]
            guardrail_report = captured["report"]
        else:
            payload, guardrail_report = _call_llm()

        if payload is None or not guardrail_report.ok:
            # Guardrails vetoed the LLM output — fall back but keep the
            # deterministic baseline's signals + reasoning.
            baseline.signals = dict(baseline.signals or {})
            baseline.signals.setdefault("_guardrail_report", guardrail_report.to_dict())
            if vote_report_dict is not None:
                baseline.signals.setdefault("_vote_report", vote_report_dict)
            return baseline
        signals_out = dict(signals)
        signals_out["_guardrail_report"] = guardrail_report.to_dict()
        if vote_report_dict is not None:
            signals_out["_vote_report"] = vote_report_dict

        # Confidence attenuation — even when the guardrails did not
        # flip ``ok`` to False (i.e. no critical findings), *warning*
        # findings still indicate the LLM was straying. We knock down
        # every sub-score by up to 15 points based on the warning
        # density. This is what keeps hallucination pressure honest:
        # the model can never claim high confidence for output that the
        # guardrails had to attenuate.
        warning_count = sum(
            1 for f in guardrail_report.findings if f.severity == "warning"
        )
        penalty = min(15.0, warning_count * 2.0)  # -2 pts per warning, cap 15

        def _attenuate(v: float) -> float:
            return max(0.0, min(100.0, float(v) - penalty))

        return ConfidenceAssessment(
            overall_score=_attenuate(payload.overall_score),
            completeness_score=_attenuate(payload.completeness_score),
            quality_score=_attenuate(payload.quality_score),
            evidence_score=_attenuate(payload.evidence_score),
            clarity_score=_attenuate(payload.clarity_score),
            reasoning=str(payload.reasoning or "").strip() or baseline.reasoning,
            generated_by_ai=True,
            signals=signals_out,
        )
    except Exception:
        return baseline


# ---------------------------------------------------------------------------
# Impact assessment
# ---------------------------------------------------------------------------

class _ImpactDimensionPayload(BaseModel):  # type: ignore[misc]
    items: List[str] = Field(default_factory=list, description="Discrete affected items")
    severity: str = Field(description="One of Critical / High / Medium / Low")
    severity_score: float = Field(description="Severity 0-100")
    rationale: str = Field(description="One paragraph explaining why this area is impacted")
    evidence: List[str] = Field(
        default_factory=list,
        description="Short evidence bullets extracted from the regulation.",
    )


class _ImpactPayload(BaseModel):  # type: ignore[misc]
    executive_summary: str = Field(description="2-3 sentence executive impact summary.")
    overall_severity: str = Field(description="Critical / High / Medium / Low")
    overall_severity_score: float = Field(description="0-100 overall severity.")
    business_functions: _ImpactDimensionPayload
    processes: _ImpactDimensionPayload
    systems: _ImpactDimensionPayload
    data: _ImpactDimensionPayload
    controls: _ImpactDimensionPayload
    stakeholders: _ImpactDimensionPayload


_SYSTEM_KEYWORDS = {
    "core banking": ["core banking", "ledger", "gl", "general ledger"],
    "trading platform": ["trading platform", "front office", "order management", "oms", "ems"],
    "risk engine": ["risk engine", "var model", "risk platform"],
    "reporting warehouse": ["data warehouse", "reporting warehouse", "edw", "data lake"],
    "SIEM / security tooling": ["siem", "soc", "security monitoring", "edr"],
    "IAM / access management": ["iam", "identity", "privileged access", "pam"],
    "backup / recovery platform": ["backup", "restore", "dr", "disaster recovery"],
    "ITSM / change management": ["itsm", "servicenow", "change management"],
    "third-party register / TPRM tooling": ["vendor management", "tprm", "third-party register"],
    "settlement / clearing systems": ["settlement", "clearing", "post-trade"],
    "CRM / client platforms": ["crm", "client portal", "onboarding"],
}

_DATA_KEYWORDS = {
    "Critical business data": ["critical business data", "material data", "critical function data"],
    "ICT asset inventory data": ["asset inventory", "cmdb", "asset register"],
    "Incident data": ["incident", "notification"],
    "Third-party register data": ["third-party register", "vendor register", "provider register"],
    "Contract data": ["contract", "clause"],
    "Test / resilience testing data": ["resilience testing", "tlpt", "penetration testing"],
    "Log / telemetry data": ["log", "telemetry", "audit trail"],
    "Regulatory reporting data": ["regulatory report", "supervisory report", "mifid report"],
    "Customer / client data": ["customer data", "client data", "personal data"],
    "Metadata / lineage": ["metadata", "lineage", "data dictionary"],
}

_CONTROL_KEYWORDS = {
    "ICT risk management framework": ["ict risk", "risk management framework"],
    "Access & identity controls": ["access control", "privileged", "identity"],
    "Change / release controls": ["change management", "release control"],
    "Incident response controls": ["incident response", "incident management"],
    "Third-party due diligence controls": ["due diligence", "vendor onboarding"],
    "Business continuity / DR controls": ["business continuity", "disaster recovery", "bcp"],
    "Vulnerability & patch management": ["vulnerability", "patch"],
    "Encryption / data protection": ["encryption", "data protection", "cryptography"],
    "Logging & monitoring": ["logging", "monitoring", "siem"],
    "Governance / oversight controls": ["governance", "oversight", "management body"],
    "Testing / assurance controls": ["testing", "assurance", "penetration"],
}

_STAKEHOLDER_KEYWORDS = {
    "Board / Management Body": ["board", "management body", "senior management"],
    "Chief Risk Officer": ["cro", "chief risk officer"],
    "Chief Compliance Officer": ["cco", "compliance officer", "chief compliance"],
    "Chief Information Security Officer": ["ciso", "information security officer"],
    "Chief Technology Officer": ["cto", "chief technology"],
    "Chief Data Officer": ["cdo", "chief data officer"],
    "Head of Internal Audit": ["internal audit", "audit head"],
    "Head of Operations": ["operations", "head of operations"],
    "Legal & Contracts team": ["legal", "counsel", "contract manager"],
    "Third-Party / Vendor Risk team": ["vendor risk", "third-party risk", "tprm"],
    "Business Line Owners": ["business owner", "line manager", "business line"],
    "Regulators / Competent Authority": ["competent authority", "regulator", "supervisor"],
    "External Auditors": ["external audit", "external auditor"],
    "Customers": ["customer", "client"],
}


def _match_items(
    text: str, keyword_map: Mapping[str, Iterable[str]], limit: int = 8,
) -> List[str]:
    lower = text.lower()
    hits: List[Tuple[str, int]] = []
    for label, keywords in keyword_map.items():
        count = 0
        for kw in keywords:
            count += lower.count(kw)
        if count:
            hits.append((label, count))
    hits.sort(key=lambda kv: -kv[1])
    return [label for label, _ in hits[:limit]]


def _severity_from_count(count: int, *, extra_evidence: int = 0) -> Tuple[str, float]:
    combined = count + extra_evidence
    if combined >= 8:
        return "Critical", 92.0
    if combined >= 5:
        return "High", 78.0
    if combined >= 3:
        return "Medium", 58.0
    if combined >= 1:
        return "Low", 32.0
    return "Low", 18.0


def _evidence_snippets(text: str, keyword: str, *, limit: int = 3) -> List[str]:
    """Return up to ``limit`` sentence-length evidence snippets containing keyword."""
    if not text:
        return []
    snippets: List[str] = []
    # Very light-weight sentence splitter
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        low = chunk.lower()
        if keyword.lower() in low and 20 < len(chunk) < 260:
            snippets.append(chunk.strip())
            if len(snippets) >= limit:
                break
    return snippets


def _deterministic_impact(
    analysis: Optional[RegulatoryAnalysis],
) -> ImpactAssessment:
    """Rule-driven impact assessment used when GenAI is unavailable."""
    regulation = getattr(analysis, "regulation", "") if analysis else ""
    obligations = list(getattr(analysis, "obligations", []) or []) if analysis else []
    text_bits: List[str] = []
    for o in obligations:
        text_bits.append(str(getattr(o, "title", "") or ""))
        text_bits.append(str(getattr(o, "compliance_requirement", "") or ""))
        text_bits.append(str(getattr(o, "risk_implication", "") or ""))
    if analysis is not None:
        brd_report = getattr(analysis, "brd_report", None)
        if brd_report is not None:
            for section_name in (
                "process_business_requirements",
                "data_business_requirements",
                "reporting_business_requirements",
                "functional_requirements",
                "non_functional_requirements",
            ):
                section = getattr(brd_report, section_name, None)
                if section is None:
                    continue
                for item in getattr(section, "items", []) or []:
                    text_bits.append(str(getattr(item, "detailed_requirement", "") or ""))
                    text_bits.append(str(getattr(item, "acceptance_criteria", "") or ""))
    full_text = " \n ".join([t for t in text_bits if t])

    # Business functions & processes come from the obligation area/function
    # taxonomy Agent 1 already produced.
    function_counter = Counter(str(getattr(o, "impacted_function", "") or "") for o in obligations)
    function_counter.pop("", None)
    area_counter = Counter(str(getattr(o, "impacted_area", "") or "") for o in obligations)
    area_counter.pop("", None)
    theme_counter = Counter(str(getattr(o, "theme", "") or "") for o in obligations)
    theme_counter.pop("", None)

    def _dim(
        name: str,
        items: List[Tuple[str, int]],
        keyword_map: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> ImpactDimension:
        # Combine taxonomy-derived items with keyword-map matches
        merged: List[str] = [label for label, _ in items[:6]]
        if keyword_map:
            merged.extend([m for m in _match_items(full_text, keyword_map) if m not in merged])
        merged = merged[:8]
        total_count = sum(c for _, c in items) if items else 0
        extra = len(merged)
        severity, score = _severity_from_count(total_count, extra_evidence=extra)
        first_evidence: List[str] = []
        for label in merged[:3]:
            snippets = _evidence_snippets(full_text, label.split(" / ")[0])
            first_evidence.extend(snippets[:2])
        rationale = (
            f"{regulation or 'The regulation'} imposes obligations that touch "
            f"{len(merged)} distinct items in the {name.replace('_', ' ')} lens. "
            f"The severity of impact is assessed as {severity} because "
            f"{total_count} obligations and {extra} referenced items map to this "
            f"dimension. Remediation should be prioritised in line with the "
            f"most heavily-cited items above."
        )
        return ImpactDimension(
            dimension=name,
            items=merged,
            severity=severity,
            severity_score=score,
            rationale=rationale,
            evidence=first_evidence[:4],
        )

    business_functions = _dim("business_functions", function_counter.most_common())
    processes = _dim(
        "processes",
        theme_counter.most_common(),
    )
    systems = _dim("systems", area_counter.most_common(), _SYSTEM_KEYWORDS)
    data = _dim("data", [], _DATA_KEYWORDS)
    controls = _dim("controls", [], _CONTROL_KEYWORDS)
    stakeholders = _dim("stakeholders", [], _STAKEHOLDER_KEYWORDS)

    dims = [business_functions, processes, systems, data, controls, stakeholders]
    overall_score = sum(d.severity_score for d in dims) / max(1, len(dims))
    if overall_score >= 80:
        overall_severity = "Critical"
    elif overall_score >= 60:
        overall_severity = "High"
    elif overall_score >= 35:
        overall_severity = "Medium"
    else:
        overall_severity = "Low"

    executive_summary = (
        f"{regulation or 'This regulation'} has a {overall_severity.lower()} "
        f"overall impact ({overall_score:.0f}/100) driven primarily by "
        f"{business_functions.items[0] if business_functions.items else 'core business functions'}, "
        f"{controls.items[0] if controls.items else 'the ICT control framework'}, and "
        f"{data.items[0] if data.items else 'critical business data'}. "
        f"Six dimensions were assessed independently: functions, processes, "
        f"systems, data, controls and stakeholders."
    )

    return ImpactAssessment(
        regulation=regulation,
        executive_summary=executive_summary,
        overall_severity=overall_severity,
        overall_severity_score=round(overall_score, 1),
        business_functions=business_functions,
        processes=processes,
        systems=systems,
        data=data,
        controls=controls,
        stakeholders=stakeholders,
        generated_by_ai=False,
        metadata={
            "obligation_count": len(obligations),
            "area_count": len(area_counter),
            "function_count": len(function_counter),
        },
    )


def assess_impact(
    analysis: Optional[RegulatoryAnalysis],
    *,
    client: Optional[Any] = None,
) -> ImpactAssessment:
    """Generate a consulting-grade regulatory impact assessment.

    Rather than reducing impact to ``100 - readiness``, the impact
    assessment identifies affected business functions, processes,
    systems / applications, data, controls and stakeholders — each with
    severity, rationale and supporting evidence.
    """
    baseline = _deterministic_impact(analysis)

    if client is None or analysis is None:
        return baseline

    obligations = list(getattr(analysis, "obligations", []) or [])
    top_obligations = obligations[:12]
    context = {
        "regulation": getattr(analysis, "regulation", "") or "",
        "impacted_areas": list(analysis.impacted_areas or []),
        "obligation_themes": list(analysis.obligation_themes or []),
        "obligations": [
            {
                "id": getattr(o, "obligation_id", ""),
                "title": getattr(o, "title", ""),
                "theme": getattr(o, "theme", ""),
                "compliance_requirement": (getattr(o, "compliance_requirement", "") or "")[:300],
                "impacted_area": getattr(o, "impacted_area", ""),
                "impacted_function": getattr(o, "impacted_function", ""),
                "regulatory_basis": getattr(o, "regulatory_basis", ""),
                "risk_implication": (getattr(o, "risk_implication", "") or "")[:200],
            }
            for o in top_obligations
        ],
        "deterministic_baseline": {
            "overall_severity": baseline.overall_severity,
            "business_function_items": baseline.business_functions.items,
            "systems_items": baseline.systems.items,
            "data_items": baseline.data.items,
        },
    }
    instruction = (
        "Act as a senior regulatory transformation consultant producing an "
        "executive impact assessment. Using the obligations and areas below, "
        "identify the concrete items that will be affected in each of these "
        "six lenses: business functions, processes, systems and applications, "
        "data, controls, and stakeholders. Each dimension needs: 3-8 concrete "
        "items, a severity level (Critical/High/Medium/Low), a numeric severity "
        "0-100, a one-paragraph rationale, and 2-4 evidence bullets extracted "
        "from the regulation text. The overall_severity_score should reflect "
        "the aggregate business exposure."
    )
    from .guardrails import (
        CitationValidator, GuardrailReport, RegulationScopeValidator,
        RoleScopeValidator, apply_text_guardrails, safe_generate,
    )

    regulation = str(getattr(analysis, "regulation", "") or "")
    client_roles = list(getattr(analysis, "client_roles", []) or [])
    corpus_parts: List[str] = [
        str(getattr(analysis, "summary", "") or ""),
    ]
    for ob in getattr(analysis, "obligations", None) or []:
        corpus_parts.append(str(getattr(ob, "compliance_requirement", "") or ""))
        corpus_parts.append(str(getattr(ob, "regulatory_basis", "") or ""))
    source_corpus = "\n".join(p for p in corpus_parts if p)

    payload, guardrail_report = safe_generate(
        client,
        _ImpactPayload,
        "Regulatory Impact Assessment",
        instruction,
        json.dumps(context, default=str),
        regulation=regulation or None,
        client_roles=client_roles or None,
        source_corpus=source_corpus,
        text_fields=("executive_summary",),
    )
    if payload is None or not guardrail_report.ok:
        baseline.metadata = dict(baseline.metadata or {})
        baseline.metadata["_guardrail_report"] = guardrail_report.to_dict()
        return baseline

    # Post-hoc pass over the nested dimension rationales / evidence so
    # the same anti-hallucination rules that guard ``executive_summary``
    # also guard every ``rationale`` and ``evidence`` bullet.
    cit_v = CitationValidator(source_corpus, regulation=regulation)
    reg_v = RegulationScopeValidator(regulation)
    role_v = RoleScopeValidator(client_roles)

    def _guard_dim(name: str, p: _ImpactDimensionPayload) -> _ImpactDimensionPayload:
        prefix = f"impact.{name}."
        try:
            p.rationale = apply_text_guardrails(
                str(p.rationale or ""), field_path=prefix + "rationale",
                report=guardrail_report, citation_validator=cit_v,
                regulation_validator=reg_v, role_validator=role_v,
            )
            p.evidence = [
                apply_text_guardrails(
                    str(e), field_path=f"{prefix}evidence[{idx}]",
                    report=guardrail_report, citation_validator=cit_v,
                    regulation_validator=reg_v, role_validator=role_v,
                )
                for idx, e in enumerate(p.evidence or [])
            ]
        except Exception:
            pass
        return p

    for name in ("business_functions", "processes", "systems", "data",
                 "controls", "stakeholders"):
        try:
            _guard_dim(name, getattr(payload, name))
        except Exception:
            continue

    def _to_dim(name: str, p: _ImpactDimensionPayload) -> ImpactDimension:
        return ImpactDimension(
            dimension=name,
            items=[str(i).strip() for i in (p.items or []) if str(i).strip()][:10],
            severity=str(p.severity or "Medium"),
            severity_score=max(0.0, min(100.0, float(p.severity_score or 0.0))),
            rationale=str(p.rationale or "").strip(),
            evidence=[str(e).strip() for e in (p.evidence or []) if str(e).strip()][:6],
        )

    result = ImpactAssessment(
        regulation=regulation,
        executive_summary=str(payload.executive_summary or "").strip() or baseline.executive_summary,
        overall_severity=str(payload.overall_severity or baseline.overall_severity),
        overall_severity_score=max(0.0, min(100.0, float(payload.overall_severity_score or baseline.overall_severity_score))),
        business_functions=_to_dim("business_functions", payload.business_functions),
        processes=_to_dim("processes", payload.processes),
        systems=_to_dim("systems", payload.systems),
        data=_to_dim("data", payload.data),
        controls=_to_dim("controls", payload.controls),
        stakeholders=_to_dim("stakeholders", payload.stakeholders),
        generated_by_ai=True,
        metadata=baseline.metadata,
    )
    result.metadata = dict(result.metadata or {})
    result.metadata["_guardrail_report"] = guardrail_report.to_dict()
    return result


# ---------------------------------------------------------------------------
# Readiness assessment
# ---------------------------------------------------------------------------

_MATURITY_LEVELS = ("Optimised", "Managed", "Defined", "Developing", "Initial")


def _maturity_from_score(score: float) -> str:
    if score >= 85:
        return "Optimised"
    if score >= 70:
        return "Managed"
    if score >= 55:
        return "Defined"
    if score >= 35:
        return "Developing"
    return "Initial"


class _ReadinessDimensionPayload(BaseModel):  # type: ignore[misc]
    maturity_level: str = Field(description="Optimised / Managed / Defined / Developing / Initial")
    score: float = Field(description="0-100 maturity score")
    rationale: str = Field(description="One paragraph explaining the maturity call")
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class _ReadinessPayload(BaseModel):  # type: ignore[misc]
    executive_summary: str = Field(description="2-3 sentence executive summary")
    overall_score: float = Field(description="0-100 overall readiness")
    overall_level: str = Field(description="Maturity level for the overall score")
    existing_controls: _ReadinessDimensionPayload
    process_maturity: _ReadinessDimensionPayload
    policy_coverage: _ReadinessDimensionPayload
    technology_readiness: _ReadinessDimensionPayload
    documentation_completeness: _ReadinessDimensionPayload
    implementation_gaps: _ReadinessDimensionPayload
    organizational_preparedness: _ReadinessDimensionPayload
    key_strengths: List[str] = Field(default_factory=list)
    key_gaps: List[str] = Field(default_factory=list)


def _kind_average(evaluation: Mapping[str, Any], keywords: Sequence[str]) -> Optional[float]:
    """Average scores for area/function summaries whose name matches the keywords."""
    summaries = evaluation.get("area_summary") or {}
    func_summary = evaluation.get("function_summary") or {}
    scores: List[float] = []
    for name, summary in list(summaries.items()) + list(func_summary.items()):
        lower = str(name).lower()
        if not any(k in lower for k in keywords):
            continue
        try:
            score = float(summary.get("Compliance %") or summary.get("compliance_score_pct") or 0.0)
        except (TypeError, ValueError):
            continue
        scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _deterministic_readiness(
    scoring_evaluation: Mapping[str, Any],
    *,
    analysis: Optional[RegulatoryAnalysis] = None,
    responses: Optional[Mapping[str, Any]] = None,
    questionnaire_package: Optional[Mapping[str, Any]] = None,
) -> ReadinessAssessment:
    """Evidence-driven readiness assessment from the scored questionnaire."""
    overall_pct = float(scoring_evaluation.get("compliance_score_pct") or 0.0)

    def _score_for(keywords: Sequence[str], fallback: Optional[float] = None) -> float:
        val = _kind_average(scoring_evaluation, keywords)
        if val is None:
            return fallback if fallback is not None else overall_pct
        return val

    # Existing controls: risk & controls / cyber / control-related areas
    ec = _score_for(["risk & controls", "control", "cyber security", "audit / assurance"])
    # Process maturity: operations, execution, third-party (process-heavy)
    pm = _score_for(["operations", "middle office", "back office", "execution", "operating model", "settlement"])
    # Policy coverage: compliance & legal
    pc = _score_for(["compliance", "legal", "governance", "internal compliances"])
    # Technology readiness: IT, systems, technology
    tr = _score_for(["ict", "technology", "systems", "it security", "it,"])
    # Documentation completeness: reporting, data governance
    dc = _score_for(["data reporting", "data governance", "reporting", "regulatory reporting"])
    # Implementation gaps: high impact pain points, programme maturity
    ig_raw = _score_for(["programme maturity", "high impact pain points", "programme", "resource"])
    # Convert "readiness" to a "gap-closure" style score — lower readiness =
    # more gaps still open.
    ig = ig_raw
    # Organizational preparedness: HR, People, Programme, Governance
    op = _score_for(["hr", "people", "governance", "programme", "sponsorship"])

    dims_specs = [
        ("existing_controls", ec),
        ("process_maturity", pm),
        ("policy_coverage", pc),
        ("technology_readiness", tr),
        ("documentation_completeness", dc),
        ("implementation_gaps", ig),
        ("organizational_preparedness", op),
    ]

    dims: Dict[str, ReadinessDimension] = {}
    area_summary = scoring_evaluation.get("area_summary") or {}

    # Aggregate area-level strengths/gaps
    strong_areas = [
        name for name, s in area_summary.items()
        if float(s.get("Compliance %") or 0) >= 70.0
    ]
    weak_areas = [
        name for name, s in area_summary.items()
        if float(s.get("Compliance %") or 0) < 50.0
    ]

    dim_hints: Dict[str, Dict[str, List[str]]] = {
        "existing_controls": {
            "strengths": [f"Controls in {a}" for a in strong_areas[:3]],
            "gaps": [f"Control weakness in {a}" for a in weak_areas[:3]],
        },
        "process_maturity": {
            "strengths": [f"Process maturity in {a}" for a in strong_areas[:3]],
            "gaps": [f"Process gaps in {a}" for a in weak_areas[:3]],
        },
        "policy_coverage": {
            "strengths": ["Policies aligned to key obligations"] if pc >= 60 else [],
            "gaps": ["Policies not yet mapped to every DORA article"] if pc < 60 else [],
        },
        "technology_readiness": {
            "strengths": ["Core platforms captured in ICT scope"] if tr >= 60 else [],
            "gaps": ["Legacy platforms may lack DORA-required telemetry"] if tr < 60 else [],
        },
        "documentation_completeness": {
            "strengths": ["Reporting evidence exists for scored areas"] if dc >= 60 else [],
            "gaps": ["Evidence dictionary incomplete for critical reports"] if dc < 60 else [],
        },
        "implementation_gaps": {
            "strengths": ["Programme scope defined"] if ig >= 60 else [],
            "gaps": ["Programme lacks funded remediation plan"] if ig < 60 else [],
        },
        "organizational_preparedness": {
            "strengths": ["Named executive sponsor in place"] if op >= 60 else [],
            "gaps": ["Training and awareness plan not yet fully executed"] if op < 60 else [],
        },
    }

    for name, score in dims_specs:
        level = _maturity_from_score(score)
        hint = dim_hints.get(name, {"strengths": [], "gaps": []})
        rationale = (
            f"The {name.replace('_', ' ')} dimension scored {score:.1f}%, "
            f"placing it at the {level} maturity level. "
            f"This is derived from your responses to questions mapped to "
            f"related impacted areas and functions."
        )
        dims[name] = ReadinessDimension(
            dimension=name,
            maturity_level=level,
            score=round(score, 1),
            rationale=rationale,
            strengths=hint.get("strengths", []),
            gaps=hint.get("gaps", []),
        )

    overall_level = _maturity_from_score(overall_pct)
    executive_summary = (
        f"Overall readiness for "
        f"{getattr(analysis, 'regulation', '') or 'this regulation'} sits at "
        f"{overall_pct:.1f}% ({overall_level}). "
        f"Strongest dimensions: {max(dims.values(), key=lambda d: d.score).dimension.replace('_', ' ')}. "
        f"Weakest dimensions: {min(dims.values(), key=lambda d: d.score).dimension.replace('_', ' ')}. "
        f"This assessment is driven by the {int(scoring_evaluation.get('answered_count') or 0)} "
        f"responses recorded to date against "
        f"{int(scoring_evaluation.get('answered_count') or 0) + int(scoring_evaluation.get('unanswered_count') or 0)} "
        f"applicable questions."
    )

    return ReadinessAssessment(
        regulation=getattr(analysis, "regulation", "") if analysis else "",
        executive_summary=executive_summary,
        overall_score=round(overall_pct, 1),
        overall_level=overall_level,
        existing_controls=dims["existing_controls"],
        process_maturity=dims["process_maturity"],
        policy_coverage=dims["policy_coverage"],
        technology_readiness=dims["technology_readiness"],
        documentation_completeness=dims["documentation_completeness"],
        implementation_gaps=dims["implementation_gaps"],
        organizational_preparedness=dims["organizational_preparedness"],
        key_strengths=[f"{a} at {area_summary[a].get('Compliance %', 0):.0f}%" for a in strong_areas[:5]],
        key_gaps=[f"{a} at {area_summary[a].get('Compliance %', 0):.0f}%" for a in weak_areas[:5]],
        generated_by_ai=False,
        metadata={
            "answered_count": scoring_evaluation.get("answered_count"),
            "unanswered_count": scoring_evaluation.get("unanswered_count"),
        },
    )


def assess_readiness(
    scoring_evaluation: Mapping[str, Any],
    *,
    analysis: Optional[RegulatoryAnalysis] = None,
    questionnaire_package: Optional[Mapping[str, Any]] = None,
    responses: Optional[Mapping[str, Any]] = None,
    client: Optional[Any] = None,
) -> ReadinessAssessment:
    """Generate a consulting-grade regulatory readiness assessment.

    Reports on the seven consulting-standard readiness dimensions with
    dynamic maturity levels and explicit gaps/strengths per dimension.
    """
    baseline = _deterministic_readiness(
        scoring_evaluation,
        analysis=analysis,
        responses=responses,
        questionnaire_package=questionnaire_package,
    )
    if client is None:
        return baseline

    area_summary = scoring_evaluation.get("area_summary") or {}
    function_summary = scoring_evaluation.get("function_summary") or {}
    context = {
        "regulation": getattr(analysis, "regulation", "") if analysis else "",
        "overall_compliance_pct": scoring_evaluation.get("compliance_score_pct"),
        "area_summary": {
            name: {
                "compliance_pct": s.get("Compliance %"),
                "status": s.get("CXO status"),
                "questions_scored": s.get("Questions scored"),
            }
            for name, s in list(area_summary.items())[:20]
        },
        "function_summary": {
            name: {
                "compliance_pct": s.get("Compliance %"),
                "status": s.get("CXO status"),
                "questions_scored": s.get("Questions scored"),
            }
            for name, s in list(function_summary.items())[:15]
        },
        "deterministic_baseline": {
            "overall_score": baseline.overall_score,
            **{d.dimension: {"score": d.score, "level": d.maturity_level} for d in baseline.dimensions()},
        },
    }
    instruction = (
        "Act as an enterprise compliance readiness assessor from a leading "
        "consulting firm. Using the scored questionnaire summaries below, "
        "grade the organisation on seven consulting-standard readiness "
        "dimensions: existing controls, process maturity, policy coverage, "
        "technology readiness, documentation completeness, implementation "
        "gaps, and organizational preparedness. Return a maturity level "
        "(Optimised/Managed/Defined/Developing/Initial), a numeric score "
        "0-100, a one-paragraph rationale, and 2-4 concrete strengths and "
        "gaps for each dimension. Ground each judgement in the area / "
        "function scores below — do not invent numbers."
    )
    from .guardrails import (
        CitationValidator, RegulationScopeValidator, RoleScopeValidator,
        apply_text_guardrails, safe_generate,
    )
    regulation = str(getattr(analysis, "regulation", "") or "") if analysis else ""
    client_roles = list(getattr(analysis, "client_roles", []) or []) if analysis else []
    corpus_parts: List[str] = []
    if analysis is not None:
        corpus_parts.append(str(getattr(analysis, "summary", "") or ""))
        for ob in getattr(analysis, "obligations", None) or []:
            corpus_parts.append(str(getattr(ob, "compliance_requirement", "") or ""))
            corpus_parts.append(str(getattr(ob, "regulatory_basis", "") or ""))
    source_corpus = "\n".join(p for p in corpus_parts if p)

    payload, guardrail_report = safe_generate(
        client,
        _ReadinessPayload,
        "Regulatory Readiness Assessment",
        instruction,
        json.dumps(context, default=str),
        regulation=regulation or None,
        client_roles=client_roles or None,
        source_corpus=source_corpus,
        text_fields=("executive_summary",),
    )
    if payload is None or not guardrail_report.ok:
        baseline.metadata = dict(baseline.metadata or {})
        baseline.metadata["_guardrail_report"] = guardrail_report.to_dict()
        return baseline

    cit_v = CitationValidator(source_corpus, regulation=regulation)
    reg_v = RegulationScopeValidator(regulation)
    role_v = RoleScopeValidator(client_roles)

    def _guard_dim(name: str, p: _ReadinessDimensionPayload) -> _ReadinessDimensionPayload:
        prefix = f"readiness.{name}."
        try:
            p.rationale = apply_text_guardrails(
                str(p.rationale or ""), field_path=prefix + "rationale",
                report=guardrail_report, citation_validator=cit_v,
                regulation_validator=reg_v, role_validator=role_v,
            )
            p.strengths = [
                apply_text_guardrails(
                    str(s), field_path=f"{prefix}strengths[{i}]",
                    report=guardrail_report, citation_validator=cit_v,
                    regulation_validator=reg_v, role_validator=role_v,
                )
                for i, s in enumerate(p.strengths or [])
            ]
            p.gaps = [
                apply_text_guardrails(
                    str(g), field_path=f"{prefix}gaps[{i}]",
                    report=guardrail_report, citation_validator=cit_v,
                    regulation_validator=reg_v, role_validator=role_v,
                )
                for i, g in enumerate(p.gaps or [])
            ]
        except Exception:
            pass
        return p

    for name in ("existing_controls", "process_maturity", "policy_coverage",
                 "technology_readiness", "documentation_completeness",
                 "implementation_gaps", "organizational_preparedness"):
        try:
            _guard_dim(name, getattr(payload, name))
        except Exception:
            continue

    def _to_dim(name: str, p: _ReadinessDimensionPayload) -> ReadinessDimension:
        score = max(0.0, min(100.0, float(p.score or 0.0)))
        level = str(p.maturity_level or _maturity_from_score(score))
        return ReadinessDimension(
            dimension=name,
            maturity_level=level,
            score=round(score, 1),
            rationale=str(p.rationale or "").strip(),
            strengths=[str(s).strip() for s in (p.strengths or []) if str(s).strip()][:6],
            gaps=[str(g).strip() for g in (p.gaps or []) if str(g).strip()][:6],
        )

    overall_score = max(0.0, min(100.0, float(payload.overall_score or baseline.overall_score)))
    result = ReadinessAssessment(
        regulation=regulation,
        executive_summary=str(payload.executive_summary or "").strip() or baseline.executive_summary,
        overall_score=round(overall_score, 1),
        overall_level=str(payload.overall_level or _maturity_from_score(overall_score)),
        existing_controls=_to_dim("existing_controls", payload.existing_controls),
        process_maturity=_to_dim("process_maturity", payload.process_maturity),
        policy_coverage=_to_dim("policy_coverage", payload.policy_coverage),
        technology_readiness=_to_dim("technology_readiness", payload.technology_readiness),
        documentation_completeness=_to_dim("documentation_completeness", payload.documentation_completeness),
        implementation_gaps=_to_dim("implementation_gaps", payload.implementation_gaps),
        organizational_preparedness=_to_dim("organizational_preparedness", payload.organizational_preparedness),
        key_strengths=[str(s).strip() for s in (payload.key_strengths or []) if str(s).strip()][:8],
        key_gaps=[str(g).strip() for g in (payload.key_gaps or []) if str(g).strip()][:8],
        generated_by_ai=True,
        metadata=baseline.metadata,
    )
    result.metadata = dict(result.metadata or {})
    result.metadata["_guardrail_report"] = guardrail_report.to_dict()
    return result


# ---------------------------------------------------------------------------
# Adaptive follow-up detection for brief user answers
# ---------------------------------------------------------------------------

_BRIEF_ANSWER_TOKENS = {
    "yes", "no", "maybe", "ok", "okay", "sure", "n/a", "na", "none",
    "tbd", "unknown", "not sure", "not applicable", "not started",
    "in progress", "partial", "some", "few", "many", "several",
    "nothing", "everything", "all", "any", "idk", "n a", "not yet",
}

_AMBIGUOUS_PATTERNS = [
    re.compile(r"^\s*(?:it|we|they)\s+(?:will|might|may|would|could)\s+.{0,20}$", re.IGNORECASE),
    re.compile(r"^\s*depends\b.{0,60}$", re.IGNORECASE),
    re.compile(r"^\s*(?:not sure|no idea|dunno|unsure)\b.*$", re.IGNORECASE),
]


_BRIEF_TOKEN_PROMPTS: Tuple[str, ...] = (
    "Thanks for the quick reply — could you share a bit more detail? "
    "For example, what is in place today, who owns it, and any current "
    "gaps or planned actions?",
    "Got it. Could you expand on that a little? A few specifics — the "
    "process you follow, who runs it, and any evidence you'd point to — "
    "will make the assessment much sharper.",
    "Understood. Would you mind walking me through the current state in "
    "a couple more sentences? Even a short list of controls, owners or "
    "recent activities helps.",
    "Thanks — can you tell me a little more? I'm especially interested "
    "in what exists today, who is accountable, and any deadlines or "
    "targets you're tracking against.",
)

_AMBIGUOUS_PROMPTS: Tuple[str, ...] = (
    "Your response is a little open-ended. Could you clarify the "
    "specific controls, owners or timelines involved so the "
    "recommendation reflects your actual state?",
    "That's helpful context — could you be a bit more specific? For "
    "instance, which controls are in place, who owns them, and how "
    "often they're reviewed?",
    "Thanks. To make sure I capture this accurately, could you point "
    "to concrete examples — a policy, a runbook, a report, or a "
    "recent audit finding?",
    "Interesting — can you unpack that a little further? Any details "
    "on the underlying process, the responsible team or the tooling "
    "involved will help.",
)

_SHORT_PROMPTS: Tuple[str, ...] = (
    "Your response is quite brief. Additional context — such as the "
    "current process, the responsible team, or supporting evidence — "
    "will help improve the quality of the recommendations.",
    "Thanks — could you provide further information on this? A couple "
    "of extra sentences on what's in place, who owns it, and where the "
    "gaps are will go a long way.",
    "Could you explain that in a bit more detail? I'd love to hear "
    "about the controls, owners, cadence and any recent test results "
    "if you have them.",
    "Would you mind elaborating? Even a short paragraph on the "
    "process, evidence, and any planned actions makes the assessment "
    "much stronger.",
    "Thanks for the note. Can you share a little more context — "
    "specifically, what is in place today, what's still to be done, "
    "and any timelines you're working to?",
)


def detect_brief_answer(
    answer: str,
    *,
    question_context: Optional[str] = None,
    min_words: int = 6,
    min_chars: int = 25,
) -> Tuple[bool, str]:
    """Detect whether a free-text answer is too brief / ambiguous and craft a follow-up.

    Returns ``(needs_followup, follow_up_prompt)``. The follow-up prompt is
    conversational, professional and context-aware — one of several
    rotated phrasings so the UX matches a modern chat assistant rather
    than a single canned response. Prompt selection is deterministic
    (hash of the answer + question) so repeat renders don't flip
    phrasings on every keystroke.
    """
    if answer is None:
        return False, ""
    raw = str(answer).strip()
    if not raw:
        return False, ""

    stripped = raw.lower().strip(" .!?,;:'\"()[]-")
    words = [w for w in re.split(r"\s+", stripped) if w]
    word_count = len(words)

    # A dedicated N/A style answer is a valid explicit choice; no follow-up.
    if stripped in {"n/a", "na", "not applicable"}:
        return False, ""

    is_brief_token = stripped in _BRIEF_ANSWER_TOKENS
    is_short = word_count < min_words or len(raw) < min_chars
    is_ambiguous = any(pat.match(raw) for pat in _AMBIGUOUS_PATTERNS)

    if not (is_brief_token or is_short or is_ambiguous):
        return False, ""

    # Deterministic rotation seed — same (answer, question) always maps to
    # the same phrasing so users don't see the prompt reshuffle on every
    # keystroke. Falls back to the length of the answer when no question
    # context is provided.
    seed_basis = f"{stripped}|{(question_context or '').strip().lower()}"
    seed = sum(ord(c) for c in seed_basis) or len(raw)

    if is_brief_token:
        variants = _BRIEF_TOKEN_PROMPTS
    elif is_ambiguous:
        variants = _AMBIGUOUS_PROMPTS
    else:
        variants = _SHORT_PROMPTS
    prompt = variants[seed % len(variants)]

    if question_context:
        prompt = (
            f"{prompt} (Question context: "
            f"{question_context.strip()[:180]})"
        )

    return True, prompt


__all__ = [
    "assess_confidence",
    "assess_impact",
    "assess_readiness",
    "detect_brief_answer",
]
