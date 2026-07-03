"""Agent 1 — Regulatory Analysis.

Role in the pipeline
--------------------
Takes a parsed regulation document (and any extra context) and produces a
structured :class:`~models.workflow_models.RegulatoryAnalysis` containing a
list of :class:`~models.workflow_models.Obligation` records.

For each obligation we extract:

* impacted area + impacted function (downstream Agent 2/3 routing)
* obligation theme (e.g. ``ICT Risk Management``)
* compliance requirement statement
* deadline (when surfaced in the source text)
* control expectations
* evidence needs
* risk implication

Implementation strategy
-----------------------
We reuse the existing :func:`services.brd_frd_generator.build_brd_frd_report`
pipeline. That function already:

* Hits the PwC GenAI Shared Service when available, falls back to
  deterministic offline content when not.
* Returns a richly-structured ``DoraDetailedBRD`` containing requirement
  tables aligned to DORA articles, plus a control framework, risks, etc.

We then derive obligations from the BRD's requirement rows + control
checkpoints, classifying each row into impacted area/function via the keyword
taxonomies already maintained in
:mod:`services.questionnaire_generator`. This avoids duplicating taxonomy /
LLM logic and keeps a single source of truth for regulatory wording.

Carrying the generated ``DoraDetailedBRD`` on the result means Agent 2 does
not have to regenerate it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from models.workflow_models import (
    Obligation,
    ParsedDocument,
    RegulatoryAnalysis,
)
from services.brd_frd_generator import (
    ControlCheckpointItem,
    DoraDetailedBRD,
    RequirementItem,
    build_brd_frd_report,
    calculate_overall_confidence,
)
from services.genai_service import GenAIClient
from services.regulatory_intelligence_service import RegulatoryIntelligencePackage
from services.source_traceability import control_key, requirement_key
from services.questionnaire_generator import (
    AREA_KEYWORDS,
    FUNCTION_KEYWORDS,
    impacted_labels_for_requirement,
    infer_themes,
    requirements_from_report,
)

StatusCallback = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


class RegulatoryAnalysisAgent:
    """Agent 1 implementation.

    The agent is stateless; each ``analyze`` call produces a self-contained
    ``RegulatoryAnalysis`` bundle.
    """

    def __init__(self, *, client: Optional[GenAIClient] = None) -> None:
        self.client = client

    def analyze(
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
    ) -> RegulatoryAnalysis:
        """Produce a :class:`RegulatoryAnalysis` for the supplied input.

        Parameters
        ----------
        parsed_document
            Output of :func:`services.document_parser.parse_document`. May be
            ``None`` when the user only supplies a regulation code; the BRD
            generator will then rely entirely on context retrieved from the
            Regulatory Intelligence Pipeline (Stage 1 + optional Stage 2).
        regulation
            Free-form regulation label (``"DORA"``, ``"MiFID II"``, etc.).
        tier
            Institutional tier label, used by the BRD generator for
            proportionality.
        status
            Optional callable used to surface progress messages to the UI
            (``st.status`` writer).
        regulator_selection
            Optional list of regulator codes to scope Stage 1 to (``["EBA",
            "ESMA"]``). ``None`` / ``["ALL"]`` searches every approved
            regulator.
        consulting_selection
            Optional list of consulting firm codes to scope Stage 2 to.
        include_consulting_guidance
            Set to False to skip Stage 2 (regulator-only context).
        intelligence_package
            Optional pre-built :class:`RegulatoryIntelligencePackage` reused
            from a previous call (e.g. when the UI previewed sources on Page 1
            and we don't want to double-query the network).
        """
        extra_context = parsed_document.text if parsed_document and not parsed_document.is_empty else None
        report, metadata = build_brd_frd_report(
            regulation=regulation,
            tier=tier,
            extra_context=extra_context,
            status=status,
            client=self.client,
            regulator_selection=regulator_selection,
            consulting_selection=consulting_selection,
            include_consulting_guidance=include_consulting_guidance,
            intelligence_package=intelligence_package,
        )

        source_refs_map: Dict[str, List[Dict[str, Any]]] = (
            metadata.get("source_references_by_item") or {}
        )
        obligations = self._extract_obligations(report, source_refs_map)
        themes = sorted({o.theme for o in obligations})
        areas = sorted({o.impacted_area for o in obligations})

        summary = (
            f"Regulatory analysis for {regulation} ({tier}) produced "
            f"{len(obligations)} obligations across {len(areas)} impacted areas "
            f"and {len(themes)} obligation themes. Overall BRD coverage confidence "
            f"is {calculate_overall_confidence(report)}."
        )

        return RegulatoryAnalysis(
            regulation=regulation,
            tier=tier,
            summary=summary,
            impacted_areas=areas,
            obligation_themes=themes,
            obligations=obligations,
            used_genai=bool(metadata.get("used_genai_shared_service")),
            metadata=metadata,
            brd_report=report,
        )

    # ------------------------------------------------------------------
    # Obligation extraction (deterministic)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_obligations(
        report: DoraDetailedBRD,
        source_refs_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Obligation]:
        """Convert BRD/FRD requirements + control checkpoints into obligations.

        ``source_refs_map`` is the ``{REQ:<id> | CTRL:<stage>:<checkpoint>
        -> [SourceReference dict]}`` mapping produced by
        :func:`services.source_traceability.attach_source_references` and
        carried on ``RegulatoryAnalysis.metadata``. When supplied the agent
        copies each requirement's citation list onto the resulting
        ``Obligation`` so downstream agents (Agent 2 RTM, the BRD/Questionnaire
        UI) can render source references without re-running the matcher.
        """
        source_refs_map = source_refs_map or {}
        requirements = requirements_from_report(report)

        obligations: List[Obligation] = []
        seq = 1
        for req in requirements:
            areas = impacted_labels_for_requirement(req, AREA_KEYWORDS, "Risk & Controls framework")
            functions = impacted_labels_for_requirement(req, FUNCTION_KEYWORDS, "Compliance & Legal")
            primary_area = areas[0] if areas else "Risk & Controls framework"
            primary_function = functions[0] if functions else "Compliance & Legal"
            theme = (req.themes[0] if req.themes else "General regulatory coverage")

            # Citations follow the BRD requirement that produced this
            # obligation. ``req.source_id`` is the BRD-assigned ID (e.g.
            # ``BR-PRO-001``) which is what the source-references map is
            # keyed on.
            sources = list(source_refs_map.get(requirement_key(req.source_id), []))

            obligations.append(Obligation(
                obligation_id=f"OBL-{seq:03d}",
                title=req.requirement or req.category or req.normalized_id,
                theme=theme,
                compliance_requirement=req.detail or req.requirement,
                impacted_area=primary_area,
                impacted_function=primary_function,
                deadline=_deadline_hint(req.detail or req.requirement or req.alignment),
                control_expectations=_control_expectations(req, report.control_framework.lifecycle_checkpoints),
                evidence_needs=_evidence_needs(req),
                risk_implication=_risk_implication(req, report),
                source_requirement_id=req.normalized_id,
                regulatory_basis=req.alignment or req.source_section,
                priority=req.priority or "Should",
                confidence=int(req.confidence or 92),
                source_references=sources,
            ))
            seq += 1

        # Control framework checkpoints surface obligations that are not always
        # captured as BR rows (e.g. governance-only checkpoints).
        for cp in report.control_framework.lifecycle_checkpoints:
            cp_sources = list(source_refs_map.get(
                control_key(cp.stage, cp.control_checkpoint), [],
            ))
            obligations.append(_obligation_from_checkpoint(cp, seq, cp_sources))
            seq += 1

        return obligations


# ---------------------------------------------------------------------------
# Helpers (kept private and free of LLM calls so they work offline)
# ---------------------------------------------------------------------------

_DEADLINE_KEYWORDS = (
    "within", "no later than", "by ", "deadline", "calendar days",
    "business days", "annually", "quarterly", "monthly", "weekly",
)


def _deadline_hint(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for kw in _DEADLINE_KEYWORDS:
        if kw in lower:
            # Surface a short snippet around the matched keyword.
            idx = lower.index(kw)
            start = max(0, idx - 12)
            end = min(len(text), idx + 80)
            return text[start:end].strip()
    return None


def _control_expectations(req: Any, checkpoints: List[ControlCheckpointItem]) -> List[str]:
    text = " ".join([
        getattr(req, "requirement", ""),
        getattr(req, "detail", ""),
        getattr(req, "acceptance", ""),
    ]).lower()
    matches: List[str] = []
    for cp in checkpoints:
        haystack = " ".join([cp.control_checkpoint, cp.requirement, cp.stage]).lower()
        if any(token and token in haystack for token in text.split()[:25]):
            matches.append(f"{cp.stage}: {cp.control_checkpoint}")
        if len(matches) >= 3:
            break
    if not matches:
        matches = ["Establish documented control aligned to the obligation."]
    return matches


_EVIDENCE_DEFAULTS = (
    "Policy / procedure document",
    "Workflow record",
    "Dashboard or KPI extract",
    "Approval / attestation log",
    "Test or audit report",
)


def _evidence_needs(req: Any) -> List[str]:
    text = " ".join([
        getattr(req, "requirement", ""),
        getattr(req, "detail", ""),
        getattr(req, "acceptance", ""),
    ]).lower()
    themes = infer_themes(text)
    bucket: List[str] = []
    if "data" in text or "register" in text or "Data and evidence" in themes:
        bucket.append("System extract with mandatory data fields")
    if "incident" in text or "Incident reporting" in themes:
        bucket.append("Incident workflow record with timestamps")
    if "contract" in text or "third-party" in text or "Third-party risk" in themes:
        bucket.append("Contract clause assessment / vendor register entry")
    if "test" in text or "Resilience testing" in themes:
        bucket.append("Resilience testing report and lessons learned")
    if "governance" in text or "Governance" in themes:
        bucket.append("Governance forum minutes and decision log")
    if not bucket:
        bucket = list(_EVIDENCE_DEFAULTS[:3])
    return bucket[:4]


def _risk_implication(req: Any, report: DoraDetailedBRD) -> str:
    priority = getattr(req, "priority", "") or "Should"
    impact = "regulatory readiness gap"
    if priority.lower() == "must":
        impact = "material regulatory exposure and possible supervisory finding"
    elif priority.lower() == "could":
        impact = "operational maturity gap with limited regulatory exposure"
    alignment = getattr(req, "alignment", "") or "the mapped regulation"
    return (
        f"Failure to satisfy this obligation may result in a {impact} against "
        f"{alignment}. Tier proportionality and existing compensating controls "
        f"should be considered when prioritising remediation."
    )


def _obligation_from_checkpoint(
    cp: ControlCheckpointItem,
    seq: int,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Obligation:
    theme = f"Control: {cp.stage}"
    return Obligation(
        obligation_id=f"OBL-{seq:03d}",
        title=cp.control_checkpoint,
        theme=theme,
        compliance_requirement=cp.requirement,
        impacted_area="Risk & Controls framework",
        impacted_function="Compliance & Legal",
        deadline=None,
        control_expectations=[f"{cp.stage}: {cp.control_checkpoint}"],
        evidence_needs=[cp.evidence] if cp.evidence else [],
        risk_implication=(
            f"Weakness in {cp.control_checkpoint} during the {cp.stage} stage "
            f"may compromise the regulatory control objective."
        ),
        source_requirement_id=f"CTRL-{cp.stage[:3].upper()}-{seq:03d}",
        regulatory_basis=f"Control framework / {cp.stage}",
        priority="Must",
        confidence=95,
        source_references=list(sources or []),
    )


__all__ = ["RegulatoryAnalysisAgent"]
