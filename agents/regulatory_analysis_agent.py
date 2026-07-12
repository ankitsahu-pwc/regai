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
from services.client_profile import (
    client_profile_context_text,
    is_client_profile_populated,
    normalize_client_profile,
)
from services.client_roles import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_PARTIAL,
    APPLICABILITY_UNCERTAIN,
    RoleApplicability,
    RoleAwareInterpretation,
    build_role_aware_interpretation,
    derive_role_applicability,
    get_institution_type,
    normalize_client_roles,
)
from services.genai_service import GenAIClient
from services.obligation_verb import classify_verb_from_sources
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
        client_roles: Optional[Sequence[str]] = None,
        client_profile: Optional[Dict[str, Any]] = None,
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
        client_roles
            Institution types the analysis should be scoped to (e.g.
            ``["Commercial Bank", "Broker Dealer (Small)"]``). Drives the
            Client Role-Aware Regulatory Interpretation: obligations are
            tagged with per-role applicability, and out-of-scope items are
            marked so downstream agents can filter accordingly. Empty list =
            generic (pre-role-aware) analysis.
        client_profile
            Optional keyword profile collected on Page 1 (organization
            profile, business lines, products in scope, countries of
            operation, legal entities, vendor & third parties). Each value
            is a list of keywords (curated + free-form). The keywords are
            appended to the interpretation engine's context corpus and
            prepended to the BRD LLM prompt so the generated content is
            scoped to the profile, not to a generic FS baseline.
        """
        roles = normalize_client_roles(client_roles) if client_roles else []
        profile = normalize_client_profile(client_profile) if client_profile else {}
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
            client_roles=roles,
            client_profile=profile,
        )

        source_refs_map: Dict[str, List[Dict[str, Any]]] = (
            metadata.get("source_references_by_item") or {}
        )
        obligations = self._extract_obligations(report, source_refs_map)

        regulation_context = self._build_regulation_context(
            metadata=metadata,
            extra_context=extra_context,
        )
        # Splice the Client Profile keywords into the corpus the
        # deterministic interpretation engine reads. That way profile
        # keywords count as regulatory-surface signal when the engine
        # scores per-role applicability — a Digital Bank profile with
        # ``Cloud Service Provider`` in Vendor & Third Parties will
        # correctly bias obligations touching cloud outsourcing.
        profile_context = client_profile_context_text(profile) if profile else ""
        if profile_context:
            regulation_context = (
                f"{regulation_context}\n\n--- Client Profile Keywords ---\n"
                f"{profile_context}"
            )

        interpretation = build_role_aware_interpretation(
            regulation=regulation,
            client_roles=roles,
            regulation_context=regulation_context,
            obligations=obligations,
        )

        # Tag each obligation with the per-role applicability record so
        # downstream agents (BRD, RTM, questionnaire, recommendations) never
        # need to reinterpret the regulation. This mutates obligations in
        # place; interpretation.per_obligation_applicability keeps the same
        # RoleApplicability records for the exported bundle.
        for obligation in obligations:
            rows = interpretation.per_obligation_applicability.get(
                obligation.obligation_id, []
            )
            self._tag_obligation_with_roles(obligation, rows, roles=roles)

        themes = sorted({o.theme for o in obligations})
        areas = sorted({o.impacted_area for o in obligations})

        if roles:
            in_scope_count = sum(
                1 for o in obligations if o.is_applicable_for(roles)
            )
            role_label = ", ".join(roles)
            summary = (
                f"Client role-aware regulatory analysis of {regulation} "
                f"({tier}) for {role_label}: {in_scope_count} of "
                f"{len(obligations)} obligations are in scope for the "
                f"selected institution type(s), across {len(areas)} impacted "
                f"areas and {len(themes)} obligation themes. Overall BRD "
                f"coverage confidence is {calculate_overall_confidence(report)}."
            )
        else:
            summary = (
                f"Regulatory analysis for {regulation} ({tier}) produced "
                f"{len(obligations)} obligations across {len(areas)} impacted areas "
                f"and {len(themes)} obligation themes. Overall BRD coverage confidence "
                f"is {calculate_overall_confidence(report)}. "
                f"No client role was selected — the output is generic; select an "
                f"institution type on Page 1 to enable role-aware interpretation."
            )

        # ------------------------------------------------------------------
        # Guardrail sweep across every obligation. This is a *deterministic*
        # post-hoc pass — no additional LLM calls — that:
        #
        # * scrubs any AI meta-leakage the model may have slipped into the
        #   BRD requirement text before it was carried onto the obligation
        #   ("As an AI language model…", "OpenAI", etc.);
        # * replaces citations (Article / RTS / Chapter / …) that do NOT
        #   appear anywhere in the regulation corpus with the standard
        #   "[citation not verified against source]" marker so the
        #   downstream artefacts (BRD, RTM, questionnaire) never silently
        #   propagate a fabricated reference;
        # * flags mentions of institution types that are NOT in the
        #   selected client-role list, so silent scope expansion is
        #   visible on the audit trail; and
        # * flags off-scope regulation names (a DORA run referencing MiFID
        #   II applicability, etc.).
        #
        # The aggregated report is attached to the analysis metadata so
        # the Streamlit UI can render the anti-hallucination audit trail
        # on Page 2 next to the Regulatory Analysis.
        # ------------------------------------------------------------------
        from services.guardrails import (
            CitationValidator, GuardrailReport, RegulationScopeValidator,
            RoleScopeValidator, apply_text_guardrails,
        )

        agent_report = GuardrailReport(component="regulatory_analysis_agent")
        citation_v = CitationValidator(regulation_context, regulation=regulation)
        regulation_v = RegulationScopeValidator(regulation)
        role_v = RoleScopeValidator(roles) if roles else None

        for obligation in obligations:
            prefix = f"obligation[{obligation.obligation_id}]."
            for attr in (
                "title", "theme", "compliance_requirement", "regulatory_basis",
                "risk_implication", "role_interpretation",
            ):
                value = getattr(obligation, attr, None)
                if isinstance(value, str) and value:
                    new_value = apply_text_guardrails(
                        value, field_path=prefix + attr,
                        report=agent_report,
                        citation_validator=citation_v,
                        regulation_validator=regulation_v,
                        role_validator=role_v,
                    )
                    if new_value != value:
                        try:
                            setattr(obligation, attr, new_value)
                        except Exception:
                            pass
            for list_attr in ("control_expectations", "evidence_needs"):
                items = getattr(obligation, list_attr, None) or []
                new_items: List[str] = []
                for idx, item in enumerate(items):
                    if isinstance(item, str) and item:
                        new_items.append(apply_text_guardrails(
                            item, field_path=f"{prefix}{list_attr}[{idx}]",
                            report=agent_report,
                            citation_validator=citation_v,
                            regulation_validator=regulation_v,
                            role_validator=role_v,
                        ))
                    else:
                        new_items.append(item)
                try:
                    setattr(obligation, list_attr, new_items)
                except Exception:
                    pass

        # Persist client roles + profile + interpretation on the analysis
        # metadata so exports and persistence pick them up automatically.
        metadata_with_roles = dict(metadata)
        metadata_with_roles["client_roles"] = list(roles)
        metadata_with_roles["client_profile"] = dict(profile)
        metadata_with_roles["role_interpretation"] = interpretation.to_dict()
        # Merge the agent-level guardrail sweep into the BRD-level
        # guardrail bundle so the UI has a single place to look.
        agent_report_dict = agent_report.to_dict()
        existing = dict(metadata_with_roles.get("guardrails") or {})
        existing.setdefault("reports", [])
        existing["reports"] = list(existing["reports"]) + [agent_report_dict]
        totals = dict(existing.get("totals") or {})
        totals["citations_verified"] = int(totals.get("citations_verified", 0)) + agent_report.citations_verified
        totals["citations_flagged"] = int(totals.get("citations_flagged", 0)) + agent_report.citations_flagged
        totals["meta_leaks_scrubbed"] = int(totals.get("meta_leaks_scrubbed", 0)) + agent_report.meta_leaks_scrubbed
        totals.setdefault("off_scope_regulations", [])
        totals["off_scope_regulations"] = sorted(set(
            list(totals["off_scope_regulations"]) + list(agent_report.off_scope_regulations)
        ))
        totals.setdefault("off_scope_roles", [])
        totals["off_scope_roles"] = sorted(set(
            list(totals["off_scope_roles"]) + list(agent_report.off_scope_roles)
        ))
        totals["bundles_run"] = int(totals.get("bundles_run", 0)) + 1
        existing["totals"] = totals
        metadata_with_roles["guardrails"] = existing

        return RegulatoryAnalysis(
            regulation=regulation,
            tier=tier,
            summary=summary,
            impacted_areas=areas,
            obligation_themes=themes,
            obligations=obligations,
            used_genai=bool(metadata.get("used_genai_shared_service")),
            metadata=metadata_with_roles,
            brd_report=report,
            client_roles=list(roles),
            role_interpretation=interpretation.to_dict(),
            client_profile=dict(profile),
        )

    # ------------------------------------------------------------------
    # Role-aware helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regulation_context(
        *,
        metadata: Dict[str, Any],
        extra_context: Optional[str],
    ) -> str:
        """Assemble a text corpus the interpretation engine can scan for
        role-specific keywords.

        Combines:
          * the uploaded regulation text (when provided);
          * every retrieved regulator publication snippet / title; and
          * the BRD's source-references catalogue titles.

        The engine only *reads* this string — it does not send it to a
        third-party service.
        """
        chunks: List[str] = []
        if extra_context:
            chunks.append(extra_context)
        for row in metadata.get("all_sources_ranked") or []:
            chunks.append(str(row.get("title") or ""))
            chunks.append(str(row.get("snippet") or ""))
        for row in metadata.get("source_references_catalogue") or []:
            chunks.append(str(row.get("title") or ""))
            chunks.append(str(row.get("regulation_reference") or ""))
        for row in metadata.get("official_sources") or []:
            chunks.append(str(row.get("title") or ""))
            chunks.append(str(row.get("snippet") or ""))
        return " \n".join(c for c in chunks if c)

    @staticmethod
    def _tag_obligation_with_roles(
        obligation: Obligation,
        rows: Sequence[RoleApplicability],
        *,
        roles: Sequence[str],
    ) -> None:
        """Populate the client-role fields on ``obligation`` in place.

        ``rows`` is the ordered list of
        :class:`~services.client_roles.RoleApplicability` records produced by
        :func:`build_role_aware_interpretation`. When no roles were selected
        the fields are left empty and downstream agents fall back to their
        pre-role-aware behaviour.
        """
        if not rows and not roles:
            return

        role_applicability: List[Dict[str, Any]] = []
        applicable: List[str] = []
        partial: List[str] = []
        not_applicable: List[str] = []
        uncertain: List[str] = []
        interpretation_bits: List[str] = []

        for row in rows:
            role_applicability.append(row.to_dict())
            if row.applicability == APPLICABILITY_APPLICABLE:
                applicable.append(row.role)
            elif row.applicability == APPLICABILITY_PARTIAL:
                partial.append(row.role)
            elif row.applicability == APPLICABILITY_NOT_APPLICABLE:
                not_applicable.append(row.role)
            else:
                uncertain.append(row.role)
            interpretation_bits.append(
                f"{row.role}: {row.applicability} — {row.rationale}"
            )

        obligation.applicable_roles = applicable
        obligation.partial_roles = partial
        obligation.not_applicable_roles = not_applicable
        obligation.uncertain_roles = uncertain
        obligation.role_applicability = role_applicability
        obligation.role_interpretation = "\n".join(interpretation_bits)

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

            obligation_verb = classify_verb_from_sources((
                req.requirement,
                req.detail,
                req.alignment,
            ))

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
                obligation_verb=obligation_verb,
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
    obligation_verb = classify_verb_from_sources((
        cp.requirement,
        cp.control_checkpoint,
    )) or "Must"
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
        obligation_verb=obligation_verb,
        confidence=95,
        source_references=list(sources or []),
    )


__all__ = ["RegulatoryAnalysisAgent"]
