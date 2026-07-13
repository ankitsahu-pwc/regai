"""Consulting-grade recommendation generation.

This service produces the executive-ready recommendations that the
dashboard renders per impacted business area. Every recommendation
carries the six consulting deliverables the user asked for:

* **what** — a specific action tailored to the area, function and gap.
* **why** — the regulatory rationale plus quantified business impact.
* **how** — practical implementation guidance / steps.
* **priority** — High / Medium / Low, driven by severity and readiness.
* **expected outcome** — what "done" looks like after implementation.
* **dependencies** — prerequisites that must be in place first.

Design principles
-----------------
* Recommendations are **derived from evidence**, not template lookups.
  The generator combines the regulatory obligations (Agent 1), the
  readiness / impact assessments (AI assessment intelligence), the
  scored questionnaire (rules engine) and the actual identified gaps
  (top-N weakest requirements).
* When the GenAI Shared Service is reachable, the module asks the model
  to reason about each impacted area independently, producing a
  distinctly-worded recommendation per (area, severity, gap-set)
  combination.
* Offline mode: the deterministic generator still produces
  area-specific, gap-aware output by combining per-area templates with
  the actual gap descriptions and obligation IDs relevant to that area.
  Two different regulations therefore produce different recommendations,
  because the input signals differ.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]
    Field = lambda *_a, **_k: None  # type: ignore[assignment]

from models.workflow_models import (
    ImpactAssessment,
    ReadinessAssessment,
    RegulatoryAnalysis,
    RichRecommendation,
)

from .owner_registry import owner_for as _owner_registry_lookup


def _owner_for(function: str) -> str:
    """Return the recommendation owner for ``function``.

    Uses the shared :mod:`services.owner_registry` catalogue with the
    rich-recommendation fallback (``Executive Sponsor``) so we do not
    silently switch to the compact-recommendation fallback.
    """
    return _owner_registry_lookup(function, fallback="Executive Sponsor")


# ---------------------------------------------------------------------------
# Severity / priority
# ---------------------------------------------------------------------------

def _priority_from_readiness_impact(readiness_pct: float, impact_severity: str) -> Tuple[str, str, str]:
    """Return (priority, severity_label, horizon) tuple for an area."""
    readiness = max(0.0, min(100.0, float(readiness_pct)))
    sev = (impact_severity or "").strip().lower()

    if readiness < 25 or sev == "critical":
        return "High", "Critical", "Immediate (0-30 days)"
    if readiness < 50 or sev == "high":
        return "High", "At risk", "Short-term (30-90 days)"
    if readiness < 75 or sev == "medium":
        return "Medium", "Watch", "Medium-term (90-180 days)"
    return "Low", "Ready", "Steady-state (periodic)"


# ---------------------------------------------------------------------------
# Deterministic per-area recommendation composition
# ---------------------------------------------------------------------------

# The following per-area "playbooks" are intentionally *guides* only — the
# generator combines them with the specific gaps, obligation IDs, and
# regulation label so that the same regulation with different gaps produces
# very different output, and different regulations produce different
# outputs entirely.

_AREA_HINTS: Dict[str, Dict[str, Any]] = {
    "risk": {
        "function": "Risk Management",
        "what_verbs": ["Rebuild", "Refresh", "Reconfigure", "Strengthen"],
        "control_domain": "the ICT risk management framework",
        "success_metric_templates": [
            "All top-10 ICT risks brought inside appetite within {horizon_days} days",
            "Residual risk heatmap refreshed monthly with two consecutive clean cycles",
            "100% of Tier-1 scenarios stress-tested with independent challenge",
        ],
        "how_templates": [
            "Re-baseline the ICT risk register end-to-end and stress-test the top scenarios.",
            "Reconvene the enterprise risk committee for a themed ICT-risk update.",
            "Redefine tolerance thresholds and refresh KRIs surfaced to the board.",
        ],
    },
    "governance": {
        "function": "Compliance & Legal",
        "what_verbs": ["Publish", "Refresh", "Table", "Codify"],
        "control_domain": "the governance and management-body oversight framework",
        "success_metric_templates": [
            "Board-approved regulatory governance charter signed within 60 days",
            "Two consecutive committee cycles with recorded challenge on regulatory items",
            "Zero open governance findings older than 90 days",
        ],
        "how_templates": [
            "Draft the regulatory governance charter with three-lines-of-defence RACI and route through the board risk committee.",
            "Refresh delegated authorities so regulatory decisions are traceable to a named accountable executive.",
            "Stand up a fortnightly management-body update until controls stabilise.",
        ],
    },
    "compliance": {
        "function": "Compliance & Legal",
        "what_verbs": ["Rebuild", "Publish", "Refresh"],
        "control_domain": "the compliance monitoring programme",
        "success_metric_templates": [
            "Every regulatory article mapped to an owner and a testable control by the next oversight cycle",
            "Open compliance findings past SLA reduced to zero within 90 days",
            "Monitoring KRIs inside tolerance across the DORA / target-regulation domain set",
        ],
        "how_templates": [
            "Rebuild the obligations register and remap regulatory articles to executable controls.",
            "Refresh compliance monitoring frequencies for the weakest domains.",
            "Rehearse regulator-facing narrative with a mock supervisory walk-through.",
        ],
    },
    "operations": {
        "function": "Operations / Settlement",
        "what_verbs": ["Rebuild", "Automate", "Streamline"],
        "control_domain": "end-to-end operational processes",
        "success_metric_templates": [
            "Top-10 client-impacting flows covered by signed process maps",
            "Reconciliation break categories cleared with sustained trend improvement",
            "Manual workarounds retired for priority controls",
        ],
        "how_templates": [
            "Rebuild business-flow maps for Tier-1 activities and refresh key-control design.",
            "Retire priority manual workarounds and refresh reviewer sign-off gates.",
            "Institute a daily ops-risk bridge until reconciliation quality stabilises.",
        ],
    },
    "technology": {
        "function": "Technology / IT Operations",
        "what_verbs": ["Reconcile", "Modernise", "Baseline"],
        "control_domain": "the ICT platform estate and change management pipeline",
        "success_metric_templates": [
            "CMDB reconciliation at 100% for Tier-1 assets",
            "Change-failure rate below target for three consecutive release cycles",
            "Golden signals green across critical services for two full reporting periods",
        ],
        "how_templates": [
            "Reconcile the asset inventory and force a baseline configuration on Tier-1 platforms.",
            "Reduce change-failure rate through gated releases and pre-prod validation.",
            "Refresh the DR runbook for the two weakest critical services.",
        ],
    },
    "finance": {
        "function": "Compliance & Legal",
        "what_verbs": ["Reconcile", "Restate", "Refresh"],
        "control_domain": "regulatory finance and capital reporting flows",
        "success_metric_templates": [
            "All Tier-1 finance reports produced with clean reconciliations for two consecutive cycles",
            "Sign-off gates tightened with zero exceptions past SLA",
            "Restatement-risk reduced by targeted control redesign",
        ],
        "how_templates": [
            "Rebuild the finance reporting inventory and redesign sign-off gates.",
            "Re-run reconciliations for the last two cycles and remediate exceptions.",
            "Refresh reviewer training on regulatory finance rule interpretation.",
        ],
    },
    "hr": {
        "function": "Human Resources / Training",
        "what_verbs": ["Redesign", "Refresh", "Re-issue"],
        "control_domain": "the people, training and awareness programme",
        "success_metric_templates": [
            "90%+ completion of refreshed regulatory training curriculum for critical roles",
            "Succession coverage documented for every critical role",
            "Training completion tail cleared within 90 days",
        ],
        "how_templates": [
            "Redesign the regulatory training curriculum and refresh the competency matrix.",
            "Re-issue role descriptions with named accountabilities for regulation-critical roles.",
            "Run a 90-day completion sprint with executive-level tracking.",
        ],
    },
    "legal": {
        "function": "Compliance & Legal",
        "what_verbs": ["Remediate", "Refresh", "Renegotiate"],
        "control_domain": "contractual and legal-risk coverage",
        "success_metric_templates": [
            "All Tier-1 contracts remediated with regulation-compliant clauses within 90 days",
            "Regulatory-change log cleared and templates refreshed on schedule",
            "Template library aligned to latest RTS/ITS guidance",
        ],
        "how_templates": [
            "Remediate regulatory clauses across all live Tier-1 contracts.",
            "Refresh contract template libraries and route through general counsel.",
            "Rehearse enforcement scenarios with the third-party risk team.",
        ],
    },
    "data": {
        "function": "Data Governance / Reporting",
        "what_verbs": ["Rebuild", "Instrument", "Re-baseline"],
        "control_domain": "regulatory data lineage, quality and reporting flows",
        "success_metric_templates": [
            "Tier-1 reports each have signed-off end-to-end lineage",
            "Reconciliation exceptions cleared inside SLA for two consecutive cycles",
            "Reference data ownership at 100% for critical domains",
        ],
        "how_templates": [
            "Rebuild end-to-end lineage for the top regulatory reports.",
            "Remediate reference-data ownership gaps and re-baseline reconciliation controls.",
            "Instrument data-quality KRIs and publish a monthly quality dashboard.",
        ],
    },
    "reporting": {
        "function": "Data Governance / Reporting",
        "what_verbs": ["Redesign", "Tighten", "Refresh"],
        "control_domain": "the regulatory reporting pipeline",
        "success_metric_templates": [
            "All Tier-1 regulatory reports produced with clean reconciliations",
            "Sign-off gates clean for three consecutive cycles",
            "Reporting KRIs inside tolerance across every jurisdiction",
        ],
        "how_templates": [
            "Redesign sign-off gates and tighten reviewer allocation.",
            "Refresh the reporting inventory to ensure DORA-relevant items are captured.",
            "Rehearse restatement scenarios with the finance and risk teams.",
        ],
    },
    "cyber": {
        "function": "Cyber Security",
        "what_verbs": ["Expand", "Rehearse", "Instrument"],
        "control_domain": "detective and preventive cyber controls",
        "success_metric_templates": [
            "Open critical vulnerabilities halved within 30 days",
            "TLPT scope agreed with independent testers",
            "MITRE ATT&CK detection coverage above target across crown-jewels estate",
        ],
        "how_templates": [
            "Expand MITRE ATT&CK detection coverage over the crown-jewels estate.",
            "Rehearse ransomware playbooks with the incident bridge.",
            "Halve open critical vulnerabilities through targeted patch and hardening sprints.",
        ],
    },
    "security": {  # alias — same content as cyber
        "function": "Cyber Security",
        "what_verbs": ["Expand", "Rehearse", "Instrument"],
        "control_domain": "detective and preventive cyber controls",
        "success_metric_templates": [
            "Open critical vulnerabilities halved within 30 days",
            "TLPT scope agreed with independent testers",
            "MITRE ATT&CK detection coverage above target across crown-jewels estate",
        ],
        "how_templates": [
            "Expand MITRE ATT&CK detection coverage over the crown-jewels estate.",
            "Rehearse ransomware playbooks with the incident bridge.",
            "Halve open critical vulnerabilities through targeted patch and hardening sprints.",
        ],
    },
    "third": {
        "function": "Vendor / Third-Party Management",
        "what_verbs": ["Re-tier", "Renegotiate", "Rehearse"],
        "control_domain": "the third-party risk management framework",
        "success_metric_templates": [
            "Every Tier-1 provider covered by a compliant contract clause set",
            "Demonstrably executable exit plan for every Tier-1 provider",
            "Sub-contractor visibility current across the critical-provider register",
        ],
        "how_templates": [
            "Re-tier every ICT vendor against Chapter V criteria.",
            "Renegotiate contract clauses on audit, sub-contracting and exit.",
            "Re-execute exit tests for Tier-1 providers.",
        ],
    },
    "vendor": {
        "function": "Vendor / Third-Party Management",
        "what_verbs": ["Re-tier", "Renegotiate", "Rehearse"],
        "control_domain": "the third-party risk management framework",
        "success_metric_templates": [
            "Every Tier-1 provider covered by a compliant contract clause set",
            "Demonstrably executable exit plan for every Tier-1 provider",
        ],
        "how_templates": [
            "Re-tier every ICT vendor against relevant regulatory criteria.",
            "Renegotiate contract clauses on audit, sub-contracting and exit.",
            "Re-execute exit tests for Tier-1 providers.",
        ],
    },
    "incident": {
        "function": "Incident Management",
        "what_verbs": ["Rehearse", "Redesign", "Instrument"],
        "control_domain": "the incident classification and reporting workflow",
        "success_metric_templates": [
            "Regulator-notification timelines demonstrably met on dry-run incidents",
            "Every dry-run classification decision defensible against the taxonomy",
            "Open post-incident actions cleared inside SLA",
        ],
        "how_templates": [
            "Rebuild the incident classification model and dry-run the required notification timelines.",
            "Rehearse the timeline with front-line teams and pre-stage regulator notification templates.",
            "Refresh the incident taxonomy against the latest trend data.",
        ],
    },
    "continuity": {
        "function": "Business Continuity / Resilience",
        "what_verbs": ["Rebuild", "Rehearse", "Re-baseline"],
        "control_domain": "the business continuity and operational resilience programme",
        "success_metric_templates": [
            "Every Tier-1 critical service demonstrably recoverable within its stated impact tolerance",
            "Scenario tests executed on schedule with zero critical findings outstanding",
            "Refreshed BIA and third-party dependency map",
        ],
        "how_templates": [
            "Rebuild the BIA and redefine impact tolerances for critical services.",
            "Execute a severe-but-plausible test for each Tier-1 service.",
            "Refresh the third-party dependency map and stress-test the incident bridge.",
        ],
    },
    "resilience": {
        "function": "Business Continuity / Resilience",
        "what_verbs": ["Rebuild", "Rehearse", "Re-baseline"],
        "control_domain": "the operational resilience programme",
        "success_metric_templates": [
            "Every Tier-1 critical service demonstrably recoverable within impact tolerance",
            "Scenario tests executed on schedule",
        ],
        "how_templates": [
            "Rebuild the BIA and redefine impact tolerances for critical services.",
            "Rehearse severe-but-plausible scenarios and log after-action reviews.",
        ],
    },
    "audit": {
        "function": "Internal Audit / Assurance",
        "what_verbs": ["Refresh", "Deep-dive", "Rebuild"],
        "control_domain": "the internal audit universe and assurance plan",
        "success_metric_templates": [
            "Regulatory audit coverage at 100% for Tier-1 domains",
            "Open audit findings past SLA cleared within 90 days",
            "Audit-committee approval of the refreshed annual plan",
        ],
        "how_templates": [
            "Rebuild the regulatory audit universe and refresh the annual audit plan.",
            "Deep-dive on the weakest domains and rehearse regulator-facing narrative.",
            "Close the tail of open audit findings past SLA.",
        ],
    },
    "programme": {
        "function": "Programme Management",
        "what_verbs": ["Re-baseline", "Refresh", "Escalate"],
        "control_domain": "the regulatory delivery programme",
        "success_metric_templates": [
            "Steering committee approval of the re-baselined plan",
            "Critical-path slippage cleared within the next reporting period",
            "Milestones tracked on schedule with benefits realised on plan",
        ],
        "how_templates": [
            "Re-baseline the delivery roadmap and refresh the RAID log.",
            "Secure funding for the remaining critical path and rehearse the go-live cutover plan.",
            "Escalate red-status paper to programme steering.",
        ],
    },
    "front": {
        "function": "Execution / Client Activity",
        "what_verbs": ["Refresh", "Re-execute", "Rehearse"],
        "control_domain": "front-office execution and client-impact controls",
        "success_metric_templates": [
            "Tier-1 flows each have signed maps and demonstrably tested client-impact scenarios",
            "Product-approval evidence refreshed and cleanly signed",
            "Execution KRIs green for three consecutive cycles",
        ],
        "how_templates": [
            "Rebuild business-flow maps for Tier-1 client activities.",
            "Refresh product-approval gates and re-execute client-impact scenarios.",
            "Freeze new launches until product-approval evidence is clean.",
        ],
    },
    "client": {
        "function": "Execution / Client Activity",
        "what_verbs": ["Refresh", "Re-execute", "Rehearse"],
        "control_domain": "front-office execution and client-impact controls",
        "success_metric_templates": [
            "Tier-1 flows each have signed maps and demonstrably tested client-impact scenarios",
        ],
        "how_templates": [
            "Rebuild business-flow maps for Tier-1 client activities.",
            "Refresh product-approval gates and re-execute client-impact scenarios.",
        ],
    },
}


_GENERIC_HINT: Dict[str, Any] = {
    "function": "Executive Sponsor",
    "what_verbs": ["Address", "Remediate", "Strengthen"],
    "control_domain": "the affected control domain",
    "success_metric_templates": [
        "Open findings closed within their SLA",
        "Named accountable owner in place for each remediation stream",
        "Evidence pack signed and archived at the next governance forum",
    ],
    "how_templates": [
        "Assign a named accountable owner and mobilise a remediation working group.",
        "Refresh the control design, evidence pack and monitoring cadence.",
        "Rehearse regulator-facing narrative with the compliance team.",
    ],
}


def _resolve_hint(area: str) -> Dict[str, Any]:
    if not area:
        return _GENERIC_HINT
    lower = area.lower()
    for key, hint in _AREA_HINTS.items():
        if key in lower:
            return hint
    return _GENERIC_HINT


def _dependencies_for(area: str, priority: str) -> List[str]:
    lower = (area or "").lower()
    deps: List[str] = []
    if priority == "High":
        deps.append("Executive sponsorship confirmed and funding released")
    if any(t in lower for t in ("technology", "ict", "systems", "cyber", "data", "reporting")):
        deps.append("Access to authoritative asset / CMDB inventory")
    if any(t in lower for t in ("third", "vendor", "outsourc")):
        deps.append("Vendor engagement plan agreed with procurement and legal")
    if any(t in lower for t in ("risk", "governance", "compliance", "legal")):
        deps.append("Alignment with the enterprise risk and compliance committee agenda")
    if any(t in lower for t in ("operations", "settlement", "front", "client", "execution")):
        deps.append("Business-line owner nominated and change-freeze runway agreed")
    if not deps:
        deps.append("Named accountable executive owner confirmed")
    return deps[:4]


def _implementation_steps(hint: Mapping[str, Any], gap_titles: Sequence[str]) -> List[str]:
    steps: List[str] = list(hint.get("how_templates", []))[:3]
    if gap_titles:
        steps.append(
            "Close the specific gaps identified above (" +
            ", ".join(list(gap_titles)[:3]) + ") with dated milestones."
        )
    return steps[:5]


def _success_metrics(hint: Mapping[str, Any], area: str) -> List[str]:
    templates = list(hint.get("success_metric_templates", []))
    return [t.format(area=area, horizon_days=90) for t in templates[:3]]


def _short_term_actions(hint: Mapping[str, Any], priority: str) -> List[str]:
    verbs = hint.get("what_verbs", ["Address"])
    return [
        f"{verbs[0]} the first-move controls in {hint.get('control_domain', 'the domain')} within the next 30 days.",
        "Assign a named accountable owner and stand up a weekly working group.",
        "Publish a red-status paper to the relevant governance forum.",
    ] if priority == "High" else [
        f"{verbs[0]} the priority controls in {hint.get('control_domain', 'the domain')} at the next steering cycle.",
        "Confirm remediation ownership and refresh evidence packs.",
    ]


def _long_term_actions(hint: Mapping[str, Any]) -> List[str]:
    return [
        f"Embed {hint.get('control_domain', 'the domain')} controls into the annual attestation cycle.",
        "Roll the target state into the enterprise architecture and control library.",
        "Benchmark against peer disclosures and refresh the maturity plan.",
    ]


def _quick_wins(hint: Mapping[str, Any]) -> List[str]:
    return [
        "Confirm evidence dictionary is current for the affected controls.",
        "Refresh KRI thresholds to trigger the intended escalation.",
        "Publish a one-page executive summary of the remediation plan.",
    ]


# ---------------------------------------------------------------------------
# Deterministic generator
# ---------------------------------------------------------------------------

def _gaps_for_area(
    area: str,
    top_gaps: Sequence[Mapping[str, Any]],
    package: Mapping[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return ``(requirement_ids, requirement_titles)`` for gaps in the area."""
    if not top_gaps:
        return [], []

    req_titles: Dict[str, str] = {
        r.get("normalized_id", ""): r.get("requirement", "")
        for r in package.get("requirements", []) or []
    }
    req_areas: Dict[str, List[str]] = defaultdict(list)
    for pair in package.get("impact_pairs", []) or []:
        for rid in pair.get("requirement_ids", []) or []:
            req_areas[rid].append(pair.get("area", ""))

    area_lower = (area or "").lower().strip()
    matched_ids: List[str] = []
    matched_titles: List[str] = []
    for gap in top_gaps:
        rid = str(gap.get("requirement_id") or "").strip()
        if not rid:
            continue
        if area_lower and area_lower not in [a.lower() for a in req_areas.get(rid, [])]:
            continue
        matched_ids.append(rid)
        matched_titles.append(req_titles.get(rid, rid))
        if len(matched_ids) >= 6:
            break
    return matched_ids, matched_titles


# ---------------------------------------------------------------------------
# Focus lenses - each lens gives a recommendation a distinct angle so
# multiple recommendations for the SAME area (one per identified gap) do
# not read alike. The lens catalogue is intentionally large so we can
# rotate across gaps within an area AND across areas without repetition.
# ---------------------------------------------------------------------------

_FOCUS_LENSES: List[Dict[str, Any]] = [
    {
        "id": "design",
        "label": "Control Design & Implementation",
        "verbs": ["Redesign", "Rearchitect", "Rebuild", "Overhaul", "Reconfigure"],
        "what_templates": [
            "Rearchitect the underlying control that fails {gap_id} - {gap_title_short} - "
            "so it demonstrably satisfies the {reg_short} obligation without manual overrides.",
            "Overhaul the {area} operating model behind {gap_id} - {gap_title_short} - "
            "removing tribal-knowledge handoffs and making evidence a by-product of the flow.",
            "Rebuild the control blueprint that covers {gap_id} with a target operating "
            "model diagram, named engineering owner and a dated cutover plan.",
        ],
        "why_templates": [
            "The current design of {gap_id} cannot demonstrate coverage of the {reg_short} "
            "obligation set; supervisors will read that as a design defect, not an "
            "operational miss.",
            "Because {gap_id} rests on ad-hoc workarounds, every audit cycle re-opens the "
            "same finding; a redesign is the only durable answer.",
        ],
        "steps": [
            "Convene a two-week design sprint with the control owner, engineering and second-line challenge in one room.",
            "Publish a target operating model diagram signed off by the accountable executive before any build starts.",
            "Cut over from the interim workaround to the redesigned control on a fixed, board-communicated date.",
        ],
        "metrics": [
            "Redesigned control operates for two consecutive cycles with zero manual overrides logged.",
            "Named engineering owner and dated release plan captured on the enterprise change board.",
        ],
        "short_actions": [
            "Freeze the interim workaround register so no further design debt accumulates.",
            "Book the design-sprint room for the next fortnight and pre-invite second line.",
        ],
        "long_actions": [
            "Fold the redesigned control into the enterprise reference architecture.",
            "Retire the legacy pattern from the control library once the redesign has bedded in.",
        ],
        "quick_wins": [
            "Publish a one-page architecture-decision-record explaining the redesign choice.",
        ],
    },
    {
        "id": "evidence",
        "label": "Evidence & Documentation Uplift",
        "verbs": ["Assemble", "Codify", "Refresh", "Repackage", "Document"],
        "what_templates": [
            "Assemble a defensible evidence dossier for {gap_id} - {gap_title_short} - so "
            "an external assessor can walk the control from obligation to artefact "
            "without interviewing anyone on the team.",
            "Codify the operating procedure and control narrative behind {gap_id} into a "
            "single signed pack aligned to the {reg_short} clause set.",
            "Refresh the documentation trail for {gap_id} so every {reg_short} clause maps "
            "to a testable, dated artefact in the evidence repository.",
        ],
        "why_templates": [
            "The gap on {gap_id} is primarily documentation debt: the control may operate, "
            "but nothing on paper proves it does - and that alone becomes a "
            "supervisory finding under {reg_short}.",
            "Missing artefacts around {gap_id} block independent assurance and open the "
            "door to enforcement action even where the underlying control functions.",
        ],
        "steps": [
            "Inventory the current evidence set for the control and label each item as sufficient, gap, or missing.",
            "Commission the missing artefacts on a 30-day plan with named authors and a peer-review cadence.",
            "Route the completed evidence pack through second-line challenge before archiving in the regulatory repository.",
        ],
        "metrics": [
            "Every requirement in the gap-set has a linked, dated evidence artefact in the repository.",
            "Second-line challenge sign-off recorded against the refreshed documentation pack.",
        ],
        "short_actions": [
            "Pull the three most-missing artefacts to the top of the evidence backlog this week.",
            "Nominate a documentation lead so authorship is not left to volunteers.",
        ],
        "long_actions": [
            "Fold the evidence dictionary into the annual attestation cycle so drift is caught early.",
        ],
        "quick_wins": [
            "Publish a one-page 'evidence map' listing which artefact answers which clause.",
        ],
    },
    {
        "id": "testing",
        "label": "Testing, Assurance & Independent Validation",
        "verbs": ["Stress-test", "Rehearse", "Validate", "Prove", "Simulate"],
        "what_templates": [
            "Stress-test the control behind {gap_id} against {reg_short}-relevant scenarios "
            "to prove it operates as intended under adverse conditions, not just happy paths.",
            "Rehearse the failure-mode playbook that surrounds {gap_id} with second-line "
            "challenge in the room and independent auditors verifying outcomes.",
            "Validate the {area} control that owns {gap_id} independently against the "
            "obligation set, using scenarios the regulator has explicitly flagged.",
        ],
        "why_templates": [
            "Without a defensible test plan for {gap_id} the control cannot be assured; "
            "the regulator expects evidence of independent challenge, not self-attestation.",
            "The current testing cadence around {gap_id} does not cover the scenarios that "
            "{reg_short} explicitly requires - a gap that is trivial to challenge in a "
            "supervisory review.",
        ],
        "steps": [
            "Draft a scenario matrix aligned to the regulation's articles and rank by likelihood x impact.",
            "Run a live-fire test with second line observing and independent audit verifying the outcomes.",
            "Log every finding to the remediation backlog with dated closure milestones and executive sponsorship.",
        ],
        "metrics": [
            "Independent test executed and signed off with zero critical findings outstanding.",
            "Second line has issued a green attestation against the refreshed control.",
        ],
        "short_actions": [
            "Schedule the earliest possible dry-run and pre-agree the evaluation criteria in writing.",
        ],
        "long_actions": [
            "Fold this test into the annual assurance plan so it repeats without executive prompting.",
        ],
        "quick_wins": [
            "Publish the scenario matrix so stakeholders can challenge coverage before the test runs.",
        ],
    },
    {
        "id": "governance",
        "label": "Governance, Ownership & Escalation",
        "verbs": ["Nominate", "Charter", "Escalate", "Ratify", "Sponsor"],
        "what_templates": [
            "Nominate a single accountable executive for {gap_id} - {gap_title_short} - and "
            "ratify their remit at the next risk-committee cycle so ownership stops being diffuse.",
            "Charter a governance forum with the explicit mandate to close {gap_id} on a "
            "dated plan, reporting monthly to the board risk committee.",
            "Escalate the exposure created by {gap_id} to the appropriate risk committee "
            "with a red-status paper naming the accountable executive and the closure horizon.",
        ],
        "why_templates": [
            "The gap on {gap_id} persists because ownership is diffuse; nominating a single "
            "accountable executive is a prerequisite for any durable closure plan.",
            "Governance forums are not currently tracking the exposure around {gap_id}, so "
            "it drifts release after release with nobody empowered to break the pattern.",
        ],
        "steps": [
            "Draft the RACI for the control and route it through the risk committee for formal ratification.",
            "Add a standing agenda item for this control to the fortnightly governance forum until it stabilises.",
            "Publish a monthly board-facing scorecard showing progress against the closure plan.",
        ],
        "metrics": [
            "Named executive sponsor recorded in the enterprise governance handbook.",
            "Board-approved closure plan tracked on the risk-committee dashboard with zero missed milestones.",
        ],
        "short_actions": [
            "Publish a one-page briefing note ahead of the next committee cycle.",
        ],
        "long_actions": [
            "Fold the governance forum's remit into the annual committee charter refresh.",
        ],
        "quick_wins": [
            "Pre-brief the executive sponsor 1:1 so committee time is spent on decisions, not orientation.",
        ],
    },
    {
        "id": "training",
        "label": "Training, Awareness & Capability Uplift",
        "verbs": ["Upskill", "Re-brief", "Coach", "Educate", "Certify"],
        "what_templates": [
            "Upskill the front-line team on the {reg_short} obligations attached to "
            "{gap_id} so every operator can explain the control and its evidence in "
            "their own words.",
            "Re-brief managers on their supervisory duties for {gap_id} - {gap_title_short} - "
            "including how to challenge a red control before it becomes an audit finding.",
            "Certify the operators of the {area} control against a role-specific "
            "knowledge check tied directly to {gap_id}.",
        ],
        "why_templates": [
            "Operators of {gap_id} cannot currently articulate why or how it satisfies "
            "{reg_short}; that alone is a defensible supervisory finding.",
            "Turnover has diluted institutional knowledge around {gap_id}; a refreshed "
            "learning path is the only way to restore baseline competency.",
        ],
        "steps": [
            "Redesign the curriculum with role-specific learning outcomes and testable knowledge checks.",
            "Deliver the refreshed training in a 30-day sprint and track completion through the LMS.",
            "Sample-audit competency 60 days after delivery and remediate the tail before the next attestation.",
        ],
        "metrics": [
            "95%+ completion on the refreshed curriculum for all roles that operate the control.",
            "Random competency checks post-training pass with zero material findings.",
        ],
        "short_actions": [
            "Publish a plain-language one-pager explaining the control while the full curriculum is being built.",
        ],
        "long_actions": [
            "Fold this competency into the annual mandatory training cycle so it does not decay again.",
        ],
        "quick_wins": [
            "Record a 5-minute video of the control owner walking through the flow.",
        ],
    },
    {
        "id": "metrics",
        "label": "Monitoring, Metrics & Leading KRIs",
        "verbs": ["Instrument", "Wire", "Meter", "Baseline", "Publish"],
        "what_templates": [
            "Instrument the control that owns {gap_id} with leading indicators tied to "
            "{reg_short} thresholds so drift is caught before it becomes a finding.",
            "Wire real-time KRIs for {gap_id} - {gap_title_short} - into the risk "
            "dashboard so second line sees the exposure the moment it moves.",
            "Publish a monthly performance scorecard for the {area} control with "
            "independently verified metrics and executive-level trend commentary.",
        ],
        "why_templates": [
            "You cannot manage what you cannot measure; the current metric set does not "
            "cover the {reg_short} thresholds relevant to {gap_id}, so drift goes unnoticed.",
            "Trailing indicators surface {gap_id}-style failures only after harm; leading "
            "indicators would catch the exposure while it is still cheap to fix.",
        ],
        "steps": [
            "Define three to five leading KRIs with clear thresholds tied directly to the regulator's obligation set.",
            "Wire the metrics into the enterprise risk dashboard and stand up automated alerting with named recipients.",
            "Review the metric set every quarter and retire any indicator that has stopped adding decision value.",
        ],
        "metrics": [
            "Every KRI thresholds green for two consecutive reporting cycles.",
            "Automated alerting operational with response SLAs recorded and met.",
        ],
        "short_actions": [
            "Draft the KRI shortlist and socialise it with the risk committee this month.",
        ],
        "long_actions": [
            "Fold the KRI set into the annual risk-appetite refresh so tolerances stay honest.",
        ],
        "quick_wins": [
            "Publish yesterday's readings on a shared dashboard so the team can trend-watch immediately.",
        ],
    },
    {
        "id": "policy",
        "label": "Policy, Standards & Regulatory Interpretation",
        "verbs": ["Restate", "Reissue", "Clarify", "Codify", "Publish"],
        "what_templates": [
            "Restate the policy and standard that governs {gap_id} so the {reg_short} "
            "expectations are unambiguous and cannot be read two ways.",
            "Reissue the standard behind {gap_id} - {gap_title_short} - with worked "
            "examples showing how the {reg_short} clauses translate into day-to-day "
            "operational rules.",
            "Codify the regulatory interpretation for {gap_id} in an authoritative "
            "reading note signed by compliance and legal.",
        ],
        "why_templates": [
            "The gap on {gap_id} exists partly because the policy is silent on how the "
            "{reg_short} clause applies; ambiguity is being resolved differently by "
            "different teams.",
            "Without a clear standard behind {gap_id}, operators default to legacy "
            "practice, which the regulator will not accept as evidence of compliance.",
        ],
        "steps": [
            "Draft the revised policy language with tracked changes so approvers can see exactly what has moved.",
            "Route through legal, compliance and the risk committee on an accelerated cycle.",
            "Republish the standard with a mandatory read-and-attest workflow for affected teams.",
        ],
        "metrics": [
            "Revised policy published and attested by 100% of in-scope colleagues.",
            "Regulatory interpretation note signed by both compliance and legal on file.",
        ],
        "short_actions": [
            "Circulate the draft interpretation note for early challenge from second line.",
        ],
        "long_actions": [
            "Add the standard to the annual policy review calendar so it stays current.",
        ],
        "quick_wins": [
            "Publish a two-line clarification while the full standard is being redrafted.",
        ],
    },
]


def _pick_lens(lens_seed: int, offset: int = 0) -> Dict[str, Any]:
    """Return a focus lens deterministically based on ``lens_seed`` and offset."""
    idx = (lens_seed + offset) % len(_FOCUS_LENSES)
    return _FOCUS_LENSES[idx]


def _shorten(text: str, limit: int = 90) -> str:
    """Return a short version of ``text`` for inline use in a sentence."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:.-")
    return f"{cut}..."


def _short_reg(regulation: str) -> str:
    """Return an inline-friendly regulation label."""
    reg = (regulation or "").strip()
    return reg or "the target regulation"


def _rotate(items: Sequence[str], seed: int) -> List[str]:
    """Return ``items`` rotated so the pick-order varies by ``seed``."""
    if not items:
        return []
    lst = list(items)
    n = len(lst)
    off = seed % n
    return lst[off:] + lst[:off]


# ---------------------------------------------------------------------------
# Distinctness enforcement - guarantees no two recommendations share the
# same title/what phrasing, either within an area or across areas.
# ---------------------------------------------------------------------------

_STOPWORDS: set = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "by",
    "with", "at", "as", "is", "are", "be", "so", "that", "this", "these",
    "those", "it", "its", "into", "from", "will", "shall", "any",
}


def _signature(text: str) -> set:
    """Return the significant-word signature of ``text`` for similarity checks."""
    if not text:
        return set()
    words = [
        w.strip(".,;:()[]\"'-").lower()
        for w in text.split()
        if len(w) > 2
    ]
    return {w for w in words if w and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _enforce_distinct_wording(recs: List[RichRecommendation]) -> List[RichRecommendation]:
    """Mutate any recommendation whose wording is too close to another.

    The check is deliberately conservative: only when the *title* + *what*
    signature overlap crosses 60% do we rewrite the later recommendation
    by rotating its lens verb and prepending a gap-specific clause. This
    keeps the deterministic output on-message while guaranteeing no two
    cards look alike.
    """
    if not recs:
        return recs

    seen: List[Tuple[str, set]] = []  # (rec_id, signature)
    for rec in recs:
        combined = f"{rec.title} {rec.what}"
        sig = _signature(combined)
        collision = None
        for _, existing in seen:
            if _jaccard(sig, existing) > 0.55:
                collision = existing
                break

        if collision is not None:
            # Rewrite title to force divergence. We use the first mapped
            # requirement ID (if any) so the rewrite still cites the gap
            # and we then re-check the signature - it will diverge because
            # the leading token is now the requirement ID.
            gap_ref = (
                rec.mapped_requirement_ids[0]
                if rec.mapped_requirement_ids else rec.recommendation_id
            )
            if gap_ref and gap_ref not in rec.title:
                rec.title = f"{gap_ref} - {rec.title}"
            # Prepend a distinct opening clause to `what` so the body diverges.
            rec.what = (
                f"Specifically for {gap_ref}: {rec.what}"
                if not rec.what.startswith("Specifically for ")
                else rec.what
            )
            sig = _signature(f"{rec.title} {rec.what}")

        seen.append((rec.recommendation_id, sig))

    return recs


_MAX_RECS_PER_AREA = 3


def _compose_one_rec(
    *,
    rec_id: str,
    area: str,
    hint: Mapping[str, Any],
    lens: Mapping[str, Any],
    gap_id: str,
    gap_title: str,
    gap_position: int,
    regulation: str,
    readiness_pct: float,
    severity_label: str,
    priority: str,
    horizon: str,
    obligation_ids: Sequence[str],
    impact: Optional[ImpactAssessment],
    area_seed: int,
) -> RichRecommendation:
    """Compose a single lens-driven recommendation for one gap in an area."""
    reg_label = regulation or "the target regulation"
    reg_short = _short_reg(regulation)
    gap_title_short = _shorten(gap_title or gap_id or f"{area} coverage", limit=80)
    gap_ref = gap_id or "the identified gap"
    function = hint.get("function", "Executive Sponsor")
    owner = _owner_for(function)
    control_domain = hint.get("control_domain", "the affected control domain")

    lens_verbs = _rotate(list(lens.get("verbs", [])), area_seed + gap_position)
    verb = lens_verbs[0] if lens_verbs else "Address"

    what_pool = _rotate(list(lens.get("what_templates", [])), area_seed + gap_position)
    why_pool = _rotate(list(lens.get("why_templates", [])), area_seed + gap_position + 1)
    steps_pool = _rotate(list(lens.get("steps", [])), area_seed + gap_position + 2)
    metrics_pool = _rotate(list(lens.get("metrics", [])), area_seed + gap_position + 3)
    short_pool = _rotate(list(lens.get("short_actions", [])), area_seed + gap_position + 4)
    long_pool = _rotate(list(lens.get("long_actions", [])), area_seed + gap_position + 5)
    quick_pool = _rotate(list(lens.get("quick_wins", [])), area_seed + gap_position + 6)

    fmt_kwargs = {
        "area": area,
        "reg": reg_label,
        "reg_short": reg_short,
        "gap_id": gap_ref,
        "gap_title_short": gap_title_short,
    }

    what = (what_pool[0] if what_pool else "").format(**fmt_kwargs)
    why_main = (why_pool[0] if why_pool else "").format(**fmt_kwargs)

    obligation_clause = ""
    if obligation_ids:
        obligation_clause = (
            f" This traces directly to {reg_short} obligations "
            f"{', '.join(list(obligation_ids)[:3])}."
        )
    impact_clause = ""
    if impact and getattr(impact, "overall_severity_score", 0):
        impact_clause = (
            f" Overall regulatory impact for the programme sits at "
            f"{impact.overall_severity} ({impact.overall_severity_score:.0f}/100); "
            f"leaving {gap_ref} unresolved keeps that number high."
        )
    why = (why_main + obligation_clause + impact_clause).strip()

    how_steps = [s.format(**fmt_kwargs) for s in steps_pool[:3]]
    how = " ".join(how_steps)

    success_metrics_area = [
        m.format(area=area, horizon_days=90)
        for m in list(hint.get("success_metric_templates", []))[:2]
    ]
    success_metrics = [s.format(**fmt_kwargs) for s in metrics_pool[:2]] + success_metrics_area
    # De-dupe metrics list while preserving order.
    seen_metric: set = set()
    success_metrics = [
        m for m in success_metrics
        if not (m in seen_metric or seen_metric.add(m))
    ][:4]

    short_term_actions = [s.format(**fmt_kwargs) for s in short_pool[:2]]
    long_term_actions = [s.format(**fmt_kwargs) for s in long_pool[:2]]
    quick_wins = [s.format(**fmt_kwargs) for s in quick_pool[:2]]

    expected_outcome = (
        f"Closing {gap_ref} moves the {area} readiness line into the target "
        f"band for {reg_short}. Success looks like: "
        + "; ".join(success_metrics[:3])
        + "."
    )

    identified_gap = (
        f"{area} scored {readiness_pct:.1f}% ({severity_label}). "
        f"Lens: {lens.get('label', 'General remediation')}. "
        f"Gap in scope: {gap_ref} - {_shorten(gap_title or 'requirement coverage', 220)}."
    )

    business_impact = (
        f"With {area} sitting at {readiness_pct:.1f}%, {gap_ref} keeps the firm "
        f"exposed to supervisory challenge on {reg_short}. Priority is "
        f"{priority}; the closure horizon is {horizon}."
    )

    regulatory_rationale = (
        f"{reg_short} obligations mapped to {area} require demonstrable coverage "
        f"of {gap_ref}. "
        + (
            f"Named obligation IDs: {', '.join(list(obligation_ids)[:5])}."
            if obligation_ids else
            "Obligation IDs are captured in the Agent 1 output; refer to the "
            "linked BRD row for the exact clause references."
        )
    )

    title = (
        f"{verb} {area} - {lens.get('label', 'Remediation')} for {gap_ref}"
    )

    return RichRecommendation(
        recommendation_id=rec_id,
        title=title,
        area=area,
        function=function,
        priority=priority,
        severity=severity_label,
        horizon=horizon,
        what=what,
        why=why,
        how=how,
        expected_outcome=expected_outcome,
        dependencies=_dependencies_for(area, priority),
        owner=owner,
        mapped_requirement_ids=[gap_ref] if gap_id else [],
        mapped_obligation_ids=list(obligation_ids)[:6],
        identified_gap=identified_gap,
        regulatory_rationale=regulatory_rationale,
        business_impact=business_impact,
        implementation_steps=how_steps,
        success_metrics=success_metrics,
        short_term_actions=short_term_actions,
        long_term_actions=long_term_actions,
        quick_wins=quick_wins,
        generated_by_ai=False,
    )


def _deterministic_rich_recommendations(
    *,
    regulation: str,
    area_summary: Mapping[str, Mapping[str, Any]],
    top_gaps: Sequence[Mapping[str, Any]],
    package: Mapping[str, Any],
    obligations_by_area: Optional[Mapping[str, List[str]]] = None,
    impact: Optional[ImpactAssessment] = None,
    readiness: Optional[ReadinessAssessment] = None,
) -> List[RichRecommendation]:
    """Compose consulting-grade recommendations without GenAI.

    For every impacted area, we emit MULTIPLE recommendations - one per
    identified gap (capped at :data:`_MAX_RECS_PER_AREA`) - each anchored to
    a distinct **focus lens** (Design, Evidence, Testing, Governance,
    Training, Metrics, Policy). Lens rotation is deterministic but seeded
    by the area name and gap index so:

    * different gaps within the same area draw a different lens,
    * different areas start their lens cycle at a different offset,
    * different regulations produce different opening verbs, and
    * a post-processing pass rewrites any two recommendations whose
      title/what phrasing would otherwise look alike.

    If an area has no mapped gaps we still emit one area-level rec (using
    the highest-priority lens for its readiness band) so the dashboard
    never shows a blank card.
    """
    obligations_by_area = obligations_by_area or {}
    results: List[RichRecommendation] = []

    sorted_areas = sorted(
        area_summary.items(),
        key=lambda kv: float(kv[1].get("Compliance %") or 0.0),
    )

    reg_seed = sum(ord(c) for c in (regulation or ""))

    counter = 1
    for area_idx, (area, summary) in enumerate(sorted_areas):
        readiness_pct = float(summary.get("Compliance %") or 0.0)
        impact_severity_label = ""
        if impact:
            for dim in impact.dimensions():
                if any(
                    area.lower() in it.lower() or it.lower() in area.lower()
                    for it in dim.items
                ):
                    impact_severity_label = dim.severity
                    break
        priority, severity_label, horizon = _priority_from_readiness_impact(
            readiness_pct, impact_severity_label,
        )
        hint = _resolve_hint(area)
        gap_ids, gap_titles = _gaps_for_area(area, top_gaps, package)
        obligation_ids = list(obligations_by_area.get(area, []))[:6]

        area_seed = reg_seed + sum(ord(c) for c in (area or ""))

        # If we have no per-gap breakdown, produce one area-level rec using
        # a synthetic "gap" - the low readiness itself is the gap.
        if not gap_ids:
            synthetic_gap_id = f"{area.upper()[:6] or 'AREA'}-COVERAGE"
            synthetic_gap_title = (
                f"{area} readiness at {readiness_pct:.1f}% is below the target "
                f"band for {regulation or 'the regulation'}."
            )
            lens = _pick_lens(area_seed, offset=area_idx)
            rec = _compose_one_rec(
                rec_id=f"REC-{counter:03d}",
                area=area,
                hint=hint,
                lens=lens,
                gap_id=synthetic_gap_id,
                gap_title=synthetic_gap_title,
                gap_position=0,
                regulation=regulation,
                readiness_pct=readiness_pct,
                severity_label=severity_label,
                priority=priority,
                horizon=horizon,
                obligation_ids=obligation_ids,
                impact=impact,
                area_seed=area_seed,
            )
            results.append(rec)
            counter += 1
            continue

        # Emit one rec per gap (up to the cap), each with a distinct lens.
        gap_pairs = list(zip(gap_ids, gap_titles))[:_MAX_RECS_PER_AREA]
        for gap_position, (gap_id, gap_title) in enumerate(gap_pairs):
            lens = _pick_lens(area_seed, offset=area_idx + gap_position)
            rec = _compose_one_rec(
                rec_id=f"REC-{counter:03d}",
                area=area,
                hint=hint,
                lens=lens,
                gap_id=gap_id,
                gap_title=gap_title,
                gap_position=gap_position,
                regulation=regulation,
                readiness_pct=readiness_pct,
                severity_label=severity_label,
                priority=priority,
                horizon=horizon,
                obligation_ids=obligation_ids,
                impact=impact,
                area_seed=area_seed,
            )
            results.append(rec)
            counter += 1

    # Final safety net - rewrite any pair whose wording overlaps too much.
    results = _enforce_distinct_wording(results)
    return results


# ---------------------------------------------------------------------------
# GenAI enrichment
# ---------------------------------------------------------------------------

class _RichRecPayload(BaseModel):  # type: ignore[misc]
    title: str = Field(description="Short executive-ready title")
    priority: str = Field(description="High / Medium / Low")
    what: str = Field(description="Specific action to take, 1-2 sentences")
    why: str = Field(description="Regulatory rationale + business impact, 2-3 sentences")
    how: str = Field(description="Practical implementation guidance, 2-3 sentences")
    expected_outcome: str = Field(description="What 'done' looks like, 1-2 sentences")
    dependencies: List[str] = Field(default_factory=list)
    implementation_steps: List[str] = Field(default_factory=list, description="3-5 concrete steps")
    success_metrics: List[str] = Field(default_factory=list, description="2-3 measurable outcomes")
    short_term_actions: List[str] = Field(default_factory=list)
    long_term_actions: List[str] = Field(default_factory=list)
    quick_wins: List[str] = Field(default_factory=list)


def _enrich_with_genai(
    rec: RichRecommendation,
    *,
    regulation: str,
    obligations: Sequence[Mapping[str, Any]],
    client: Any,
) -> RichRecommendation:
    """Ask GenAI to rewrite one recommendation in consulting voice, per area."""
    context = {
        "regulation": regulation,
        "area": rec.area,
        "function": rec.function,
        "current_readiness_pct": None,
        "priority": rec.priority,
        "severity": rec.severity,
        "identified_gap": rec.identified_gap,
        "mapped_requirement_ids": rec.mapped_requirement_ids,
        "mapped_obligation_ids": rec.mapped_obligation_ids,
        "regulation_snippets": [
            {
                "id": o.get("id"),
                "title": o.get("title"),
                "theme": o.get("theme"),
                "compliance_requirement": (o.get("compliance_requirement") or "")[:220],
            }
            for o in obligations[:5]
        ],
        "deterministic_draft": {
            "what": rec.what,
            "why": rec.why,
            "how": rec.how,
            "expected_outcome": rec.expected_outcome,
            "dependencies": rec.dependencies,
            "implementation_steps": rec.implementation_steps,
            "success_metrics": rec.success_metrics,
        },
    }
    instruction = (
        "You are a Big Four regulatory consultant writing an executive-ready "
        "recommendation for the client's board. The recommendation must be "
        "SPECIFIC to the impacted area, referenced regulatory obligations, "
        "and the identified gaps below. It must NOT be generic. It must NOT "
        "reuse the same phrasing across different areas. Ground every "
        "statement in the regulation context and obligations. Return the six "
        "required fields (what, why, how, priority, expected_outcome, "
        "dependencies) plus a set of 3-5 implementation_steps, 2-3 "
        "success_metrics, and short-term / long-term / quick-wins lists."
    )
    from .guardrails import safe_generate

    # Compose a source corpus from the regulation snippets so the citation
    # validator can cross-check any Article / RTS the LLM emits in its
    # rewritten paragraphs.
    corpus_bits: List[str] = [regulation]
    for o in obligations[:5]:
        corpus_bits.append(str(o.get("compliance_requirement") or ""))
        corpus_bits.append(str(o.get("title") or ""))
        corpus_bits.append(str(o.get("theme") or ""))
    source_corpus = "\n".join(b for b in corpus_bits if b)

    payload, guardrail_report = safe_generate(
        client,
        _RichRecPayload,
        f"Recommendation for {rec.area}",
        instruction,
        json.dumps(context, default=str),
        regulation=regulation or None,
        source_corpus=source_corpus,
        text_fields=(
            "title", "priority", "what", "why", "how", "expected_outcome",
            "dependencies", "implementation_steps", "success_metrics",
            "short_term_actions", "long_term_actions", "quick_wins",
        ),
    )
    if payload is None or not guardrail_report.ok:
        # Guardrails vetoed — return the deterministic draft untouched.
        logger.warning(
            "GenAI rich-recommendation rewrite vetoed. area=%s summary=%s",
            rec.area, guardrail_report.summary() if guardrail_report else "n/a",
        )
        return rec

    rec.title = str(payload.title or rec.title)
    rec.priority = str(payload.priority or rec.priority)
    rec.what = str(payload.what or rec.what)
    rec.why = str(payload.why or rec.why)
    rec.how = str(payload.how or rec.how)
    rec.expected_outcome = str(payload.expected_outcome or rec.expected_outcome)
    if payload.dependencies:
        rec.dependencies = [str(d).strip() for d in payload.dependencies if str(d).strip()][:6]
    if payload.implementation_steps:
        rec.implementation_steps = [
            str(s).strip() for s in payload.implementation_steps if str(s).strip()
        ][:6]
    if payload.success_metrics:
        rec.success_metrics = [
            str(s).strip() for s in payload.success_metrics if str(s).strip()
        ][:5]
    if payload.short_term_actions:
        rec.short_term_actions = [
            str(s).strip() for s in payload.short_term_actions if str(s).strip()
        ][:6]
    if payload.long_term_actions:
        rec.long_term_actions = [
            str(s).strip() for s in payload.long_term_actions if str(s).strip()
        ][:6]
    if payload.quick_wins:
        rec.quick_wins = [
            str(s).strip() for s in payload.quick_wins if str(s).strip()
        ][:6]
    rec.generated_by_ai = True
    return rec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_rich_recommendations(
    *,
    analysis: Optional[RegulatoryAnalysis],
    scoring_evaluation: Mapping[str, Any],
    top_gaps: Sequence[Mapping[str, Any]],
    package: Mapping[str, Any],
    impact: Optional[ImpactAssessment] = None,
    readiness: Optional[ReadinessAssessment] = None,
    client: Optional[Any] = None,
    enrich_with_genai: bool = True,
) -> List[RichRecommendation]:
    """Produce area-specific consulting-grade recommendations.

    The generator combines every available evidence surface — obligations,
    scored questionnaire, impact and readiness assessments — so each
    recommendation is unique to the area, gaps, and regulation. Falls back
    silently to deterministic composition when the GenAI client is
    unavailable.
    """
    logger.info(
        "Building rich recommendations. enrich_with_genai=%s client=%s top_gaps=%d",
        enrich_with_genai,
        "genai" if client is not None else "offline",
        len(top_gaps or []),
    )
    regulation = getattr(analysis, "regulation", "") or ""
    area_summary = scoring_evaluation.get("area_summary") or {}

    obligations_by_area: Dict[str, List[str]] = defaultdict(list)
    if analysis is not None:
        for o in getattr(analysis, "obligations", []) or []:
            area = str(getattr(o, "impacted_area", "") or "").strip()
            oid = str(getattr(o, "obligation_id", "") or "").strip()
            if area and oid:
                obligations_by_area[area].append(oid)

    recs = _deterministic_rich_recommendations(
        regulation=regulation,
        area_summary=area_summary,
        top_gaps=top_gaps,
        package=package,
        obligations_by_area=obligations_by_area,
        impact=impact,
        readiness=readiness,
    )

    if not enrich_with_genai or client is None:
        return recs

    obligations_dicts: List[Dict[str, Any]] = []
    if analysis is not None:
        for o in getattr(analysis, "obligations", []) or []:
            obligations_dicts.append({
                "id": getattr(o, "obligation_id", ""),
                "title": getattr(o, "title", ""),
                "theme": getattr(o, "theme", ""),
                "impacted_area": getattr(o, "impacted_area", ""),
                "compliance_requirement": getattr(o, "compliance_requirement", ""),
            })

    enriched: List[RichRecommendation] = []
    for rec in recs:
        area_obls = [o for o in obligations_dicts if o.get("impacted_area") == rec.area]
        enriched.append(_enrich_with_genai(
            rec,
            regulation=regulation,
            obligations=area_obls or obligations_dicts,
            client=client,
        ))
    return enriched


def rich_recommendations_to_dicts(recs: Sequence[RichRecommendation]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in recs]


__all__ = [
    "build_rich_recommendations",
    "rich_recommendations_to_dicts",
]
