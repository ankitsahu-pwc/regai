"""Gap-analysis engine — powers the "Gap Identification" page.

Collects five families of gaps across the current session:

1. **Missing evidence** — obligations / RTM rows without any
   :class:`services.source_traceability.SourceReference`.
2. **Missing interpretations** — obligations with an empty
   ``role_interpretation`` or an empty ``role_applicability`` when
   client roles are selected.
3. **Missing requirements** — BRD sections that fall short of the
   expected minimum item count declared by
   :data:`services.brd_frd_generator._SECTION_MINIMUM_COUNTS`.
4. **Low-confidence findings** — obligations with ``confidence < 90``,
   scored questions with confidence tags below the threshold, and any
   assessment sub-score below 75.
5. **Human review required** — items with critical guardrail findings,
   items whose ``VoteReport.winner`` fell back to ``deterministic``, and
   items where the LLM-as-judge returned ``REVIEW``.

The engine is pure data — it does not touch Streamlit — so the same
computation can back the UI, an export, or a REST endpoint later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


#: Confidence threshold below which an item is flagged. Kept as a module
#: constant so callers can override with a stricter policy per demo.
LOW_CONFIDENCE_THRESHOLD: float = 90.0

#: Expected minimum requirement counts per BRD section. Mirrors
#: :data:`services.brd_frd_generator._SECTION_MINIMUM_COUNTS` so we can
#: report shortfalls without importing a private constant. Update in
#: lock-step if the BRD generator's expectations change.
_BRD_SECTION_MINIMUMS = {
    "process_business_requirements":   14,
    "data_business_requirements":      14,
    "reporting_business_requirements": 10,
    "functional_requirements":         18,
    "non_functional_requirements":     10,
}


@dataclass
class GapItem:
    """One gap surfaced by the engine.

    ``item_type`` groups the row into one of the five tabs; ``severity``
    is one of ``"critical" | "high" | "medium" | "low"`` and drives
    sorting + CSS class on the page.
    """

    item_type: str
    severity: str
    subject: str
    detail: str
    obligation_id: str = ""
    requirement_id: str = ""
    question_id: str = ""
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapReport:
    """Full audit output — one bucket per Gap Identification tab."""

    missing_evidence: List[GapItem] = field(default_factory=list)
    missing_interpretations: List[GapItem] = field(default_factory=list)
    missing_requirements: List[GapItem] = field(default_factory=list)
    low_confidence: List[GapItem] = field(default_factory=list)
    human_review: List[GapItem] = field(default_factory=list)

    def total(self) -> int:
        return sum([
            len(self.missing_evidence),
            len(self.missing_interpretations),
            len(self.missing_requirements),
            len(self.low_confidence),
            len(self.human_review),
        ])

    def by_severity(self) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for bucket in (
            self.missing_evidence,
            self.missing_interpretations,
            self.missing_requirements,
            self.low_confidence,
            self.human_review,
        ):
            for item in bucket:
                counts[item.severity] = counts.get(item.severity, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _find_missing_evidence(obligations: Iterable[Any]) -> List[GapItem]:
    """Flag obligations with no source_references."""
    out: List[GapItem] = []
    for ob in obligations or []:
        refs = list(getattr(ob, "source_references", []) or [])
        if refs:
            continue
        out.append(GapItem(
            item_type="missing_evidence",
            severity="high",
            subject=str(getattr(ob, "title", None) or getattr(ob, "obligation_id", "")),
            detail=(
                "No source citation attached. The regulation-search matcher "
                "could not link this obligation to a retrieved publication."
            ),
            obligation_id=str(getattr(ob, "obligation_id", "")),
            requirement_id=str(getattr(ob, "source_requirement_id", "")),
            remediation=(
                "Attach a source URL + article/clause reference, or mark "
                "the obligation as manually reviewed."
            ),
        ))
    return out


def _find_missing_interpretations(
    obligations: Iterable[Any],
    *,
    client_roles: Sequence[str],
) -> List[GapItem]:
    """Flag obligations whose role interpretation is empty."""
    out: List[GapItem] = []
    if not client_roles:
        return out  # Nothing to interpret if no roles selected.
    for ob in obligations or []:
        role_interp = str(getattr(ob, "role_interpretation", "") or "").strip()
        applicability = list(getattr(ob, "role_applicability", []) or [])
        if role_interp and applicability:
            continue
        detail_bits: List[str] = []
        if not role_interp:
            detail_bits.append("No role interpretation narrative.")
        if not applicability:
            detail_bits.append("Per-role applicability list is empty.")
        out.append(GapItem(
            item_type="missing_interpretation",
            severity="medium",
            subject=str(getattr(ob, "title", None) or getattr(ob, "obligation_id", "")),
            detail=" ".join(detail_bits),
            obligation_id=str(getattr(ob, "obligation_id", "")),
            remediation="Re-run Agent 1's role-aware interpretation pass.",
        ))
    return out


def _find_missing_requirements(brd_report: Optional[Any]) -> List[GapItem]:
    """Flag BRD sections below expected minimum item count."""
    out: List[GapItem] = []
    if brd_report is None:
        return out
    for section_attr, minimum in _BRD_SECTION_MINIMUMS.items():
        section = getattr(brd_report, section_attr, None)
        if section is None:
            continue
        actual = len(getattr(section, "items", []) or [])
        if actual >= minimum:
            continue
        gap = minimum - actual
        severity = "critical" if actual == 0 else ("high" if gap >= 5 else "medium")
        out.append(GapItem(
            item_type="missing_requirement",
            severity=severity,
            subject=section_attr.replace("_", " ").title(),
            detail=(
                f"Section carries {actual} requirement(s); the BRD "
                f"generator's minimum for this section is {minimum} "
                f"(shortfall = {gap})."
            ),
            remediation=(
                "Re-run Agent 2 with a richer regulation extract, or add "
                "requirements manually before export."
            ),
        ))
    return out


def _find_low_confidence(
    obligations: Iterable[Any],
    scoring_evaluation: Optional[Mapping[str, Any]],
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> List[GapItem]:
    out: List[GapItem] = []

    for ob in obligations or []:
        try:
            conf = float(getattr(ob, "confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= threshold:
            continue
        severity = "critical" if conf < 60 else ("high" if conf < 75 else "medium")
        out.append(GapItem(
            item_type="low_confidence",
            severity=severity,
            subject=str(getattr(ob, "title", None) or getattr(ob, "obligation_id", "")),
            detail=(
                f"Baseline confidence = {conf:.0f}% (threshold {threshold:.0f}%)."
            ),
            obligation_id=str(getattr(ob, "obligation_id", "")),
            requirement_id=str(getattr(ob, "source_requirement_id", "")),
            remediation="Review the source citation and regulatory basis.",
        ))

    if scoring_evaluation is not None:
        confidence_pct = float(scoring_evaluation.get("evaluation_confidence_pct") or 100.0)
        if confidence_pct < threshold:
            out.append(GapItem(
                item_type="low_confidence",
                severity="high" if confidence_pct < 60 else "medium",
                subject="Assessment · evaluation confidence",
                detail=(
                    f"Evaluation confidence {confidence_pct:.1f}% is below "
                    f"the {threshold:.0f}% threshold."
                ),
                remediation="Answer more questions or attach evidence to raise confidence.",
            ))

    return out


def _find_human_review(
    obligations: Iterable[Any],
    scoring_evaluation: Optional[Mapping[str, Any]],
    analysis_metadata: Optional[Mapping[str, Any]],
) -> List[GapItem]:
    """Aggregate items that need explicit human adjudication.

    Sources checked:

    * Assessment ``signals["_guardrail_report"]`` — critical findings.
    * Assessment ``signals["_vote_report"]`` — winner fell back to
      deterministic OR judge returned REVIEW.
    * Analysis ``metadata["_guardrail_report"]`` — same, at the
      analysis-level.
    * Obligations with the classifier verb ``Must / Shall`` but no
      source citation (a mandatory obligation without evidence is the
      highest-priority review item).
    """
    out: List[GapItem] = []

    def _push_guardrail(report: Optional[Mapping[str, Any]], component: str) -> None:
        if not report:
            return
        for finding in report.get("findings") or []:
            if finding.get("severity") != "critical":
                continue
            out.append(GapItem(
                item_type="human_review",
                severity="critical",
                subject=f"{component}: {finding.get('category', 'guardrail')}",
                detail=str(finding.get("message") or "Guardrail flagged a critical issue."),
                remediation=str(
                    finding.get("remediation")
                    or "Human review required before this output can be exported."
                ),
                metadata={"finding": dict(finding)},
            ))

    def _push_vote(report: Optional[Mapping[str, Any]], component: str) -> None:
        if not report:
            return
        winner = str(report.get("winner") or "")
        judge = report.get("judge_verdict") or {}
        judge_winner = str((judge or {}).get("winner") or "")
        if winner == "deterministic" and report.get("n_llm_candidates"):
            out.append(GapItem(
                item_type="human_review",
                severity="medium",
                subject=f"{component}: LLM output rejected",
                detail=str(report.get("reason") or "LLM output failed the vote."),
                remediation=(
                    "Deterministic baseline used. Human should confirm before export."
                ),
                metadata={"vote": dict(report)},
            ))
        if judge_winner == "REVIEW":
            out.append(GapItem(
                item_type="human_review",
                severity="high",
                subject=f"{component}: judge escalated to REVIEW",
                detail=str((judge or {}).get("rationale") or "Judge could not decide."),
                remediation="Adjudicate manually.",
                metadata={"vote": dict(report)},
            ))

    if scoring_evaluation is not None:
        signals = scoring_evaluation.get("signals") or scoring_evaluation.get("_signals") or {}
        if isinstance(signals, Mapping):
            _push_guardrail(signals.get("_guardrail_report"), "Assessment scoring")
            _push_vote(signals.get("_vote_report"), "Assessment scoring")
        _push_guardrail(scoring_evaluation.get("_guardrail_report"), "Assessment scoring")
        _push_vote(scoring_evaluation.get("_vote_report"), "Assessment scoring")

    if analysis_metadata is not None:
        _push_guardrail(analysis_metadata.get("_guardrail_report"), "Regulatory analysis")
        _push_vote(analysis_metadata.get("_vote_report"), "Regulatory analysis")

    # Mandatory-without-evidence sweep.
    for ob in obligations or []:
        verb = str(getattr(ob, "obligation_verb", "") or "").strip().lower()
        if verb not in {"must", "shall"}:
            continue
        refs = list(getattr(ob, "source_references", []) or [])
        if refs:
            continue
        out.append(GapItem(
            item_type="human_review",
            severity="high",
            subject=str(getattr(ob, "title", None) or getattr(ob, "obligation_id", "")),
            detail=(
                "Mandatory obligation (verb = "
                f"{verb.title()}) has no source citation."
            ),
            obligation_id=str(getattr(ob, "obligation_id", "")),
            remediation=(
                "Attach citation or explicitly downgrade the verb after review."
            ),
            metadata={"verb": verb},
        ))

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_gap_report(
    *,
    analysis: Optional[Any] = None,
    rtm_artifact: Optional[Any] = None,
    scoring_evaluation: Optional[Mapping[str, Any]] = None,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> GapReport:
    """Compute the full gap report for the current session state.

    All arguments are optional so the page can render partial reports
    while the user is still walking through the wizard.
    """
    obligations: List[Any] = list(getattr(analysis, "obligations", []) or []) if analysis else []
    client_roles = list(getattr(analysis, "client_roles", []) or []) if analysis else []
    brd_report = getattr(analysis, "brd_report", None) if analysis else None
    analysis_metadata = getattr(analysis, "metadata", None) if analysis else None

    # RTM entries may carry citations that the obligation list lacks (or
    # vice versa). Merge for the missing-evidence check so we don't
    # false-positive on obligations whose RTM row already has a citation.
    rtm_entries = list(getattr(rtm_artifact, "entries", []) or []) if rtm_artifact else []
    rtm_refs_by_ob: Dict[str, int] = {}
    for entry in rtm_entries:
        ob_id = str(getattr(entry, "obligation_id", "") or "")
        if not ob_id:
            continue
        rtm_refs_by_ob[ob_id] = rtm_refs_by_ob.get(ob_id, 0) + len(
            getattr(entry, "source_references", []) or [],
        )

    def _obligation_effective_ref_count(ob: Any) -> int:
        own = len(getattr(ob, "source_references", []) or [])
        rtm = rtm_refs_by_ob.get(str(getattr(ob, "obligation_id", "")), 0)
        return own + rtm

    # Filter obligations for the missing_evidence detector using the
    # merged count so we don't double-report.
    obligations_missing_refs = [
        ob for ob in obligations if _obligation_effective_ref_count(ob) == 0
    ]

    return GapReport(
        missing_evidence=_find_missing_evidence(obligations_missing_refs),
        missing_interpretations=_find_missing_interpretations(
            obligations, client_roles=client_roles,
        ),
        missing_requirements=_find_missing_requirements(brd_report),
        low_confidence=_find_low_confidence(
            obligations, scoring_evaluation, threshold=threshold,
        ),
        human_review=_find_human_review(
            obligations, scoring_evaluation, analysis_metadata,
        ),
    )


__all__ = [
    "GapItem",
    "GapReport",
    "LOW_CONFIDENCE_THRESHOLD",
    "build_gap_report",
]
