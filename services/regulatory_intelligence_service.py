"""Coordinator for the hierarchical Regulatory Intelligence Pipeline.

This service is the single entry point the rest of the application should use
when it needs regulatory context. It chains Stage 1 (official regulators) and
Stage 2 (approved consulting firms) and produces a single combined package:

::

    User Request
        |
        v
    Stage 1: Official Regulation Search
        |  -> OfficialRegulationResult[]
        v
    Regulation Selection (top by confidence_score)
        |
        v
    Stage 2: Consulting Guidance Search (anchored on Stage 1 hits)
        |  -> ConsultingGuidanceResult[]
        v
    RegulatoryIntelligencePackage

Why this lives in its own module
--------------------------------
* The BRD generator (Agent 1) only needs the *combined context string* plus a
  structured source list — it should not know about DDGS, allow-lists or
  ranking rules. That separation is enforced here.
* The Streamlit UI (Page 1, Page 2) only ever depends on
  :class:`RegulatoryIntelligencePackage`, so swapping DDGS for another search
  backend later (corporate RAG, Bing API, etc.) is a one-file change.
* The pipeline preserves the **source prioritisation rule**:
  Official Regulator > Official Legislation > Consulting Guidance. Consulting
  output is rendered into prompt context with an unmistakable "Supplementary"
  banner so the LLM can never treat it as the primary obligation source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .consulting_guidance_fetcher import (
    ConsultingGuidanceResult,
    fetch_consulting_guidance,
)
from .official_regulation_fetcher import (
    OfficialRegulationResult,
    fetch_official_regulations,
)
from .search_config import (
    SOURCE_PRIORITY,
    SOURCE_TYPE_CONSULTING_GUIDANCE,
    SOURCE_TYPE_OFFICIAL_LEGISLATION,
    SOURCE_TYPE_OFFICIAL_REGULATOR,
)


StatusCallback = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Package returned to callers
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryIntelligencePackage:
    """Combined output of Stage 1 + Stage 2.

    Attributes
    ----------
    regulation
        The free-form regulation label supplied by the user.
    regulator_selection
        The regulator codes that were searched (or ``["ALL"]``).
    consulting_selection
        The consulting firm codes that were searched (or ``["ALL"]``).
    official_results
        Stage 1 hits (regulator-domain only).
    consulting_results
        Stage 2 hits (consulting-domain only, anchored on a Stage 1 hit).
    context_text
        Prompt-ready text block with official content first and a clearly
        labelled "Supplementary Implementation Guidance" section after.
        Empty string when no live content was retrieved.
    source_summary
        Quick stats consumed by the UI (counts per source type, etc.).
    diagnostics
        Human-readable log lines from both stages (used by the UI's
        "Search diagnostics" expander).
    errors
        Backend / connection errors from both stages, in a flat list.
    """

    regulation: str
    regulator_selection: List[str]
    consulting_selection: List[str]
    official_results: List[OfficialRegulationResult] = field(default_factory=list)
    consulting_results: List[ConsultingGuidanceResult] = field(default_factory=list)
    context_text: str = ""
    source_summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stage1_enabled: bool = True
    stage2_enabled: bool = True

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def has_official_content(self) -> bool:
        return bool(self.official_results)

    @property
    def has_any_content(self) -> bool:
        return bool(self.official_results or self.consulting_results)

    def all_sources(self) -> List[Dict[str, Any]]:
        """Flat list of every retrieved document, ordered by source priority.

        Each row contains the metadata required by Requirement 4:
        ``source_type``, ``regulator``, ``source_url``, ``publication_date``
        and ``confidence_score``.
        """
        rows: List[Dict[str, Any]] = []
        priority_index = {t: i for i, t in enumerate(SOURCE_PRIORITY)}

        for r in self.official_results:
            rows.append({
                "source_type": r.source_type,
                "regulator": r.regulator,
                "regulator_code": r.regulator_code,
                "consulting_firm": "",
                "title": r.title,
                "publication_type": r.publication_type or "",
                "regulation_id": r.regulation_id or "",
                "publication_date": r.publication_date or "",
                "source_url": r.url,
                "snippet": r.snippet,
                "confidence_score": r.confidence_score,
                "backend": r.backend,
                "query": r.query,
            })
        for c in self.consulting_results:
            rows.append({
                "source_type": c.source_type,
                "regulator": c.anchor_regulator,
                "regulator_code": "",
                "consulting_firm": c.consulting_firm,
                "title": c.title,
                "publication_type": "Consulting Article",
                "regulation_id": c.anchor_regulation_id or "",
                "publication_date": c.publication_date or "",
                "source_url": c.url,
                "snippet": c.snippet,
                "confidence_score": c.confidence_score,
                "backend": c.backend,
                "query": c.query,
            })

        def _year(row: Dict[str, Any]) -> int:
            """Parse the publication year for the freshness sort key.

            Undated results collapse to ``0`` so they sink to the
            bottom of their (source_type, confidence) bucket rather
            than mixing randomly with dated ones. This mirrors the
            same freshness-aware ordering that Stage 1's Native/DDGS
            paths apply — see ``official_regulation_fetcher._sort_key``.
            """
            raw = row.get("publication_date") or ""
            try:
                return int(str(raw)[:4])
            except (TypeError, ValueError):
                return 0

        # Sort keys (all descending except source-type priority which
        # is a small numeric rank where lower is better):
        #   1. Source-type priority (Official > Legislation > Consulting)
        #   2. Confidence score (relevance)
        #   3. Publication year (freshness — latest first)
        # The confidence and year keys are wrapped in ``-`` so ``sort``
        # (which is ascending) still ranks highest first.
        rows.sort(key=lambda row: (
            priority_index.get(row["source_type"], 99),
            -float(row.get("confidence_score") or 0.0),
            -_year(row),
        ))
        return rows


# ---------------------------------------------------------------------------
# Offline fallback context (used by the BRD generator when Stage 1 is empty)
# ---------------------------------------------------------------------------
#
# The pipeline used to carry a hard-coded DORA baseline that was substituted
# whenever Stage 1 returned nothing for the ``DORA`` label. That leaked DORA
# content into runs targeting different regulations and made DORA searches
# silently succeed against a canned scaffold instead of the live regulator
# corpus. The new contract is regulation-agnostic: every regulation - DORA
# included - either flows through live official content, an uploaded
# regulation document, or the neutral "no authoritative source available"
# disclaimer below. There is no per-regulation baseline any more.


def offline_baseline_for(regulation: str) -> str:
    """Return a regulation-neutral offline baseline when Stage 1 is empty.

    The baseline is intentionally short and honest about the fact that no
    authoritative source could be retrieved for the caller's regulation
    code. It carries no regulation-specific obligations, so it can never
    misrepresent DORA (or any other regulation) content when downstream
    agents render it into a BRD / questionnaire.
    """
    label = (regulation or "").strip() or "Selected regulation"
    return (
        f"{label} - Offline Baseline\n"
        "- Stage 1 returned no live content from approved regulator domains "
        f"for `{label}`.\n"
        "- Treat downstream BRD/FRD content as derived from the LLM's pretrained\n"
        "  knowledge rather than from an authoritative regulatory publication.\n"
        f"- Re-run with regulatory search enabled and `{label}` as the target,\n"
        "  or upload the regulation PDF on Page 1 to provide authoritative context."
    )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

_OFFICIAL_HEADER = "=== OFFICIAL REGULATORY CONTEXT (Primary Source of Truth) ==="
_CONSULTING_HEADER = "=== SUPPLEMENTARY IMPLEMENTATION GUIDANCE (Not Authoritative) ==="
_OFFLINE_HEADER = "=== OFFLINE REGULATORY BASELINE (Authoritative Sources Unavailable) ==="


def _format_context(
    regulation: str,
    official: Sequence[OfficialRegulationResult],
    consulting: Sequence[ConsultingGuidanceResult],
    include_offline_baseline: bool = True,
    char_budget: int = 12000,
) -> str:
    """Render Stage 1 + Stage 2 hits into a single prompt-ready string.

    Ordering enforces the source-priority rule: official content first,
    consulting content last, and the consulting section is preceded by a
    clear "Supplementary" banner.
    """
    blocks: List[str] = []

    if official:
        section = [_OFFICIAL_HEADER]
        for r in official:
            section.append(r.as_context_block())
            section.append("")
        blocks.append("\n".join(section).rstrip())

    if consulting:
        section = [_CONSULTING_HEADER]
        section.append(
            "The following content is published by consulting firms and is "
            "provided ONLY to inform implementation roadmaps and practical "
            "interpretation. It must NEVER override or replace official "
            "regulatory obligations above."
        )
        for c in consulting:
            section.append(c.as_context_block())
            section.append("")
        blocks.append("\n".join(section).rstrip())

    if not blocks and include_offline_baseline:
        blocks.append(_OFFLINE_HEADER + "\n" + offline_baseline_for(regulation))
    elif blocks and include_offline_baseline:
        # Append the baseline as a defensive belt at the end so the LLM has
        # *some* anchor text even if a snippet is malformed. It is clearly
        # labelled so the LLM does not treat it as the primary source.
        blocks.append(_OFFLINE_HEADER + "\n" + offline_baseline_for(regulation))

    rendered = "\n\n".join(blocks).strip()
    if char_budget and len(rendered) > char_budget:
        rendered = rendered[:char_budget] + "\n... (context truncated)"
    return rendered


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------

class RegulatoryIntelligenceService:
    """Stateless façade over Stage 1 + Stage 2 fetchers.

    The class shape (rather than free functions) keeps the door open for
    plugging in different stage implementations later — e.g. a corporate RAG
    backed by the same allow-list — without touching call sites.
    """

    def __init__(
        self,
        *,
        stage1_fetcher: Callable[..., Dict[str, Any]] = fetch_official_regulations,
        stage2_fetcher: Callable[..., Dict[str, Any]] = fetch_consulting_guidance,
    ) -> None:
        self._stage1 = stage1_fetcher
        self._stage2 = stage2_fetcher

    def gather(
        self,
        regulation: str,
        *,
        regulator_selection: Optional[Sequence[str]] = None,
        consulting_selection: Optional[Sequence[str]] = None,
        max_official_results: Optional[int] = None,
        max_consulting_results: Optional[int] = None,
        max_consulting_anchors: int = 3,
        include_consulting: bool = True,
        include_offline_baseline: bool = True,
        char_budget: int = 12000,
        exhaustive: bool = False,
        status: StatusCallback = _noop,
    ) -> RegulatoryIntelligencePackage:
        """Run Stage 1 + Stage 2 and return the combined intelligence package.

        ``include_consulting=False`` disables Stage 2 regardless of the env
        flag — useful for unit tests / smoke runs where we only care about
        regulator-domain content.

        ``include_offline_baseline=True`` (default) keeps the historical
        behaviour of appending an offline baseline as a defensive belt so the
        BRD generator never sees an empty context string.

        ``exhaustive=True`` forwards to Stage 1 to enable multi-variant
        native search per regulator, a wider DDGS template set, and
        relaxed early-stop / wall-clock caps. Set this when the user
        has uploaded a regulation document and expects a full sweep of
        every publication the approved regulators expose.
        """
        diagnostics: List[str] = []
        errors: List[str] = []

        # ---- Stage 1 -------------------------------------------------------
        if exhaustive:
            status(
                "Stage 1 (exhaustive): sweeping every approved regulator "
                "domain for the uploaded regulation."
            )
        else:
            status("Stage 1: searching approved regulator domains.")
        stage1 = self._stage1(
            regulation,
            regulator_selection,
            max_results_per_query=max_official_results,
            exhaustive=exhaustive,
            status=status,
        )
        official_results: List[OfficialRegulationResult] = list(stage1.get("results") or [])
        diagnostics.extend(stage1.get("diagnostics") or [])
        errors.extend(stage1.get("errors") or [])
        stage1_enabled = bool(stage1.get("enabled", True))

        # ---- Stage 2 -------------------------------------------------------
        consulting_results: List[ConsultingGuidanceResult] = []
        stage2_enabled = include_consulting
        if include_consulting and official_results:
            status("Stage 2: anchoring on Stage 1 hits and searching consulting firms.")
            stage2 = self._stage2(
                official_results,
                consulting_selection,
                regulation_label=regulation,
                max_results_per_query=max_consulting_results,
                max_anchors=max_consulting_anchors,
                status=status,
            )
            consulting_results = list(stage2.get("results") or [])
            diagnostics.extend(stage2.get("diagnostics") or [])
            errors.extend(stage2.get("errors") or [])
            stage2_enabled = bool(stage2.get("enabled", True))
        elif include_consulting and not official_results:
            diagnostics.append(
                "Stage 2 skipped: Stage 1 returned no official publications, so "
                "consulting guidance enrichment was not triggered (consulting "
                "firms are never searched without an anchoring regulation)."
            )
            status(diagnostics[-1])
        else:
            diagnostics.append("Stage 2 disabled by caller (include_consulting=False).")
            status(diagnostics[-1])

        # ---- Combined context + summary -----------------------------------
        context_text = _format_context(
            regulation,
            official_results,
            consulting_results,
            include_offline_baseline=include_offline_baseline,
            char_budget=char_budget,
        )

        summary = {
            "official_count": len(official_results),
            "consulting_count": len(consulting_results),
            "by_source_type": _count_by_source_type(official_results, consulting_results),
            "regulators_hit": sorted({r.regulator_code for r in official_results}),
            "consulting_firms_hit": sorted({c.consulting_code for c in consulting_results}),
            "stage1_enabled": stage1_enabled,
            "stage2_enabled": stage2_enabled,
            "stage1_query_count": len(stage1.get("queries") or []),
            "stage2_query_count": (
                0 if not consulting_results else
                len({(c.query or "") for c in consulting_results})
            ),
        }

        return RegulatoryIntelligencePackage(
            regulation=regulation,
            regulator_selection=list(regulator_selection or ["ALL"]),
            consulting_selection=list(consulting_selection or ["ALL"]),
            official_results=official_results,
            consulting_results=consulting_results,
            context_text=context_text,
            source_summary=summary,
            diagnostics=diagnostics,
            errors=errors,
            stage1_enabled=stage1_enabled,
            stage2_enabled=stage2_enabled,
        )


def _count_by_source_type(
    official: Sequence[OfficialRegulationResult],
    consulting: Sequence[ConsultingGuidanceResult],
) -> Dict[str, int]:
    counts: Dict[str, int] = {
        SOURCE_TYPE_OFFICIAL_REGULATOR: 0,
        SOURCE_TYPE_OFFICIAL_LEGISLATION: 0,
        SOURCE_TYPE_CONSULTING_GUIDANCE: 0,
    }
    for r in official:
        counts[r.source_type] = counts.get(r.source_type, 0) + 1
    counts[SOURCE_TYPE_CONSULTING_GUIDANCE] = len(consulting)
    return counts


# ---------------------------------------------------------------------------
# Convenience module-level helpers (mirror the old ``monitor_regulation_updates``
# call shape so existing call sites are easy to migrate).
# ---------------------------------------------------------------------------

_DEFAULT_SERVICE = RegulatoryIntelligenceService()


def gather_regulatory_intelligence(
    regulation: str,
    *,
    regulator_selection: Optional[Sequence[str]] = None,
    consulting_selection: Optional[Sequence[str]] = None,
    include_consulting: bool = True,
    exhaustive: bool = False,
    status: StatusCallback = _noop,
) -> RegulatoryIntelligencePackage:
    """Module-level helper used by the BRD generator and the Streamlit UI."""
    return _DEFAULT_SERVICE.gather(
        regulation,
        regulator_selection=regulator_selection,
        consulting_selection=consulting_selection,
        include_consulting=include_consulting,
        exhaustive=exhaustive,
        status=status,
    )


__all__ = [
    "RegulatoryIntelligencePackage",
    "RegulatoryIntelligenceService",
    "gather_regulatory_intelligence",
    "offline_baseline_for",
]
