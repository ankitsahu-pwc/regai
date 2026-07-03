"""Source-traceability layer for BRD generation.

Every BRD requirement, obligation, control checkpoint, risk, and bullet item
should be traceable back to the regulatory publication that motivated it.
This module is the single source of truth for how source metadata is:

* captured from the ``RegulatoryIntelligencePackage`` produced by Stage 1
  (and Stage 2, if enabled) of the Regulatory Intelligence Pipeline; and
* matched to the deterministic / GenAI-generated BRD requirement rows.

Design rules
------------
1. **Never fabricate**. If we cannot anchor a requirement to at least one
   retrieved publication we emit a single ``SourceReference`` with
   ``source_type == "No live source available"`` so the BRD makes the
   missing provenance explicit instead of inventing a URL.
2. **Deterministic matching**. We match the alignment / detail / regulation
   reference text of each requirement against the title, snippet, regulation
   identifier and publication type fields surfaced by the fetchers. The
   matcher is intentionally simple (token + regex), runs offline, and is
   independent of any LLM call.
3. **Stable JSON shape**. Every reference is exported as a plain dict so it
   flows cleanly through ``json.dumps`` (used by the orchestrator,
   persistence layer, and Streamlit export panels). Pydantic stays out of
   this hot path so we never have to teach the BRD generator's LLM schema
   about citation fields.

Public API
----------
* :class:`SourceReference`              — one citation block.
* :class:`SourceCatalogue`              — full set of references retrieved
  for a single BRD generation run, plus the deterministic matcher.
* :func:`build_source_catalogue`        — entry point used by
  :mod:`services.brd_frd_generator`.
* :func:`attach_source_references`      — walk the report and produce the
  ``{item_key -> List[SourceReference]}`` mapping that the DOCX writer and
  downstream agents consume.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .consulting_guidance_fetcher import ConsultingGuidanceResult
from .official_regulation_fetcher import OfficialRegulationResult
from .regulatory_intelligence_service import RegulatoryIntelligencePackage
from .search_config import (
    SOURCE_TYPE_CONSULTING_GUIDANCE,
    SOURCE_TYPE_OFFICIAL_LEGISLATION,
    SOURCE_TYPE_OFFICIAL_REGULATOR,
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

#: Sentinel ``source_type`` written when no live regulatory publication could
#: be matched to a requirement. Renderers must NOT treat this as an
#: authoritative citation; it exists only so the BRD output is honest about
#: the missing trace.
SOURCE_TYPE_NONE = "No live source available"


@dataclass(frozen=True)
class SourceReference:
    """Citation block attached to a BRD requirement / obligation / control.

    Field choices mirror the requirement set in the user brief:

    * ``source_url``           — full URL of the publication.
    * ``title``                — title / document name.
    * ``regulator``            — regulator or issuing authority.
    * ``publication_date``     — ISO date when parsable, else free text.
    * ``regulation_reference`` — article / RTS / ITS / clause / section.
    * ``source_type``          — Official Regulator | Official Legislation
      | Consulting Guidance | Uploaded Document | No live source available.
    * ``publication_type``     — optional richer label (Regulation, RTS, ...).
    * ``confidence``           — heuristic 0.0-1.0 confidence carried over
      from the fetcher; useful for ranking when multiple sources match.
    """

    source_url: str = ""
    title: str = ""
    regulator: str = ""
    publication_date: str = ""
    regulation_reference: str = ""
    source_type: str = ""
    publication_type: str = ""
    confidence: float = 0.0

    @property
    def is_real(self) -> bool:
        """True when this reference points at a retrieved publication."""
        return bool(self.source_url) and self.source_type != SOURCE_TYPE_NONE

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict export used everywhere outside this module."""
        return asdict(self)

    def short_label(self) -> str:
        """Compact one-liner used for inline rendering (DOCX table cell, UI)."""
        parts: List[str] = []
        if self.regulator:
            parts.append(self.regulator)
        if self.regulation_reference and self.regulation_reference not in parts:
            parts.append(self.regulation_reference)
        elif self.publication_type and self.publication_type not in parts:
            parts.append(self.publication_type)
        elif self.title:
            parts.append(self.title[:80])
        if not parts:
            parts.append(self.source_type or SOURCE_TYPE_NONE)
        return " - ".join(parts)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

@dataclass
class SourceCatalogue:
    """All citations available for one BRD generation run.

    The catalogue is built once per ``build_brd_frd_report`` invocation; the
    matcher then walks the BRD report and decides which citations should be
    attached to each requirement / obligation / bullet.
    """

    regulation: str
    references: List[SourceReference] = field(default_factory=list)
    used_uploaded_document: bool = False
    used_offline_baseline: bool = False
    uploaded_document_name: str = ""

    # Cached, lowercase-collated search blobs (``index -> text``). Populated
    # lazily the first time the matcher runs.
    _search_blobs: List[str] = field(default_factory=list, repr=False)

    @property
    def has_real_sources(self) -> bool:
        return any(ref.is_real for ref in self.references)

    def fallback_reference(self) -> SourceReference:
        """Return the sentinel reference for items that fail to match.

        Prefers the strongest available signal:

        * If a regulation document was uploaded -> cite the uploaded doc.
        * Else if we are running on the offline baseline -> a sentinel
          "No live source available" reference so the BRD is honest.
        * Else (real sources exist but none matched a specific requirement)
          -> the top-confidence retrieved publication, so we never leave a
          row uncited when an authoritative publication is in scope.
        """
        if self.used_uploaded_document and self.uploaded_document_name:
            return SourceReference(
                source_url="",
                title=self.uploaded_document_name,
                regulator="Uploaded by user",
                publication_date="",
                regulation_reference=self.regulation,
                source_type="Uploaded Document",
                publication_type="User-supplied regulation document",
                confidence=0.5,
            )
        if self.used_offline_baseline or not self.references:
            return SourceReference(
                source_url="",
                title=f"{self.regulation} - offline baseline",
                regulator="N/A",
                publication_date="",
                regulation_reference=self.regulation,
                source_type=SOURCE_TYPE_NONE,
                publication_type="Offline LLM baseline",
                confidence=0.0,
            )
        # Real sources exist; pick the highest-confidence Official Regulator
        # publication so the row at least cites something authoritative.
        ranked = sorted(
            (r for r in self.references if r.is_real),
            key=lambda r: (
                0 if r.source_type == SOURCE_TYPE_OFFICIAL_REGULATOR else
                1 if r.source_type == SOURCE_TYPE_OFFICIAL_LEGISLATION else 2,
                -r.confidence,
            ),
        )
        if ranked:
            return ranked[0]
        return SourceReference(
            source_url="",
            title="",
            regulator="",
            publication_date="",
            regulation_reference=self.regulation,
            source_type=SOURCE_TYPE_NONE,
            publication_type="",
            confidence=0.0,
        )

    def to_payload(self) -> List[Dict[str, Any]]:
        """JSON-friendly export of the full catalogue."""
        return [ref.to_dict() for ref in self.references]

    # ------------------------------------------------------------------
    # Matcher
    # ------------------------------------------------------------------

    def _build_blobs(self) -> None:
        """Pre-compute the lowercase search blob for every reference once."""
        if self._search_blobs or not self.references:
            return
        blobs: List[str] = []
        for ref in self.references:
            blob = " | ".join([
                ref.title or "",
                ref.publication_type or "",
                ref.regulation_reference or "",
                ref.regulator or "",
                ref.source_type or "",
            ]).lower()
            blobs.append(blob)
        self._search_blobs = blobs

    def match(self, *texts: str, max_results: int = 3) -> List[SourceReference]:
        """Return the best-matching references for the supplied text fragments.

        The fragments typically come from a requirement: its title, detail,
        regulation alignment, acceptance criteria, etc. We score each
        reference by counting how many of the fragment's salient tokens
        (regulation IDs, article/section/RTS/ITS labels, keywords) appear in
        the reference's pre-built search blob.

        ``max_results`` caps how many references we attach so the rendered
        BRD does not become a citation soup; we keep the top-scored ones.
        """
        if not self.references:
            return [self.fallback_reference()]
        self._build_blobs()
        haystack_tokens = _salient_tokens(" ".join(t or "" for t in texts))
        if not haystack_tokens:
            return [self.fallback_reference()]

        scored: List[Tuple[float, SourceReference]] = []
        for idx, ref in enumerate(self.references):
            if not ref.is_real:
                continue
            blob = self._search_blobs[idx]
            if not blob:
                continue
            hit = 0
            for tok in haystack_tokens:
                if tok in blob:
                    hit += 1
            if hit == 0:
                continue
            # Combine hit-count with the reference's own retrieval confidence
            # so high-quality, high-relevance sources rise to the top.
            score = hit + min(1.0, max(0.0, ref.confidence))
            scored.append((score, ref))

        if not scored:
            return [self.fallback_reference()]

        scored.sort(key=lambda kv: (-kv[0], -kv[1].confidence))
        out: List[SourceReference] = []
        seen_urls: set[str] = set()
        for _, ref in scored:
            key = ref.source_url or f"{ref.title}|{ref.regulator}"
            if key in seen_urls:
                continue
            seen_urls.add(key)
            out.append(ref)
            if len(out) >= max_results:
                break
        return out or [self.fallback_reference()]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _from_official(r: OfficialRegulationResult) -> SourceReference:
    return SourceReference(
        source_url=r.url or "",
        title=r.title or "",
        regulator=r.regulator or "",
        publication_date=r.publication_date or "",
        regulation_reference=r.regulation_id or "",
        source_type=r.source_type or SOURCE_TYPE_OFFICIAL_REGULATOR,
        publication_type=r.publication_type or "",
        confidence=float(r.confidence_score or 0.0),
    )


def _from_consulting(c: ConsultingGuidanceResult) -> SourceReference:
    return SourceReference(
        source_url=c.url or "",
        title=c.title or "",
        regulator=c.consulting_firm or c.anchor_regulator or "",
        publication_date=c.publication_date or "",
        regulation_reference=c.anchor_regulation_id or c.anchor_regulation_title or "",
        source_type=c.source_type or SOURCE_TYPE_CONSULTING_GUIDANCE,
        publication_type="Consulting Article",
        confidence=float(c.confidence_score or 0.0),
    )


def build_source_catalogue(
    package: Optional[RegulatoryIntelligencePackage],
    *,
    regulation: str,
    used_uploaded_document: bool = False,
    uploaded_document_name: str = "",
) -> SourceCatalogue:
    """Return the catalogue used by :func:`attach_source_references`.

    Parameters
    ----------
    package
        The :class:`RegulatoryIntelligencePackage` returned by Stage 1 / Stage
        2 of the pipeline. ``None`` is treated like an empty package (offline
        baseline mode).
    regulation
        The free-form regulation label (``"DORA"``, ``"MiFID II"`` ...).
    used_uploaded_document
        True when the BRD prompt context included text from a user-uploaded
        regulation file (``extra_context`` was non-empty).
    uploaded_document_name
        Friendly display name for the uploaded document, when known.
    """
    references: List[SourceReference] = []
    if package is not None:
        for r in package.official_results:
            references.append(_from_official(r))
        for c in package.consulting_results:
            references.append(_from_consulting(c))

    has_official = bool(package and package.has_official_content) if package else False
    return SourceCatalogue(
        regulation=regulation,
        references=references,
        used_uploaded_document=used_uploaded_document,
        used_offline_baseline=not has_official and not used_uploaded_document,
        uploaded_document_name=uploaded_document_name,
    )


# ---------------------------------------------------------------------------
# Salient-token extraction (deterministic; offline)
# ---------------------------------------------------------------------------

#: Article / clause / section patterns we want to lift verbatim from the
#: requirement text so they can be matched against the reference blob.
_CITATION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\barticle\s+\d+[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bsection\s+\d+(\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bclause\s+\d+(\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:rts|its)[-_ ]?\d+(?:[/_-]\d+)*\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,5}/[A-Z]{2,5}/\d{2,4}/\d{1,3}[A-Z0-9]*\b"),
    re.compile(r"\bregulation\s*\(eu\)\s*\d{4}/\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bdirective\s*\(eu\)\s*\d{4}/\d{1,4}\b", re.IGNORECASE),
)

#: Keyword vocabulary used by the matcher. Order is not significant; entries
#: are case-insensitive substrings. Kept compact and regulation-agnostic so
#: future regulations (MiFID, NIS2 ...) work without code changes.
_KEYWORDS: Tuple[str, ...] = (
    "dora", "ict", "incident", "resilience", "third-party", "third party",
    "register", "outsourcing", "operational resilience", "rts", "its",
    "guideline", "supervisory", "vulnerability", "audit", "evidence",
    "governance", "risk management", "incident reporting", "backup",
    "recovery", "testing", "tlpt", "subcontract", "register of information",
    "concentration", "exit plan", "controls", "monitoring", "classification",
    "notification", "reporting", "policy", "framework", "critical function",
    "important function", "encryption", "access", "access control",
)


def _salient_tokens(text: str) -> List[str]:
    """Return a deduplicated list of search tokens distilled from ``text``.

    Combines verbatim regulatory citations (Article X, RTS/ITS, Regulation
    (EU) YYYY/NNN) with regulation-domain keywords. The output is always
    lowercase so it can be substring-matched against the reference blobs.
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = []
    for pat in _CITATION_PATTERNS:
        for m in pat.finditer(text):
            tokens.append(m.group(0).strip().lower())
    for kw in _KEYWORDS:
        if kw in lowered:
            tokens.append(kw)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Walker — attach references to every BRD item
# ---------------------------------------------------------------------------

#: Composite key used to address items in the BRD when storing the
#: ``{key -> List[SourceReference]}`` mapping. We use prefixes so a single
#: dict can hold references for requirements (``REQ:BR-PRO-001``), bullets
#: (``BUL:Executive Summary:Purpose``), controls
#: (``CTRL:Identify:ICT Inventory``) and risks (``RISK:Incomplete ICT...``)
#: without key collisions.
KEY_REQ = "REQ"
KEY_BULLET = "BUL"
KEY_CTRL = "CTRL"
KEY_RISK = "RISK"


def requirement_key(req_id: str) -> str:
    return f"{KEY_REQ}:{req_id}"


def bullet_key(section_label: str, title: str) -> str:
    return f"{KEY_BULLET}:{section_label}:{title}"


def control_key(stage: str, checkpoint: str) -> str:
    return f"{KEY_CTRL}:{stage}:{checkpoint}"


def risk_key(risk: str) -> str:
    return f"{KEY_RISK}:{risk[:120]}"


def attach_source_references(
    report: Any,
    catalogue: SourceCatalogue,
    *,
    max_per_item: int = 3,
) -> Dict[str, List[SourceReference]]:
    """Walk ``report`` and produce a ``{item_key -> [SourceReference]}`` map.

    The function never mutates the report. Callers (the DOCX writer, the
    Streamlit UI, downstream agents) then look up references for each
    requirement / control / risk / bullet using the helper key functions
    defined above.

    The keys returned are designed to be JSON-serialisable so the map can
    flow through ``BRDArtifact.metadata`` without surprises.
    """
    refs: Dict[str, List[SourceReference]] = {}

    # ------------------------------------------------------------------
    # Standard sections (bullet lists)
    # ------------------------------------------------------------------
    standard_sections: List[Tuple[str, Any]] = [
        ("Executive Summary", report.executive_summary),
        ("Objectives", report.objectives),
        ("Scope", report.scope),
        ("Stakeholders", report.stakeholders),
        ("Current State Challenges", report.current_state_challenges),
        ("Target State Overview", report.target_state_overview),
        ("Assumptions", report.assumptions),
        ("Dependencies", report.dependencies),
        ("Success Criteria", report.success_criteria),
        ("Appendix", report.appendix),
    ]
    for label, section in standard_sections:
        for item in getattr(section, "items", []) or []:
            matches = catalogue.match(item.title, item.description, label,
                                      max_results=max_per_item)
            refs[bullet_key(label, item.title)] = matches

    # ------------------------------------------------------------------
    # Requirement sections (rich rows)
    # ------------------------------------------------------------------
    requirement_sections = [
        report.process_business_requirements,
        report.data_business_requirements,
        report.reporting_business_requirements,
        report.functional_requirements,
        report.non_functional_requirements,
    ]
    for section in requirement_sections:
        for item in getattr(section, "items", []) or []:
            matches = catalogue.match(
                item.category, item.requirement, item.detailed_requirement,
                item.dora_alignment, item.acceptance_criteria,
                max_results=max_per_item,
            )
            refs[requirement_key(item.id)] = matches

    # ------------------------------------------------------------------
    # Control framework
    # ------------------------------------------------------------------
    cf = report.control_framework
    for cp in getattr(cf, "lifecycle_checkpoints", []) or []:
        matches = catalogue.match(
            cp.stage, cp.control_checkpoint, cp.requirement,
            cp.tooling_expectation, cp.evidence,
            max_results=max_per_item,
        )
        refs[control_key(cp.stage, cp.control_checkpoint)] = matches
    for label, bullets in [
        ("Preventive Controls", cf.preventive_controls),
        ("Detective Controls", cf.detective_controls),
        ("Corrective Controls", cf.corrective_controls),
        ("Governance Controls", cf.governance_controls),
        ("Tooling Integration", cf.tooling_integration),
    ]:
        for b in bullets or []:
            matches = catalogue.match(b.title, b.description, label,
                                      max_results=max_per_item)
            refs[bullet_key(label, b.title)] = matches

    # ------------------------------------------------------------------
    # Risk register
    # ------------------------------------------------------------------
    for r in getattr(report.risks_and_mitigations, "items", []) or []:
        matches = catalogue.match(r.risk, r.impact, r.mitigation, r.owner,
                                  max_results=max_per_item)
        refs[risk_key(r.risk)] = matches

    return refs


def references_to_payload(
    refs: Dict[str, List[SourceReference]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert the ``{key -> [SourceReference]}`` map to a JSON-safe dict."""
    return {key: [ref.to_dict() for ref in items] for key, items in refs.items()}


def deduplicated_catalogue_payload(
    refs_by_item: Dict[str, List[SourceReference]],
    catalogue: SourceCatalogue,
) -> List[Dict[str, Any]]:
    """Return the unique set of sources actually used somewhere in the BRD.

    Used by the DOCX writer + Streamlit panel to render the master "Source
    References" section. Falls back to the catalogue when nothing was
    matched (so the section is never empty when live sources exist).
    """
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for items in refs_by_item.values():
        for ref in items:
            if not ref.is_real:
                continue
            key = (ref.source_url, ref.title)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref.to_dict())
    if out:
        return out
    return [ref.to_dict() for ref in catalogue.references if ref.is_real]


__all__ = [
    "SOURCE_TYPE_NONE",
    "KEY_REQ",
    "KEY_BULLET",
    "KEY_CTRL",
    "KEY_RISK",
    "SourceReference",
    "SourceCatalogue",
    "attach_source_references",
    "bullet_key",
    "build_source_catalogue",
    "control_key",
    "deduplicated_catalogue_payload",
    "references_to_payload",
    "requirement_key",
    "risk_key",
]
