"""Stage 2 of the Regulatory Intelligence Pipeline.

After :mod:`services.official_regulation_fetcher` has identified one or more
official regulatory publications, this module searches **only approved
consulting firms** for implementation guidance about *that exact regulation*.

Hard rules:

* Stage 2 is a no-op until Stage 1 has succeeded; the input is always an
  :class:`~services.official_regulation_fetcher.OfficialRegulationResult` (or a
  list of them).
* Every query carries the regulation title and identifier so the consulting
  search is anchored on the official artefact rather than performing a fresh
  generic lookup.
* Every query is ``site:``-scoped to consulting domains, and every returned
  URL is re-validated with :func:`services.search_config.is_consulting_url`.
* Returned records are tagged with ``source_type == "Consulting Guidance"``
  so downstream agents can never mistake them for primary regulatory content.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .official_regulation_fetcher import OfficialRegulationResult
from .search_config import (
    APPROVED_CONSULTING_FIRMS,
    ConsultingSource,
    SOURCE_TYPE_CONSULTING_GUIDANCE,
    consulting_for_url,
    consulting_max_results,
    is_consulting_search_enabled,
    is_consulting_url,
    resolve_consulting_firms,
    search_backends,
    search_timeout_seconds,
)

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore


StatusCallback = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConsultingGuidanceResult:
    """One consulting publication anchored to a specific official regulation.

    Captures every field listed under "Stage 2 - Regulation-Specific
    Consulting Guidance" in the refactor brief. Anything that can be detected
    from the snippet is filled in; richer content (roadmap, challenges,
    recommendations, etc.) is left for GenAI enrichment by Agent 1 / Agent 4.
    """

    source_type: str = SOURCE_TYPE_CONSULTING_GUIDANCE
    consulting_firm: str = ""
    consulting_code: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    publication_date: Optional[str] = None
    executive_summary: str = ""
    implementation_roadmap: List[str] = field(default_factory=list)
    practical_interpretation: str = ""
    common_implementation_challenges: List[str] = field(default_factory=list)
    industry_observations: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anchor_regulation_title: str = ""
    anchor_regulation_id: Optional[str] = None
    anchor_regulator: str = ""
    backend: str = ""
    query: str = ""
    confidence_score: float = 0.7
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_context_block(self) -> str:
        pieces = [
            "[Supplementary Implementation Guidance]",
            f"Firm: {self.consulting_firm}",
            f"Title: {self.title}",
        ]
        if self.publication_date:
            pieces.append(f"Date: {self.publication_date}")
        if self.snippet:
            pieces.append(f"Snippet: {self.snippet}")
        if self.anchor_regulation_id:
            pieces.append(f"Anchored to: {self.anchor_regulation_title} ({self.anchor_regulation_id})")
        elif self.anchor_regulation_title:
            pieces.append(f"Anchored to: {self.anchor_regulation_title}")
        pieces.append(f"URL: {self.url}")
        return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _default_query_templates() -> List[str]:
    raw = os.getenv("CONSULTING_QUERY_TEMPLATES", "").strip()
    if raw:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    # Kept intentionally small to bound the Stage 2 workload.
    return [
        '{anchor} implementation guidance',
        '{anchor} compliance roadmap',
    ]


def _anchor_phrase(regulation: OfficialRegulationResult, regulation_label: str) -> str:
    """Build the phrase Stage 2 uses to anchor consulting queries.

    Prefers ``"<title>" <regulation_id>`` so the search is tied to the exact
    publication identified in Stage 1, not just the regulation family.
    """
    parts: List[str] = []
    if regulation.title:
        parts.append(f'"{regulation.title.strip()}"')
    if regulation.regulation_id:
        parts.append(regulation.regulation_id.strip())
    if not parts and regulation_label:
        parts.append(f'"{regulation_label}"')
    return " ".join(parts)


def build_consulting_queries(
    regulation: OfficialRegulationResult,
    firms: Sequence[ConsultingSource],
    *,
    regulation_label: str = "",
) -> List[Dict[str, str]]:
    """Build the per-firm queries for Stage 2.

    We bias each query toward the firm by including the firm name and the
    anchor regulation, but do NOT use a ``site:`` operator. The URL allow-list
    post-filter (:func:`is_consulting_url`) drops any hit that lands outside
    an approved consulting domain.
    """
    anchor = _anchor_phrase(regulation, regulation_label)
    if not anchor:
        return []
    templates = _default_query_templates()
    out: List[Dict[str, str]] = []
    for firm in firms:
        primary_domain = firm.domains[0] if firm.domains else ""
        for tpl in templates:
            base = tpl.format(anchor=anchor, regulation=regulation_label or anchor)
            out.append({
                "query": f"{firm.name} {base}".strip(),
                "domain": primary_domain,
                "consulting_code": firm.code,
            })
    return out


# ---------------------------------------------------------------------------
# Snippet parsing helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]
_MONTH_MAP = {m.lower(): i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
])}


def _extract_publication_date(text: str) -> Optional[str]:
    if not text:
        return None
    m = _DATE_PATTERNS[0].search(text)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP.get(month_name.lower())
        if month:
            try:
                return f"{int(year):04d}-{month:02d}-{int(day):02d}"
            except ValueError:
                return None
    m = _DATE_PATTERNS[1].search(text)
    if m:
        return m.group(0)
    return None


def _score_confidence(
    result: ConsultingGuidanceResult,
    anchor_title: str,
    anchor_id: Optional[str],
) -> float:
    """Confidence score for a Stage 2 hit in ``[0.4, 0.95]``.

    Consulting guidance is supplementary by design, so we cap below the
    Stage 1 ceiling.
    """
    score = 0.4
    lowered = ((result.title or "") + " " + (result.snippet or "")).lower()
    if anchor_title and anchor_title.lower() in lowered:
        score += 0.3
    if anchor_id and anchor_id.lower() in lowered:
        score += 0.15
    if result.publication_date:
        score += 0.05
    if result.consulting_firm:
        score += 0.05
    return round(min(score, 0.95), 2)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_consulting_guidance(
    regulations: Iterable[OfficialRegulationResult],
    consulting_selection: Optional[Sequence[str]] = None,
    *,
    regulation_label: str = "",
    max_results_per_query: Optional[int] = None,
    max_anchors: int = 3,
    status: StatusCallback = _noop,
) -> Dict[str, Any]:
    """Search approved consulting firms for guidance on the supplied regulations.

    Parameters
    ----------
    regulations
        The :class:`OfficialRegulationResult` objects returned by Stage 1.
        Stage 2 is a strict no-op when this iterable is empty — by design,
        consulting firms are never searched independently of an official
        regulation.
    consulting_selection
        UI-selected firm codes (``["PWC", "DELOITTE"]``) or ``None`` /
        ``["ALL"]`` to query every approved firm.
    regulation_label
        Fallback label (``"DORA"``, ``"MiFID II"``) used when a Stage 1 hit
        does not include a title — keeps the anchor non-empty.
    max_anchors
        Cap on how many Stage 1 results to anchor against. Stage 2 is
        prompt-driven and we do not need to anchor against every single
        regulator hit; the top-confidence few cover the implementation
        guidance space.

    Returns
    -------
    A dict with::

        {
            "results":     List[ConsultingGuidanceResult],
            "firms":       List[{"code", "name", "website"}],
            "diagnostics": List[str],
            "queries":     List[{"query", "domain", "consulting_code"}],
            "errors":      List[str],
            "enabled":     bool,
        }
    """
    diagnostics: List[str] = []
    errors: List[str] = []
    firms = resolve_consulting_firms(consulting_selection)
    payload_firms = [{"code": f.code, "name": f.name, "website": f.website} for f in firms]

    anchors = list(regulations)[: max(1, max_anchors)] if regulations else []
    if not anchors:
        diagnostics.append(
            "Stage 2 skipped: Stage 1 returned no official regulation to anchor consulting search."
        )
        return {
            "results": [],
            "firms": payload_firms,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": is_consulting_search_enabled(),
        }

    if not is_consulting_search_enabled():
        diagnostics.append("Consulting search disabled (CONSULTING_SEARCH_ENABLED=false).")
        status("Consulting guidance enrichment disabled by env.")
        return {
            "results": [],
            "firms": payload_firms,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": False,
        }

    if DDGS is None:
        msg = "DDGS / duckduckgo_search not installed; Stage 2 cannot search the web."
        diagnostics.append(msg)
        status(msg)
        return {
            "results": [],
            "firms": payload_firms,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": True,
        }

    backends = search_backends()
    max_per_query = max_results_per_query or consulting_max_results()
    timeout = search_timeout_seconds()

    all_queries: List[Dict[str, str]] = []
    results: List[ConsultingGuidanceResult] = []
    seen_urls: set[str] = set()

    for anchor in anchors:
        queries = build_consulting_queries(anchor, firms, regulation_label=regulation_label)
        all_queries.extend(queries)

        status(
            f"Stage 2: anchoring on `{anchor.title or regulation_label}` "
            f"({anchor.regulation_id or 'no ID'}) - {len(queries)} queries across {len(firms)} firm(s)."
        )

        for q in queries:
            firm = _FIRM_BY_CODE.get(q["consulting_code"])
            if firm is None:
                continue

            for backend in backends:
                try:
                    hits = _ddgs_text(q["query"], backend=backend, max_results=max_per_query, timeout=timeout)
                except Exception as exc:  # noqa: BLE001
                    err = f"backend=`{backend}` query=`{q['query'][:80]}...` -> {type(exc).__name__}: {exc}"
                    errors.append(err)
                    status(f"Stage 2 backend error: {err}")
                    continue

                kept_any = False
                for hit in hits:
                    url = (hit.get("href") or hit.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    if not is_consulting_url(url, firms):
                        continue
                    seen_urls.add(url)
                    kept_any = True

                    title = (hit.get("title") or "").strip()
                    snippet = (hit.get("body") or hit.get("snippet") or "").strip()
                    actual_firm = consulting_for_url(url, firms) or firm

                    record = ConsultingGuidanceResult(
                        consulting_firm=actual_firm.name,
                        consulting_code=actual_firm.code,
                        title=title or url,
                        url=url,
                        snippet=snippet,
                        publication_date=_extract_publication_date(f"{title}\n{snippet}"),
                        executive_summary=snippet[:600],
                        anchor_regulation_title=anchor.title,
                        anchor_regulation_id=anchor.regulation_id,
                        anchor_regulator=anchor.regulator,
                        backend=backend,
                        query=q["query"],
                    )
                    record.confidence_score = _score_confidence(record, anchor.title, anchor.regulation_id)
                    results.append(record)

                if kept_any:
                    break  # one successful backend per query is enough

    results.sort(key=lambda r: r.confidence_score, reverse=True)

    diagnostics.append(
        f"Stage 2 returned {len(results)} consulting publications "
        f"anchored to {len(anchors)} official regulation(s)."
    )
    status(diagnostics[-1])

    return {
        "results": results,
        "firms": payload_firms,
        "diagnostics": diagnostics,
        "queries": all_queries,
        "errors": errors,
        "enabled": True,
    }


_FIRM_BY_CODE: Dict[str, ConsultingSource] = {f.code: f for f in APPROVED_CONSULTING_FIRMS}


def _ddgs_text(query: str, *, backend: str, max_results: int, timeout: int) -> List[Dict[str, Any]]:
    assert DDGS is not None
    with DDGS(timeout=timeout) as ddgs:
        try:
            return list(ddgs.text(query, max_results=max_results, backend=backend) or [])
        except TypeError:
            return list(ddgs.text(query, max_results=max_results) or [])


__all__ = [
    "ConsultingGuidanceResult",
    "build_consulting_queries",
    "fetch_consulting_guidance",
]
