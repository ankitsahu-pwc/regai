"""Stage 1 of the Regulatory Intelligence Pipeline.

Searches **only** the approved regulator domains listed in
:mod:`services.search_config` and returns structured records describing each
official regulatory publication that matched the query.

Design rules (enforced here, not negotiable):

1. Every web query is constrained with a ``site:`` filter so the underlying
   search engine cannot drift onto a non-authoritative domain.
2. Every result is post-filtered with
   :func:`services.search_config.is_regulator_url` so that — even if a backend
   ignores the site filter — Wikipedia / blogs / news sites are dropped.
3. The function never raises. If DDGS is missing, a backend errors out, or
   no regulators match, the caller still gets a well-formed empty result and
   a human-readable diagnostic log.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from .search_config import (
    APPROVED_REGULATORS,
    PUBLICATION_TYPES,
    RegulatorSource,
    SOURCE_TYPE_OFFICIAL_LEGISLATION,
    SOURCE_TYPE_OFFICIAL_REGULATOR,
    is_regulator_url,
    is_regulatory_search_enabled,
    regulator_for_url,
    regulatory_max_results,
    resolve_regulators,
    search_backends,
    search_timeout_seconds,
)
from .native_regulator_search import (
    native_search,
    supported_regulator_codes as native_supported_codes,
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
class OfficialRegulationResult:
    """One regulatory publication retrieved from an approved regulator domain.

    The fields capture the structured metadata called out in Requirement 3
    of the refactor brief. Anything we cannot determine deterministically
    from the search snippet is left as ``None`` / empty string; downstream
    agents (Agent 1 + the BRD generator) will enrich the data using GenAI.
    """

    source_type: str                       # SOURCE_TYPE_OFFICIAL_REGULATOR or _LEGISLATION
    regulator: str                         # e.g. "European Banking Authority"
    regulator_code: str                    # e.g. "EBA"
    title: str
    url: str
    snippet: str = ""
    publication_type: Optional[str] = None # e.g. "RTS", "Guideline", "Q&A"
    regulation_id: Optional[str] = None    # e.g. "EBA/RTS/2024/05"
    publication_date: Optional[str] = None # ISO YYYY-MM-DD when parseable
    version: Optional[str] = None
    executive_summary: str = ""
    key_obligations: List[str] = field(default_factory=list)
    impacted_business_functions: List[str] = field(default_factory=list)
    related_regulations: List[str] = field(default_factory=list)
    backend: str = ""
    query: str = ""
    confidence_score: float = 0.85
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_context_block(self) -> str:
        """Render the result as a prompt-ready text block for the BRD generator."""
        pieces: List[str] = []
        pieces.append(f"Title: {self.title}")
        pieces.append(f"Regulator: {self.regulator} ({self.regulator_code})")
        if self.publication_type:
            pieces.append(f"Publication type: {self.publication_type}")
        if self.regulation_id:
            pieces.append(f"Reference: {self.regulation_id}")
        if self.publication_date:
            pieces.append(f"Publication date: {self.publication_date}")
        if self.snippet:
            pieces.append(f"Snippet: {self.snippet}")
        pieces.append(f"URL: {self.url}")
        return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _default_query_templates() -> List[str]:
    raw = os.getenv("REGULATORY_QUERY_TEMPLATES", "").strip()
    if raw:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    # A SMALL number of plain queries. DDGS aggressively rate-limits rapid
    # bursts, so we let DuckDuckGo do the ranking and rely on the URL
    # allow-list post-filter to drop anything not on an approved regulator
    # domain. This is the same low-volume pattern the original code used,
    # plus the new allow-list guarantee that Wikipedia / blogs / news never
    # leak into the BRD context.
    return [
        "{regulation} regulation official text",
        "{regulation} RTS technical standards guidelines",
    ]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _runtime_caps() -> Dict[str, Any]:
    """Caps that prevent Stage 1 from running forever on a wide regulator set.

    We run queries SEQUENTIALLY with a small inter-query delay, because DDGS
    aggressively rate-limits concurrent requests from the same IP (the symptom
    is `DDGSException: No results found` for queries that work fine in
    isolation).
    """
    return {
        # Hard upper bound on total queries dispatched. We now use 2-3 plain
        # queries (not one per regulator) so this can be tight without losing
        # coverage.
        "max_total_queries": _env_int("REGULATORY_SEARCH_MAX_QUERIES", 4),
        # Stop dispatching new queries once we have this many distinct URLs.
        "early_stop_results": _env_int("REGULATORY_SEARCH_EARLY_STOP", 10),
        # Hard wall-clock cap for the whole Stage 1 dispatch loop.
        "max_total_seconds": _env_int("REGULATORY_SEARCH_MAX_SECONDS", 30),
        # Seconds to sleep between sequential queries. 0.4-0.6s avoids DDGS's
        # anti-bot heuristics without making the full run too slow.
        "inter_query_delay_ms": _env_int("REGULATORY_SEARCH_DELAY_MS", 600),
    }


def build_queries(regulation: str, regulators: Sequence[RegulatorSource]) -> List[Dict[str, str]]:
    """Build the search queries for Stage 1.

    The query volume is intentionally low — a small number of plain queries
    biased toward the *full set* of selected regulators rather than one
    query per regulator. The URL allow-list post-filter
    (:func:`is_regulator_url`) is what guarantees that only regulator-domain
    hits survive; the search engine is only there to *find* the URLs.

    When a specific subset of regulators is selected (not "ALL"), the first
    query is augmented with the regulator names to bias the ranking toward
    those regulators. Even without that augmentation, plain regulation
    queries reliably return regulator URLs (eur-lex, eba, esma, etc.) in the
    top 5-10 results.
    """
    label = (regulation or "").strip()
    if not label:
        return []
    templates = _default_query_templates()

    # If the user picked a small subset of regulators, build one extra
    # regulator-biased query. With "ALL" selected we skip this since adding
    # 15 regulator names to one query degrades DDGS ranking.
    regulator_names = [r.name for r in regulators]
    total_regulators = sum(1 for _ in regulators)
    biased_query: Optional[str] = None
    if 1 <= total_regulators <= 4:
        biased_query = f"{' '.join(regulator_names)} {label} regulation"

    out: List[Dict[str, str]] = []
    if biased_query:
        out.append({
            "query": biased_query.strip(),
            "domain": "",
            "regulator_code": regulators[0].code if regulators else "",
        })
    for tpl in templates:
        out.append({
            "query": tpl.format(regulation=label).strip(),
            "domain": "",
            "regulator_code": "",
        })
    return out


# ---------------------------------------------------------------------------
# Metadata extraction (deterministic; runs on the snippet only)
# ---------------------------------------------------------------------------

_REG_ID_PATTERNS = [
    # EBA/RTS/2024/05  -- typical EBA / ESMA reference style
    re.compile(r"\b([A-Z]{2,5}/[A-Z]{2,5}/\d{2,4}/\d{1,3}[A-Z0-9]*)\b"),
    # EU regulations / directives
    re.compile(r"\bRegulation \(EU\)\s*\d{4}/\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bDirective \(EU\)\s*\d{4}/\d{1,4}\b", re.IGNORECASE),
    # FCA / PRA consultation / policy statements
    re.compile(r"\b(?:CP|PS|SS|DP|FG)\d{1,3}/\d{1,3}\b"),
    # ESMA references e.g. ESMA70-1234-567
    re.compile(r"\bESMA\d{1,3}-[\w\d-]+\b"),
]


def _extract_regulation_id(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _REG_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    return None


def _extract_publication_type(text: str, hints: Sequence[str]) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    # Prefer regulator-specific hints when present.
    for hint in hints:
        if hint.lower() in lowered:
            return hint.title() if hint.isalpha() else hint
    for kind in PUBLICATION_TYPES:
        token = kind.lower()
        if token in lowered:
            return kind
    return None


_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b"),
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
    m = _DATE_PATTERNS[2].search(text)
    if m:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        try:
            return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
        except ValueError:
            return None
    return None


def _score_confidence(result: "OfficialRegulationResult", regulation: str) -> float:
    """Heuristic confidence score in ``[0.5, 0.99]``.

    Anchored on:
      * Direct domain match (already enforced) -> +0.5
      * Regulation label appears in the title    -> +0.2
      * Regulation ID detected                   -> +0.15
      * Publication type detected                -> +0.1
      * Publication date detected                -> +0.05
    """
    score = 0.5
    label = (regulation or "").lower()
    if label and label in (result.title or "").lower():
        score += 0.2
    if result.regulation_id:
        score += 0.15
    if result.publication_type:
        score += 0.1
    if result.publication_date:
        score += 0.05
    return round(min(score, 0.99), 2)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_official_regulations(
    regulation: str,
    regulator_selection: Optional[Sequence[str]] = None,
    *,
    max_results_per_query: Optional[int] = None,
    status: StatusCallback = _noop,
) -> Dict[str, Any]:
    """Search only approved regulator domains for the supplied ``regulation``.

    Parameters
    ----------
    regulation
        Free-form regulation label (``"DORA"``, ``"MiFID II"`` ...).
    regulator_selection
        UI-selected regulator codes (e.g. ``["EBA", "ESMA"]``) or ``None`` /
        ``["ALL"]`` to search every approved regulator.
    max_results_per_query
        Per-backend per-query limit. Defaults to
        :func:`search_config.regulatory_max_results`.
    status
        Optional progress callback (Streamlit ``st.status`` writer).

    Returns
    -------
    A dict with::

        {
            "results":     List[OfficialRegulationResult],
            "regulators":  List[{"code", "name", "jurisdiction", "website"}],
            "diagnostics": List[str],
            "queries":     List[{"query", "domain", "regulator_code"}],
            "errors":      List[str],
            "enabled":     bool,
        }
    """
    diagnostics: List[str] = []
    errors: List[str] = []
    regulators = resolve_regulators(regulator_selection)

    payload_regulators = [
        {"code": r.code, "name": r.name, "jurisdiction": r.jurisdiction, "website": r.website}
        for r in regulators
    ]

    if not regulators:
        diagnostics.append("No approved regulators matched the selection; nothing to search.")
        return {
            "results": [],
            "regulators": payload_regulators,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": is_regulatory_search_enabled(),
        }

    if not is_regulatory_search_enabled():
        diagnostics.append("Regulatory search disabled (REGULATORY_SEARCH_ENABLED=false).")
        status("Regulatory search disabled by env; Stage 1 returns no live results.")
        return {
            "results": [],
            "regulators": payload_regulators,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": False,
        }

    results: List[OfficialRegulationResult] = []
    seen_urls: set[str] = set()
    queries: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # PRIMARY PATH: native per-regulator site search.
    #
    # Some corporate networks block / mangle every third-party search
    # engine that DDGS can route through. Each regulator's own site search
    # is on a reachable domain we already trust, so we always try this
    # first. The DDGS fallback below only runs for regulators that don't
    # have a native adapter, or to top up results when native is sparse.
    # ------------------------------------------------------------------
    timeout = search_timeout_seconds()
    native_codes_used: List[str] = []
    native_started = time.monotonic()
    for reg in regulators:
        if reg.code not in native_supported_codes():
            continue
        native_codes_used.append(reg.code)
        try:
            native_hits = native_search(
                reg.code, regulation, max_results=regulatory_max_results(), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"native:{reg.code} -> {type(exc).__name__}: {exc}")
            status(f"Stage 1 native search error for {reg.code}: {exc}")
            continue
        for hit in native_hits:
            url = (hit.url or "").strip()
            if not url or url in seen_urls:
                continue
            if not is_regulator_url(url, regulators):
                continue
            seen_urls.add(url)
            actual_reg = regulator_for_url(url, regulators) or reg
            source_type = (
                SOURCE_TYPE_OFFICIAL_LEGISLATION
                if actual_reg.source_type == SOURCE_TYPE_OFFICIAL_LEGISLATION
                else SOURCE_TYPE_OFFICIAL_REGULATOR
            )
            composite_text = f"{hit.title}\n{hit.snippet}"
            result = OfficialRegulationResult(
                source_type=source_type,
                regulator=actual_reg.name,
                regulator_code=actual_reg.code,
                title=hit.title or url,
                url=url,
                snippet=hit.snippet,
                publication_type=_extract_publication_type(
                    composite_text, actual_reg.publication_hints
                ),
                regulation_id=_extract_regulation_id(composite_text),
                publication_date=_extract_publication_date(composite_text),
                version=None,
                executive_summary=(hit.snippet or "")[:600],
                backend=f"native:{reg.code.lower()}",
                query=regulation,
            )
            result.confidence_score = _score_confidence(result, regulation)
            results.append(result)

    if native_codes_used:
        status(
            f"Stage 1 native search complete: tried {len(native_codes_used)} "
            f"regulator(s) ({', '.join(native_codes_used)}) -> "
            f"{len(results)} hit(s) in {time.monotonic() - native_started:.1f}s."
        )

    # Skip DDGS fallback if EITHER:
    #   (a) every selected regulator has a native adapter -- DDGS can only
    #       duplicate or add noise, never expand coverage; OR
    #   (b) native search alone already cleared the early-stop threshold.
    # This keeps locked-down corporate networks free of DDGS errors when
    # native results already cover the user's selection.
    caps_early = _runtime_caps()["early_stop_results"]
    selected_codes = {r.code for r in regulators}
    native_codes = set(native_supported_codes())
    all_selected_are_native = selected_codes and selected_codes.issubset(native_codes)
    skip_fallback_reason: Optional[str] = None
    if all_selected_are_native:
        skip_fallback_reason = (
            f"every selected regulator ({', '.join(sorted(selected_codes))}) "
            f"has a native adapter"
        )
    elif len(results) >= caps_early:
        skip_fallback_reason = (
            f"{len(results)} native hits >= {caps_early} early-stop threshold"
        )

    if skip_fallback_reason:
        diagnostics.append(
            f"Stage 1 satisfied by native search ({skip_fallback_reason}). "
            f"Skipping DDGS fallback."
        )
        status(diagnostics[-1])
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        return {
            "results": results,
            "regulators": payload_regulators,
            "diagnostics": diagnostics,
            "queries": queries,
            "errors": errors,
            "enabled": True,
        }

    # ------------------------------------------------------------------
    # FALLBACK PATH: DDGS (DuckDuckGo / Brave / etc.).
    # ------------------------------------------------------------------
    if DDGS is None:
        msg = "DDGS / duckduckgo_search not installed; Stage 1 native results only."
        diagnostics.append(msg)
        status(msg)
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        return {
            "results": results,
            "regulators": payload_regulators,
            "diagnostics": diagnostics,
            "queries": queries,
            "errors": errors,
            "enabled": True,
        }

    queries = build_queries(regulation, regulators)
    if not queries:
        diagnostics.append("No regulation label supplied; Stage 1 cannot build queries.")
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        return {
            "results": results,
            "regulators": payload_regulators,
            "diagnostics": diagnostics,
            "queries": [],
            "errors": errors,
            "enabled": True,
        }

    backends = search_backends()
    max_per_query = max_results_per_query or regulatory_max_results()
    caps = _runtime_caps()

    # Apply the hard cap on the dispatched query count *before* we start so
    # the user sees the actual planned workload in the status message.
    if len(queries) > caps["max_total_queries"]:
        queries = queries[: caps["max_total_queries"]]

    delay_seconds = max(0.0, caps["inter_query_delay_ms"] / 1000.0)
    status(
        f"Stage 1 DDGS fallback: running {len(queries)} sequential query/queries "
        f"across {len(regulators)} regulator(s) using backends `{','.join(backends) or 'duckduckgo'}` "
        f"(early-stop at {caps['early_stop_results']} hits, max {caps['max_total_seconds']}s, "
        f"{delay_seconds:.2f}s between queries)."
    )

    started = time.monotonic()
    deadline = started + caps["max_total_seconds"]

    for idx, q in enumerate(queries):
        if time.monotonic() > deadline:
            status("Stage 1: wall-clock cap reached, stopping query dispatch.")
            break
        if len(seen_urls) >= caps["early_stop_results"]:
            status(f"Stage 1: early-stop reached ({len(seen_urls)} hits), skipping remaining queries.")
            break

        query_text = q["query"]
        # ``regulator_code`` is optional now -- plain (regulator-agnostic)
        # queries leave it blank. When present we use it as a friendly status
        # label; the actual regulator assigned to each kept URL is resolved
        # from the URL itself via ``regulator_for_url``.
        biased_regulator = _REG_BY_CODE.get(q.get("regulator_code") or "")

        hits: List[Dict[str, Any]] = []
        for backend in backends:
            if time.monotonic() > deadline:
                break
            try:
                hits = _ddgs_text(
                    query_text, backend=backend, max_results=max_per_query, timeout=timeout
                )
            except Exception as exc:  # noqa: BLE001
                err = f"backend=`{backend}` query=`{query_text[:80]}...` -> {type(exc).__name__}: {exc}"
                errors.append(err)
                status(f"Stage 1 backend error: {err}")
                hits = []
                continue
            if hits:
                break

        kept = 0
        for hit in hits or []:
            url = (hit.get("href") or hit.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            if not is_regulator_url(url, regulators):
                continue
            seen_urls.add(url)
            kept += 1

            title = (hit.get("title") or "").strip()
            snippet = (hit.get("body") or hit.get("snippet") or "").strip()
            actual_reg = regulator_for_url(url, regulators) or biased_regulator
            if actual_reg is None:
                # Defensive: ``is_regulator_url`` already passed, so this
                # should be unreachable. Drop the hit if we somehow can't
                # map it back to a known regulator.
                continue
            source_type = (
                SOURCE_TYPE_OFFICIAL_LEGISLATION
                if actual_reg.source_type == SOURCE_TYPE_OFFICIAL_LEGISLATION
                else SOURCE_TYPE_OFFICIAL_REGULATOR
            )

            composite_text = f"{title}\n{snippet}"
            result = OfficialRegulationResult(
                source_type=source_type,
                regulator=actual_reg.name,
                regulator_code=actual_reg.code,
                title=title or url,
                url=url,
                snippet=snippet,
                publication_type=_extract_publication_type(composite_text, actual_reg.publication_hints),
                regulation_id=_extract_regulation_id(composite_text),
                publication_date=_extract_publication_date(composite_text),
                version=None,
                executive_summary=snippet[:600],
                backend=backends[0] if backends else "",
                query=query_text,
            )
            result.confidence_score = _score_confidence(result, regulation)
            results.append(result)

        if kept:
            label = (
                biased_regulator.code
                if biased_regulator is not None
                else "all selected regulators"
            )
            status(
                f"Stage 1: query {idx + 1}/{len(queries)} ({label}) -> "
                f"{kept} new hit(s) (total unique URLs: {len(seen_urls)})."
            )

        # Polite delay between queries to avoid DDGS rate-limiting.
        if delay_seconds > 0 and idx < len(queries) - 1:
            time.sleep(delay_seconds)

    results.sort(key=lambda r: r.confidence_score, reverse=True)
    elapsed = time.monotonic() - started

    diagnostics.append(
        f"Stage 1 returned {len(results)} approved-domain publication(s) "
        f"across {len(seen_urls)} unique URLs in {elapsed:.1f}s."
    )
    status(diagnostics[-1])

    return {
        "results": results,
        "regulators": payload_regulators,
        "diagnostics": diagnostics,
        "queries": queries,
        "errors": errors,
        "enabled": True,
    }


# Cache of code -> RegulatorSource used by ``fetch_official_regulations``.
_REG_BY_CODE: Dict[str, RegulatorSource] = {r.code: r for r in APPROVED_REGULATORS}


def _ddgs_text(query: str, *, backend: str, max_results: int, timeout: int) -> List[Dict[str, Any]]:
    """Thin wrapper around ``DDGS().text`` tolerant of API/version skew."""
    assert DDGS is not None  # narrowed by caller
    with DDGS(timeout=timeout) as ddgs:
        try:
            return list(ddgs.text(query, max_results=max_results, backend=backend) or [])
        except TypeError:
            return list(ddgs.text(query, max_results=max_results) or [])


__all__ = [
    "OfficialRegulationResult",
    "build_queries",
    "fetch_official_regulations",
]
