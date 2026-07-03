"""Native HTTP search adapters for approved regulator domains.

DDGS / DuckDuckGo / Brave / Yandex etc. can be unreachable on locked-down
corporate networks (firewall blocks, DNS-resolver bugs, content filtering).
Each regulator publishes its **own** search page on its own domain, which is
always reachable wherever the regulator's website itself is reachable. This
module hits those native search endpoints directly with plain ``httpx`` and
parses the HTML for result anchors that:

  1. Stay on the same approved domain (no off-site links), and
  2. Mention the search term in either the URL path or the anchor text.

This is strictly less complete than a general-purpose web search, but it
returns first-party authoritative content -- which is exactly what Stage 1
of the regulatory pipeline is meant to deliver.

To add a new regulator, drop in a new function and decorate it with
``@register("CODE")`` -- nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class NativeSearchHit:
    """One result from a regulator's native site-search."""

    url: str
    title: str
    snippet: str = ""


SearchAdapter = Callable[[str, int, int], List[NativeSearchHit]]
_ADAPTERS: Dict[str, SearchAdapter] = {}


def register(code: str) -> Callable[[SearchAdapter], SearchAdapter]:
    """Register a native search adapter for ``code`` (e.g. ``"EBA"``)."""

    def _decorator(fn: SearchAdapter) -> SearchAdapter:
        _ADAPTERS[code] = fn
        return fn

    return _decorator


def native_search(
    regulator_code: str,
    query: str,
    *,
    max_results: int = 8,
    timeout: int = 12,
) -> List[NativeSearchHit]:
    """Return search hits from a regulator's native search, or [] if unsupported."""

    adapter = _ADAPTERS.get(regulator_code)
    if adapter is None:
        return []
    try:
        return adapter(query, max_results, timeout)
    except Exception:
        # Adapters should swallow their own errors; this is the last-resort
        # safety net so one regulator's failure never breaks Stage 1 as a whole.
        return []


def supported_regulator_codes() -> List[str]:
    """List the regulator codes for which a native adapter is registered."""

    return list(_ADAPTERS.keys())


def _fetch(url: str, timeout: int) -> Optional[str]:
    """Fetch ``url`` with the default browser-ish headers; return body or None."""

    try:
        response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    except Exception:
        return None
    # Several regulator search pages (notably ESMA / EIOPA) respond with
    # HTTP 404 even when the body contains valid results -- they treat the
    # search route as "not a real document". Anything but a 5xx is fair game.
    if response.status_code >= 500:
        return None
    return response.text


# URL substrings that mark a link as navigation / pagination / facet noise,
# error pages, or other non-result chrome. Anchors hitting any of these are
# unconditionally dropped.
_DROP_HREF_TOKENS = (
    "?f%5b", "?f[0]=", "&f%5b", "&f[0]=",
    "#main", "#nav", "#search", "#content", "#top",
    "/search?", "/search-results?",
    "&start=", "?start=",
    "javascript:", "mailto:",
    "/sitemap", "/contact", "/accessibility", "/cookie",
    "/rss", "/feed", "/login", "/register",
    "/error/", "/error_", "/404", "errorpage",
)

# Titles that always indicate chrome / navigation rather than a real result.
_DROP_TITLE_LOWERCASE = {
    "main menu", "skip to content", "search", "home", "back to top",
    "accessibility feedback", "page 1", "page 2", "page 3", "page 4",
    "page 5", "page 6", "page 7", "previous", "next", "first", "last",
    "more", "rss", "feed", "documents", "press releases", "events",
    "publications", "guidelines", "news", "speeches", "consultations",
    "current", "page 1 (current)", "english", "deutsch", "français",
    "extranet", "log in", "learn more", "read more", "find out more",
}

# Language-switcher labels frequently emitted by Drupal sites (e.g. EIOPA).
_LANGUAGE_LABEL = re.compile(
    r"^[a-z]{2}\s+([a-zé\u0080-\uffff]+)$", re.IGNORECASE
)

# Small glossary so abbreviations match expanded forms in result titles/URLs.
# When the user queries "DORA" the canonical EBA page is titled
# "Digital operational resilience Act" -- the acronym never appears literally
# in the title, so we have to expand the query.
_REGULATION_SYNONYMS: Dict[str, List[str]] = {
    "dora": ["digital operational resilience", "digital-operational-resilience", "2022/2554", "32022r2554"],
    "mifid": ["markets in financial instruments", "mifid"],
    "mifid ii": ["markets in financial instruments", "mifid"],
    "gdpr": ["general data protection", "data protection regulation"],
    "aml": ["anti-money laundering", "money laundering"],
    "amld": ["anti-money laundering", "money laundering"],
    "emir": ["european market infrastructure", "emir"],
    "mar": ["market abuse regulation", "market abuse"],
    "csrd": ["corporate sustainability reporting", "sustainability reporting"],
    "crr": ["capital requirements regulation", "capital requirements"],
    "crd": ["capital requirements directive", "capital requirements"],
    "mica": ["markets in crypto-assets", "crypto-assets", "mica"],
    "nis2": ["network and information security", "nis 2", "nis2"],
    "nisd": ["network and information security"],
    "dsa": ["digital services act"],
    "dma": ["digital markets act"],
    "brrd": ["bank recovery and resolution", "recovery and resolution"],
    "psd2": ["payment services directive", "payment services"],
    "bmr": ["benchmark regulation", "benchmarks regulation"],
}


def _relevance_terms(query: str) -> List[str]:
    """Expand the query into a list of lowercase search-friendly terms.

    Combines the raw query, its kebab-case form, and any synonyms from
    :data:`_REGULATION_SYNONYMS`. We later require *at least one* of these
    terms to appear in the result URL or title.
    """

    terms: List[str] = []
    q = (query or "").strip().lower()
    if not q:
        return terms

    terms.append(q)
    kebab = re.sub(r"\s+", "-", q)
    if kebab != q:
        terms.append(kebab)

    # Per-word terms with >= 3 chars (lets "MiFID II" match results titled
    # with just "MiFID").
    for word in re.split(r"\s+", q):
        word = word.strip()
        if len(word) >= 3 and word not in terms:
            terms.append(word)

    # Synonyms from the glossary -- check both full-query and per-word keys.
    keys_to_check = {q}
    keys_to_check.update(re.split(r"\s+", q))
    for key in keys_to_check:
        for syn in _REGULATION_SYNONYMS.get(key.strip().lower(), ()):
            syn_l = syn.lower()
            if syn_l not in terms:
                terms.append(syn_l)

    return terms


def _parse_drupal_style_results(
    html: str,
    domain: str,
    query: str,
    max_results: int,
) -> List[NativeSearchHit]:
    """Extract anchors that look like organic search-result rows.

    Different regulators emit search results with different markup (some use
    ``<h2>``, some plain ``<li><a>``, some ``<div class="search-result">``).
    Rather than maintain N different selectors, we match every anchor on the
    regulator's domain and filter:

      * URL must be on ``domain`` (no off-site links).
      * URL must not hit any nav/facet/pagination/error substring in
        :data:`_DROP_HREF_TOKENS`.
      * URL path must be deep enough to be an article (>= 2 path segments)
        OR the anchor must carry Drupal's ``data-drupal-link-system-path``
        attribute (positive content-link signal).
      * Title text must be non-trivial (>= 8 chars after stripping HTML)
        and must not be a known chrome label or a language-switcher entry.

    Relevance gate: at least one of the query terms expanded by
    :func:`_relevance_terms` (raw query, kebab form, per-word, and known
    abbreviation synonyms like ``"dora" -> "digital operational resilience"``)
    must appear in the result URL or title.
    """

    relevance_terms = _relevance_terms(query)
    seen: set[str] = set()
    out: List[NativeSearchHit] = []

    # Capture the whole anchor tag (open + body) so we can inspect both
    # the attribute string and the inner text in one pass.
    pattern = re.compile(
        r'<a\b(?P<attrs>[^>]*)>(?P<body>.{4,400}?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    href_attr = re.compile(r'\bhref="([^"]+)"', re.IGNORECASE)

    for match in pattern.finditer(html):
        attrs = match.group("attrs") or ""
        href_match = href_attr.search(attrs)
        if not href_match:
            continue

        href = href_match.group(1).replace("&amp;", "&").strip()
        if not href:
            continue
        if href.startswith("/"):
            href = f"https://{domain}{href}"
        elif not href.startswith("http"):
            continue
        if domain not in href:
            continue
        href_lower = href.lower()
        if any(token in href_lower for token in _DROP_HREF_TOKENS):
            continue

        # Strip inline tags (icons, spans, bold) and collapse whitespace.
        title = re.sub(r"<[^>]+>", " ", match.group("body"))
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) < 8:
            continue
        title_lower = title.lower()
        if title_lower in _DROP_TITLE_LOWERCASE:
            continue
        # Drop language-switcher entries like "bg български", "es español".
        if _LANGUAGE_LABEL.match(title_lower):
            continue

        # Positive content-link signals:
        #   - Drupal explicitly marks content nodes with this attribute.
        #   - "Real" article URLs typically have >= 2 path segments after
        #     the domain (e.g. /activities/dora). Top-level chrome pages
        #     like /about-us, /extranet have only 1.
        has_drupal_marker = "data-drupal-link-system-path" in attrs.lower()
        path_segments = [
            s for s in href[len(f"https://{domain}"):].split("?")[0].split("/")
            if s
        ]
        deep_enough = len(path_segments) >= 2
        if not (has_drupal_marker or deep_enough):
            continue

        # Relevance gate: URL or title must mention one of the expanded
        # query terms (handles abbreviations -> expanded names).
        if relevance_terms:
            blob = (title + " " + href).lower()
            if not any(term in blob for term in relevance_terms):
                continue

        if href in seen:
            continue
        seen.add(href)
        out.append(NativeSearchHit(url=href, title=title))
        if len(out) >= max_results:
            break

    return out


# ---------------------------------------------------------------------------
# Per-regulator adapters. Add new ones below; they auto-register.
# ---------------------------------------------------------------------------


@register("EBA")
def _search_eba(query: str, max_results: int, timeout: int) -> List[NativeSearchHit]:
    url = f"https://www.eba.europa.eu/search?keywords={quote_plus(query)}"
    html = _fetch(url, timeout)
    return _parse_drupal_style_results(html, "eba.europa.eu", query, max_results) if html else []


@register("ESMA")
def _search_esma(query: str, max_results: int, timeout: int) -> List[NativeSearchHit]:
    url = f"https://www.esma.europa.eu/search-results?keys={quote_plus(query)}"
    html = _fetch(url, timeout)
    return _parse_drupal_style_results(html, "esma.europa.eu", query, max_results) if html else []


@register("EIOPA")
def _search_eiopa(query: str, max_results: int, timeout: int) -> List[NativeSearchHit]:
    url = f"https://www.eiopa.europa.eu/search?keys={quote_plus(query)}"
    html = _fetch(url, timeout)
    return _parse_drupal_style_results(html, "eiopa.europa.eu", query, max_results) if html else []


@register("FCA")
def _search_fca(query: str, max_results: int, timeout: int) -> List[NativeSearchHit]:
    url = f"https://www.fca.org.uk/search-results?search_term={quote_plus(query)}"
    html = _fetch(url, timeout)
    return _parse_drupal_style_results(html, "fca.org.uk", query, max_results) if html else []


__all__ = [
    "NativeSearchHit",
    "native_search",
    "supported_regulator_codes",
]
