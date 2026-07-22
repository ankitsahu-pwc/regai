"""Configuration for the hierarchical Regulatory Intelligence Pipeline.

This module is the single source of truth for **which websites the application
is allowed to search**. The Regulatory Impact & Readiness Assessment platform
must only consume content from trusted regulatory authorities (Stage 1) and
approved consulting firms (Stage 2). Generic internet search engines, blogs,
Wikipedia, news outlets and discussion forums are intentionally out of scope.

Adding a new regulator or a new consulting firm should only require editing
this file. None of the search logic in
:mod:`services.official_regulation_fetcher`,
:mod:`services.consulting_guidance_fetcher` or
:mod:`services.regulatory_intelligence_service` needs to change.

Public API
----------
* :class:`RegulatorSource`   – metadata for one approved regulator.
* :class:`ConsultingSource`  – metadata for one approved consulting firm.
* :data:`APPROVED_REGULATORS`           – ordered list of all approved
  regulator sources.
* :data:`APPROVED_CONSULTING_FIRMS`     – ordered list of all approved
  consulting firms.
* :data:`SOURCE_PRIORITY`               – global source-type ranking.
* :data:`PUBLICATION_TYPES`             – the regulatory publication taxonomy
  the fetchers will try to identify.
* :func:`resolve_regulators`            – translate a UI selection
  (codes / "All") into a list of :class:`RegulatorSource`.
* :func:`regulator_domains`             – flat list of allowed regulator
  hostnames.
* :func:`consulting_domains`            – flat list of allowed consulting
  hostnames.
* :func:`is_regulator_url` /
  :func:`is_consulting_url`             – guards used by the fetchers to drop
  any URL that leaks in from outside the allow-list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Source-type taxonomy
# ---------------------------------------------------------------------------

#: Document classification used throughout the pipeline. Stored on every
#: retrieved publication so downstream agents can reason about provenance.
SOURCE_TYPE_OFFICIAL_REGULATOR = "Official Regulator"
SOURCE_TYPE_OFFICIAL_LEGISLATION = "Official Legislation"
SOURCE_TYPE_CONSULTING_GUIDANCE = "Consulting Guidance"

#: Global ordered priority. The Regulatory Intelligence Service applies this
#: ordering when merging documents so consulting guidance can never override
#: official content.
SOURCE_PRIORITY: List[str] = [
    SOURCE_TYPE_OFFICIAL_REGULATOR,
    SOURCE_TYPE_OFFICIAL_LEGISLATION,
    SOURCE_TYPE_CONSULTING_GUIDANCE,
]

#: Publication taxonomy the official regulator fetcher will try to detect
#: in the page title / snippet. The values are matched case-insensitively.
PUBLICATION_TYPES: List[str] = [
    "Regulation",
    "Directive",
    "RTS",                    # Regulatory Technical Standard
    "ITS",                    # Implementing Technical Standard
    "Guideline",
    "Technical Standard",
    "Q&A",
    "Consultation Paper",
    "Supervisory Statement",
    "Enforcement Publication",
    "Opinion",
    "Recommendation",
]


# ---------------------------------------------------------------------------
# Approved regulator catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegulatorSource:
    """One approved regulator authority.

    ``code`` is a short, UI-friendly identifier used by the regulator
    selector (``"EBA"``, ``"FCA"`` ...).

    ``domains`` is the list of hostnames that are considered authoritative
    for this regulator. The fetcher will *only* keep search results whose URL
    matches one of these domains.

    ``source_type`` is either :data:`SOURCE_TYPE_OFFICIAL_REGULATOR` or
    :data:`SOURCE_TYPE_OFFICIAL_LEGISLATION` (used for EUR-Lex which is the
    legislation publisher rather than a supervisory authority).
    """

    code: str
    name: str
    jurisdiction: str
    website: str
    domains: Sequence[str]
    source_type: str = SOURCE_TYPE_OFFICIAL_REGULATOR
    description: str = ""
    publication_hints: Sequence[str] = field(default_factory=tuple)

    def matches(self, url: str) -> bool:
        """Return True if ``url``'s hostname is on this regulator's allow-list."""
        host = _hostname(url)
        if not host:
            return False
        return any(host == d or host.endswith("." + d) for d in self.domains)


APPROVED_REGULATORS: List[RegulatorSource] = [
    # ---- European Union ----------------------------------------------------
    RegulatorSource(
        code="EBA",
        name="European Banking Authority",
        jurisdiction="European Union",
        website="https://www.eba.europa.eu",
        domains=("eba.europa.eu",),
        description="EU banking regulator. Owns RTS/ITS, guidelines and Q&A for banking and DORA.",
        publication_hints=("regulatory technical standards", "guidelines", "consultation paper", "Q&A"),
    ),
    RegulatorSource(
        code="ESMA",
        name="European Securities and Markets Authority",
        jurisdiction="European Union",
        website="https://www.esma.europa.eu",
        domains=("esma.europa.eu",),
        description="EU securities and markets regulator. Owns MiFID II, MAR, EMIR technical standards.",
        publication_hints=("technical standards", "guidelines", "Q&A", "opinion"),
    ),
    RegulatorSource(
        code="ECB",
        name="European Central Bank",
        jurisdiction="European Union",
        website="https://www.ecb.europa.eu",
        domains=("ecb.europa.eu",),
        description="EU monetary policy authority. Publishes supervisory guides and opinions.",
        publication_hints=("guide", "regulation", "opinion"),
    ),
    RegulatorSource(
        code="SSM",
        name="ECB Banking Supervision (SSM)",
        jurisdiction="European Union",
        website="https://www.bankingsupervision.europa.eu",
        domains=("bankingsupervision.europa.eu",),
        description="Single Supervisory Mechanism. Supervisory expectations for significant institutions.",
        publication_hints=("supervisory expectations", "guide", "SREP"),
    ),
    RegulatorSource(
        code="EIOPA",
        name="European Insurance and Occupational Pensions Authority",
        jurisdiction="European Union",
        website="https://www.eiopa.europa.eu",
        domains=("eiopa.europa.eu",),
        description="EU insurance and pensions regulator. Solvency II RTS/ITS and DORA insurance scope.",
        publication_hints=("technical standards", "guidelines", "opinion"),
    ),
    RegulatorSource(
        code="SRB",
        name="Single Resolution Board",
        jurisdiction="European Union",
        website="https://www.srb.europa.eu",
        domains=("srb.europa.eu",),
        description="EU bank resolution authority. Publishes resolution policies and expectations.",
        publication_hints=("expectations", "policy", "guidance"),
    ),
    RegulatorSource(
        code="AMLA",
        name="Anti-Money Laundering Authority",
        jurisdiction="European Union",
        website="https://www.amla.europa.eu",
        domains=("amla.europa.eu",),
        description="EU AML/CFT supervisor. Publishes AML supervisory standards.",
        publication_hints=("standards", "guidelines"),
    ),
    RegulatorSource(
        code="DG_FISMA",
        name="European Commission - DG FISMA",
        jurisdiction="European Union",
        website="https://finance.ec.europa.eu",
        domains=("finance.ec.europa.eu",),
        description="EU Commission Directorate-General for Financial Stability, Financial Services and Capital Markets Union.",
        publication_hints=("delegated act", "implementing act", "communication", "report"),
    ),
    RegulatorSource(
        code="EUR_LEX",
        name="EUR-Lex (Official EU Legislation)",
        jurisdiction="European Union",
        website="https://eur-lex.europa.eu",
        domains=("eur-lex.europa.eu",),
        source_type=SOURCE_TYPE_OFFICIAL_LEGISLATION,
        description="Official Journal of the EU. Authoritative source for regulations, directives and delegated acts.",
        publication_hints=("regulation", "directive", "delegated regulation", "implementing regulation"),
    ),

    # ---- United Kingdom ----------------------------------------------------
    RegulatorSource(
        code="FCA",
        name="Financial Conduct Authority",
        jurisdiction="United Kingdom",
        website="https://www.fca.org.uk",
        domains=("fca.org.uk",),
        description="UK conduct regulator. Owns the FCA Handbook, policy statements and consultation papers.",
        publication_hints=("policy statement", "consultation paper", "handbook", "finalised guidance"),
    ),
    RegulatorSource(
        code="PRA",
        name="Prudential Regulation Authority",
        jurisdiction="United Kingdom",
        website="https://www.bankofengland.co.uk/prudential-regulation",
        domains=("bankofengland.co.uk",),
        description="UK prudential regulator hosted within the Bank of England. Publishes supervisory statements and policy statements.",
        publication_hints=("supervisory statement", "policy statement", "consultation paper"),
    ),

    # ---- National regulators ----------------------------------------------
    RegulatorSource(
        code="BAFIN",
        name="BaFin",
        jurisdiction="Germany",
        website="https://www.bafin.de",
        domains=("bafin.de",),
        description="German Federal Financial Supervisory Authority. Publishes circulars (Rundschreiben), guidance notices and minimum requirements (MaRisk, BAIT).",
        publication_hints=("rundschreiben", "circular", "merkblatt", "guidance notice"),
    ),
    RegulatorSource(
        code="AMF_FR",
        name="Autorite des Marches Financiers (France)",
        jurisdiction="France",
        website="https://www.amf-france.org",
        domains=("amf-france.org",),
        description="French financial markets regulator. Publishes general regulation, positions and recommendations.",
        publication_hints=("position", "recommandation", "general regulation"),
    ),
    RegulatorSource(
        code="CBI",
        name="Central Bank of Ireland",
        jurisdiction="Ireland",
        website="https://www.centralbank.ie",
        domains=("centralbank.ie",),
        description="Irish central bank and financial regulator. Publishes guidance, codes and Dear CEO letters.",
        publication_hints=("guidance", "code", "Dear CEO letter", "consultation paper"),
    ),
    RegulatorSource(
        code="DNB",
        name="De Nederlandsche Bank",
        jurisdiction="Netherlands",
        website="https://www.dnb.nl",
        domains=("dnb.nl",),
        description="Dutch central bank and prudential supervisor. Publishes Good Practices, guidance and Q&A.",
        publication_hints=("good practice", "Q&A", "guidance"),
    ),

    # ---- India -------------------------------------------------------------
    # Indian regulator cluster (banking, housing finance, real-estate,
    # capital markets, AML/CFT, insolvency and competition). Priority
    # (as documented for the Housing Finance / Mortgage lending scope):
    #   RBI            *****
    #   NHB            ****
    #   State RERA     ****
    #   MoHUA          ***
    #   CERSAI         ***
    #   DFS (MoF)      ***
    #   SEBI           **
    #   FIU-IND        **
    #   IBBI           **
    #   CCI            *
    RegulatorSource(
        code="RBI",
        name="Reserve Bank of India",
        jurisdiction="India",
        website="https://www.rbi.org.in",
        domains=("rbi.org.in",),
        description="Prudential regulation of banks and HFCs, lending norms, provisioning, LTV, interest rates, risk weights.",
        publication_hints=(
            "master direction", "master circular", "notification",
            "circular", "guidelines", "press release",
        ),
    ),
    RegulatorSource(
        code="NHB",
        name="National Housing Bank",
        jurisdiction="India",
        website="https://www.nhb.org.in",
        domains=("nhb.org.in",),
        description="Supervises HFCs, refinance schemes, housing finance guidance and sector development.",
        publication_hints=(
            "policy circular", "guidelines", "notification",
            "refinance scheme", "master circular",
        ),
    ),
    RegulatorSource(
        code="RERA",
        name="State RERA Authorities",
        jurisdiction="India",
        website="https://rera.gov.in",
        # ``RERA`` is state-scoped in India; we include the central
        # portal plus the most active state authorities so the allow-list
        # covers the majority of published orders / regulations. Missing
        # states can be added incrementally.
        domains=(
            "rera.gov.in",
            "maharera.mahaonline.gov.in",
            "maharera.maharashtra.gov.in",
            "up-rera.in",
            "rera.karnataka.gov.in",
            "rera.telangana.gov.in",
            "gujrera.gujarat.gov.in",
            "tnrera.in",
            "haryanarera.gov.in",
            "hprera.in",
            "rera.rajasthan.gov.in",
            "wbhira.in",
            "rera.mp.gov.in",
            "rera.kerala.gov.in",
            "rera.odisha.gov.in",
        ),
        description="State-level regulation of real-estate projects, developer registration, escrow requirements and homebuyer protection.",
        publication_hints=("order", "regulation", "circular", "notification"),
    ),
    RegulatorSource(
        code="MOHUA",
        name="Ministry of Housing & Urban Affairs",
        jurisdiction="India",
        website="https://mohua.gov.in",
        domains=("mohua.gov.in", "pmaymis.gov.in", "pmay-urban.gov.in"),
        description="Housing policies, PMAY and affordable housing schemes.",
        publication_hints=("scheme guidelines", "notification", "circular", "office memorandum"),
    ),
    RegulatorSource(
        code="CERSAI",
        name="CERSAI",
        jurisdiction="India",
        website="https://www.cersai.org.in",
        domains=("cersai.org.in",),
        description="Central registry of mortgages / security interests under SARFAESI; critical for mortgage due diligence and fraud prevention.",
        publication_hints=("notification", "circular", "user manual", "operating guidelines"),
    ),
    RegulatorSource(
        code="DFS",
        name="Department of Financial Services (Ministry of Finance)",
        jurisdiction="India",
        website="https://financialservices.gov.in",
        domains=("financialservices.gov.in",),
        description="Policy announcements affecting banks, HFCs and NHB (Ministry of Finance, Department of Financial Services).",
        publication_hints=("office memorandum", "notification", "press release", "gazette notification"),
    ),
    RegulatorSource(
        code="SEBI",
        name="Securities and Exchange Board of India",
        jurisdiction="India",
        website="https://www.sebi.gov.in",
        domains=("sebi.gov.in",),
        description="REITs, mortgage-backed securities and listed HFC disclosures.",
        publication_hints=("circular", "regulations", "master circular", "notification", "consultation paper"),
    ),
    RegulatorSource(
        code="FIU_IND",
        name="Financial Intelligence Unit - India",
        jurisdiction="India",
        website="https://fiuindia.gov.in",
        domains=("fiuindia.gov.in",),
        description="AML/CFT obligations for reporting entities including mortgage lenders.",
        publication_hints=("guidelines", "advisory", "notification", "typology"),
    ),
    RegulatorSource(
        code="IBBI",
        name="Insolvency and Bankruptcy Board of India",
        jurisdiction="India",
        website="https://ibbi.gov.in",
        domains=("ibbi.gov.in",),
        description="Insolvency proceedings impacting developers and secured creditors.",
        publication_hints=("regulations", "circular", "notification", "order"),
    ),
    RegulatorSource(
        code="CCI",
        name="Competition Commission of India",
        jurisdiction="India",
        website="https://www.cci.gov.in",
        domains=("cci.gov.in",),
        description="M&A and competition matters involving banks and HFCs.",
        publication_hints=("regulations", "order", "notification", "guidance note"),
    ),
]


# Lookup helpers --------------------------------------------------------------

_REGULATOR_BY_CODE = {r.code: r for r in APPROVED_REGULATORS}


def regulator_codes() -> List[str]:
    """Ordered list of every supported regulator code (UI dropdown source)."""
    return [r.code for r in APPROVED_REGULATORS]


def get_regulator(code: str) -> Optional[RegulatorSource]:
    return _REGULATOR_BY_CODE.get((code or "").upper())


def resolve_regulators(selection: Optional[Iterable[str]]) -> List[RegulatorSource]:
    """Translate a UI selection into :class:`RegulatorSource` objects.

    Accepts:
      * ``None`` or empty iterable -> every approved regulator (the
        "Search across all regulators" UX option).
      * ``["ALL"]`` -> every approved regulator (case-insensitive synonym).
      * ``["EBA", "ESMA", ...]`` -> the matching subset, preserving order.

    Unknown codes are silently dropped; the caller can compare the result
    length to its input to detect bad codes if needed.
    """
    if not selection:
        return list(APPROVED_REGULATORS)
    codes = [str(c).strip().upper() for c in selection if str(c).strip()]
    if not codes or "ALL" in codes:
        return list(APPROVED_REGULATORS)
    seen: set[str] = set()
    out: List[RegulatorSource] = []
    for code in codes:
        reg = _REGULATOR_BY_CODE.get(code)
        if reg and reg.code not in seen:
            out.append(reg)
            seen.add(reg.code)
    return out


def regulator_domains(regulators: Optional[Sequence[RegulatorSource]] = None) -> List[str]:
    """Flat list of every allowed hostname (used to construct ``site:`` queries)."""
    pool = regulators if regulators is not None else APPROVED_REGULATORS
    domains: List[str] = []
    seen: set[str] = set()
    for r in pool:
        for d in r.domains:
            if d not in seen:
                domains.append(d)
                seen.add(d)
    return domains


# ---------------------------------------------------------------------------
# Approved consulting catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsultingSource:
    """One approved consulting firm whose publications enrich Stage 2."""

    code: str
    name: str
    website: str
    domains: Sequence[str]
    source_type: str = SOURCE_TYPE_CONSULTING_GUIDANCE

    def matches(self, url: str) -> bool:
        host = _hostname(url)
        if not host:
            return False
        return any(host == d or host.endswith("." + d) for d in self.domains)


APPROVED_CONSULTING_FIRMS: List[ConsultingSource] = [
    ConsultingSource("PWC",        "PwC",            "https://www.pwc.com",         ("pwc.com",)),
    ConsultingSource("DELOITTE",   "Deloitte",       "https://www.deloitte.com",    ("deloitte.com",)),
    ConsultingSource("EY",         "EY",             "https://www.ey.com",          ("ey.com",)),
    ConsultingSource("KPMG",       "KPMG",           "https://kpmg.com",            ("kpmg.com",)),
    ConsultingSource("ACCENTURE",  "Accenture",      "https://www.accenture.com",   ("accenture.com",)),
    ConsultingSource("CAPGEMINI",  "Capgemini",      "https://www.capgemini.com",   ("capgemini.com",)),
    ConsultingSource("MCKINSEY",   "McKinsey & Co.", "https://www.mckinsey.com",    ("mckinsey.com",)),
    ConsultingSource("BCG",        "Boston Consulting Group", "https://www.bcg.com", ("bcg.com",)),
    ConsultingSource("OLIVER_WYMAN", "Oliver Wyman", "https://www.oliverwyman.com", ("oliverwyman.com",)),
    ConsultingSource("BAIN",       "Bain & Co.",     "https://www.bain.com",        ("bain.com",)),
]

_CONSULTING_BY_CODE = {c.code: c for c in APPROVED_CONSULTING_FIRMS}


def consulting_codes() -> List[str]:
    return [c.code for c in APPROVED_CONSULTING_FIRMS]


def get_consulting(code: str) -> Optional[ConsultingSource]:
    return _CONSULTING_BY_CODE.get((code or "").upper())


def resolve_consulting_firms(selection: Optional[Iterable[str]]) -> List[ConsultingSource]:
    """Translate a UI selection into :class:`ConsultingSource` objects.

    Mirrors :func:`resolve_regulators`. Empty / ``None`` / ``"ALL"`` selects
    every approved firm.
    """
    if not selection:
        return list(APPROVED_CONSULTING_FIRMS)
    codes = [str(c).strip().upper() for c in selection if str(c).strip()]
    if not codes or "ALL" in codes:
        return list(APPROVED_CONSULTING_FIRMS)
    seen: set[str] = set()
    out: List[ConsultingSource] = []
    for code in codes:
        firm = _CONSULTING_BY_CODE.get(code)
        if firm and firm.code not in seen:
            out.append(firm)
            seen.add(firm.code)
    return out


def consulting_domains(firms: Optional[Sequence[ConsultingSource]] = None) -> List[str]:
    pool = firms if firms is not None else APPROVED_CONSULTING_FIRMS
    domains: List[str] = []
    seen: set[str] = set()
    for f in pool:
        for d in f.domains:
            if d not in seen:
                domains.append(d)
                seen.add(d)
    return domains


# ---------------------------------------------------------------------------
# URL guards
# ---------------------------------------------------------------------------

def _hostname(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_regulator_url(url: str, regulators: Optional[Sequence[RegulatorSource]] = None) -> bool:
    """True if ``url`` is on the regulator allow-list (any approved regulator
    when ``regulators`` is None, otherwise restricted to the provided subset)."""
    pool = regulators if regulators is not None else APPROVED_REGULATORS
    return any(r.matches(url) for r in pool)


def is_consulting_url(url: str, firms: Optional[Sequence[ConsultingSource]] = None) -> bool:
    pool = firms if firms is not None else APPROVED_CONSULTING_FIRMS
    return any(f.matches(url) for f in pool)


def regulator_for_url(url: str, regulators: Optional[Sequence[RegulatorSource]] = None) -> Optional[RegulatorSource]:
    pool = regulators if regulators is not None else APPROVED_REGULATORS
    for r in pool:
        if r.matches(url):
            return r
    return None


def consulting_for_url(url: str, firms: Optional[Sequence[ConsultingSource]] = None) -> Optional[ConsultingSource]:
    pool = firms if firms is not None else APPROVED_CONSULTING_FIRMS
    for f in pool:
        if f.matches(url):
            return f
    return None


# ---------------------------------------------------------------------------
# Runtime configuration knobs (env-driven)
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def is_regulatory_search_enabled() -> bool:
    """Master switch for Stage 1 (official regulators).

    Both the new name and the legacy ``REGULATION_WEB_SEARCH`` /
    ``DORA_ENABLE_WEB_SEARCH`` flags are honoured so existing ``.env`` files
    keep working after the refactor.
    """
    return (
        _env_bool("REGULATORY_SEARCH_ENABLED", "true")
        or _env_bool("REGULATION_WEB_SEARCH", "false")
        or _env_bool("DORA_ENABLE_WEB_SEARCH", "false")
    )


def is_consulting_search_enabled() -> bool:
    """Master switch for Stage 2 (consulting guidance enrichment)."""
    return _env_bool("CONSULTING_SEARCH_ENABLED", "true")


def search_backends() -> List[str]:
    """Ordered list of DDGS backends to try for each query.

    Default is a single backend (``duckduckgo``) because the Stage 1 fetcher
    runs queries in parallel against many regulator domains and the secondary
    backends mostly trigger rate-limits + slow timeouts without adding new
    URLs. Set ``REGULATORY_SEARCH_BACKENDS=duckduckgo,brave,bing`` to opt back
    in to the multi-backend behaviour.
    """
    raw = (
        os.getenv("REGULATORY_SEARCH_BACKENDS")
        or os.getenv("REGULATION_SEARCH_BACKENDS")
        or os.getenv("DORA_SEARCH_BACKENDS")
        or "duckduckgo"
    )
    return [b.strip() for b in raw.split(",") if b.strip()]


def regulatory_max_results() -> int:
    """Hits requested from the search engine per query.

    We bump this higher than the previous default because Stage 1 now sends
    fewer, broader queries -- we rely on the URL allow-list post-filter to
    drop the ~70% of results from non-regulator domains, so we want to give
    it a larger pool to filter from.
    """
    return _env_int(
        "REGULATORY_SEARCH_MAX_RESULTS",
        os.getenv("REGULATION_SEARCH_MAX_RESULTS", os.getenv("DORA_SEARCH_MAX_RESULTS", "12")),
    )


def regulatory_exhaustive_max_results() -> int:
    """Hits requested per query when Stage 1 runs in **exhaustive** mode.

    Exhaustive mode is triggered when the user uploads a regulation
    document (BRD/FRD or a raw regulation PDF/DOCX) — at that point we
    know exactly which regulation is at hand and want to enumerate as
    many authoritative publications as the selected regulators expose.
    The default is intentionally higher than ``regulatory_max_results``
    so each per-regulator native site search returns a broad slate of
    hits in a single round-trip, without changing the tighter budget
    used for cheap live previews.
    """
    return _env_int(
        "REGULATORY_EXHAUSTIVE_MAX_RESULTS",
        os.getenv("REGULATORY_SEARCH_MAX_RESULTS_EXHAUSTIVE", "40"),
    )


def consulting_max_results() -> int:
    return _env_int("CONSULTING_SEARCH_MAX_RESULTS", "4")


def search_timeout_seconds() -> int:
    """Per-call DDGS timeout. Kept short because Stage 1 dispatches many
    queries in parallel and we want a slow backend to fail fast rather than
    block the whole pipeline.
    """
    return _env_int(
        "REGULATORY_SEARCH_TIMEOUT",
        os.getenv("REGULATION_SEARCH_TIMEOUT", os.getenv("DORA_SEARCH_TIMEOUT", "8")),
    )


__all__ = [
    # constants
    "SOURCE_TYPE_OFFICIAL_REGULATOR",
    "SOURCE_TYPE_OFFICIAL_LEGISLATION",
    "SOURCE_TYPE_CONSULTING_GUIDANCE",
    "SOURCE_PRIORITY",
    "PUBLICATION_TYPES",
    # regulator API
    "RegulatorSource",
    "APPROVED_REGULATORS",
    "regulator_codes",
    "get_regulator",
    "resolve_regulators",
    "regulator_domains",
    "is_regulator_url",
    "regulator_for_url",
    # consulting API
    "ConsultingSource",
    "APPROVED_CONSULTING_FIRMS",
    "consulting_codes",
    "get_consulting",
    "resolve_consulting_firms",
    "consulting_domains",
    "is_consulting_url",
    "consulting_for_url",
    # runtime config
    "is_regulatory_search_enabled",
    "is_consulting_search_enabled",
    "search_backends",
    "regulatory_max_results",
    "regulatory_exhaustive_max_results",
    "consulting_max_results",
    "search_timeout_seconds",
]
