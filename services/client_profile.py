"""Client Profile — keyword catalogs and helpers.

The Client Role-Aware pipeline is anchored by :mod:`services.client_roles`
(the *what kind of institution* dimension). Real-world regulatory
interpretation, however, also depends on **who the client is** in a much
richer sense:

* Organization Profile         — size, footprint, ownership model.
* Business Lines               — the businesses actually run by the client.
* Products in Scope            — the specific products the regulation should
                                 be applied to.
* Countries of Operation       — the jurisdictions the client operates in.
* Legal Entities               — the corporate / legal-entity shape.
* Vendor & Third Parties       — the third-party ecosystem the client
                                 depends on.

The Page 1 UI exposes each dimension as a *keyword multi-select* — the same
UX pattern used to tag CVs. Each field ships with a curated seed catalog so
users can pick from a reasonable default set; the widget also accepts
free-form keywords so users can type in any additional term they care
about.

This module owns:

* :data:`CLIENT_PROFILE_FIELDS` — ordered spec (label, key, catalog, help
  text) used by ``app.py`` to render every widget consistently.
* :func:`normalize_client_profile` — validation + de-duplication helper.
* :func:`client_profile_prompt_block` — text block prepended to the BRD /
  interpretation prompt so downstream agents know the profile.
* :func:`client_profile_context_text` — flat string used by the deterministic
  role-aware interpretation engine as extra keyword signal.

The catalogs are intentionally broad. The engine never treats an
unrecognised keyword as an error — free-form entries are always accepted and
carried through to the prompt / metadata verbatim, so users can inject
domain-specific vocabulary (e.g. an internal product code) without touching
the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Seed catalogs
# ---------------------------------------------------------------------------

ORGANIZATION_PROFILE_OPTIONS: List[str] = [
    "Group Entity",
    "Publicly Listed",
    "Privately Held",
    "Standalone",
    "Subsidiary",
    "Holding Company",
]


BUSINESS_LINES_OPTIONS: List[str] = [
    "Retail Banking",
    "Corporate Banking",
    "Investment Banking",
    "Wealth Management",
    "Private Banking",
    "Asset Management",
    "Custody Services",
    "Prime Brokerage",
    "Securities Services",
    "Payments & Cash Management",
    "Trade Finance",
    "Treasury / ALM",
    "Global Markets",
    "Equities Trading",
    "Fixed Income Trading",
    "FX Trading",
    "Rates Trading",
    "Commodities Trading",
    "Derivatives Trading",
    "Market Making",
    "Underwriting",
    "M&A Advisory",
    "Equity Capital Markets",
    "Debt Capital Markets",
    "Research",
    "Clearing & Settlement",
    "Fund Administration",
    "Fund Distribution",
    "Transfer Agency",
    "Insurance Underwriting",
    "Reinsurance",
    "Bancassurance",
    "Consumer Credit",
    "Mortgage Lending",
    "Corporate Lending",
    "Syndicated Lending",
    "Leasing",
    "Factoring",
    "Asset Finance",
    "Structured Finance",
    "Project Finance",
    "Real Estate Finance",
    "Private Equity",
    "Venture Capital",
    "Alternative Investments",
    "Digital Assets / Crypto",
    "Robo-advisory",
    "Payment Processing",
    "Merchant Acquiring",
    "E-money Issuance",
    "Remittance",
    "FX Retail Services",
    "Custody & Depositary",
    "Securities Lending",
    "Repo",
    "Correspondent Banking",
    "Card Issuance",
    "Card Acquiring",
]


PRODUCTS_IN_SCOPE_OPTIONS: List[str] = [
    "Current Accounts",
    "Savings Accounts",
    "Term Deposits",
    "Certificates of Deposit",
    "Personal Loans",
    "Auto Loans",
    "Mortgages / Home Loans",
    "Credit Cards",
    "Debit Cards",
    "Prepaid Cards",
    "Overdrafts",
    "Revolving Credit",
    "Lines of Credit",
    "Corporate Loans",
    "Syndicated Loans",
    "Trade Finance (Letters of Credit)",
    "Trade Finance (Guarantees)",
    "Documentary Collections",
    "Supply Chain Finance",
    "Cash Management",
    "Liquidity Management",
    "FX Spot",
    "FX Forwards",
    "FX Swaps",
    "FX Options",
    "Money Market Instruments",
    "Bonds",
    "Government Bonds",
    "Corporate Bonds",
    "Equities",
    "ETFs",
    "Mutual Funds",
    "UCITS",
    "AIFs",
    "Hedge Fund Products",
    "Structured Products",
    "Structured Notes",
    "Interest Rate Swaps",
    "Credit Default Swaps",
    "Equity Derivatives",
    "Commodity Derivatives",
    "OTC Derivatives",
    "Listed Derivatives",
    "Securitisations",
    "Covered Bonds",
    "Repos",
    "Reverse Repos",
    "Securities Lending Products",
    "Custody Services",
    "Depositary Services",
    "Fund Administration Services",
    "Broking Services",
    "Advisory Services",
    "Discretionary Portfolio Management",
    "Advisory Portfolio Management",
    "Prime Brokerage Services",
    "Clearing Services",
    "Settlement Services",
    "SEPA Payments",
    "SWIFT Payments",
    "Domestic Payments",
    "Cross-border Payments",
    "Instant Payments",
    "E-money",
    "Digital Wallets",
    "Buy Now Pay Later",
    "Life Insurance",
    "Non-life Insurance",
    "Health Insurance",
    "Annuities",
    "Pension Products",
    "Crypto Products",
    "Tokenised Securities",
    "Stablecoins",
]


COUNTRIES_OF_OPERATION_OPTIONS: List[str] = [
    # Regions & aggregates
    "Global",
    "European Union (EU)",
    "European Economic Area (EEA)",
    "EFTA",
    "Eurozone",
    "Nordics",
    "Baltics",
    "DACH",
    "Iberia",
    "Benelux",
    "Central & Eastern Europe (CEE)",
    "South-Eastern Europe (SEE)",
    "APAC",
    "South-East Asia (SEA)",
    "GCC",
    "MENA",
    "LATAM",
    "EMEA",
    "Africa",
    "North America",
    # Individual countries (major FS jurisdictions)
    "United States",
    "United Kingdom",
    "Ireland",
    "France",
    "Germany",
    "Netherlands",
    "Luxembourg",
    "Spain",
    "Italy",
    "Portugal",
    "Belgium",
    "Austria",
    "Sweden",
    "Denmark",
    "Norway",
    "Finland",
    "Iceland",
    "Liechtenstein",
    "Switzerland",
    "Poland",
    "Czech Republic",
    "Hungary",
    "Romania",
    "Bulgaria",
    "Greece",
    "Cyprus",
    "Malta",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Slovakia",
    "Slovenia",
    "Croatia",
    "Canada",
    "Mexico",
    "Brazil",
    "Argentina",
    "Chile",
    "Colombia",
    "Peru",
    "Australia",
    "New Zealand",
    "Japan",
    "Singapore",
    "Hong Kong SAR",
    "China (Mainland)",
    "India",
    "South Korea",
    "Taiwan",
    "Malaysia",
    "Indonesia",
    "Thailand",
    "Vietnam",
    "Philippines",
    "United Arab Emirates",
    "Saudi Arabia",
    "Qatar",
    "Bahrain",
    "Kuwait",
    "Oman",
    "Israel",
    "Turkey",
    "South Africa",
    "Nigeria",
    "Kenya",
    "Egypt",
    "Mauritius",
    "Cayman Islands",
    "Bermuda",
    "British Virgin Islands",
    "Jersey",
    "Guernsey",
    "Isle of Man",
]


LEGAL_ENTITIES_OPTIONS: List[str] = [
    "Public Limited Company",
    "Private Limited Company",
    "Limited Liability Partnership (LLP)",
    "General Partnership",
    "Sole Proprietorship",
    "Branch Office",
    "Representative Office",
    "Subsidiary",
    "Joint Venture",
    "Holding Company",
    "Financial Holding Company",
    "Special Purpose Vehicle (SPV)",
    "Special Purpose Entity (SPE)",
    "Trust",
    "Foundation",
    "Cooperative Society",
    "Mutual",
    "Investment Trust",
    "REIT",
    "InvIT",
    "SICAV",
    "SICAF",
    "FCP (Fonds Commun de Placement)",
    "UCITS",
    "AIF",
    "Umbrella Fund",
    "Sub-fund",
    "Feeder Fund",
    "Master Fund",
    "Master-Feeder Structure",
    "Ltd (UK)",
    "PLC (UK)",
    "LLP (UK)",
    "Inc. (US)",
    "LLC (US)",
    "GmbH (DE)",
    "AG (DE)",
    "KGaA (DE)",
    "S.A. (FR / LU / ES)",
    "S.à r.l. (LU / FR)",
    "S.p.A. (IT)",
    "S.r.l. (IT)",
    "Societas Europaea (SE)",
    "Free Zone Entity",
    "Offshore Entity",
    "Onshore Entity",
    "Regulated Parent Entity",
    "Non-regulated Group Company",
    "Public Sector Undertaking",
]


VENDOR_THIRD_PARTIES_OPTIONS: List[str] = [
    "Cloud Service Provider (CSP)",
    "IaaS Provider",
    "PaaS Provider",
    "SaaS Provider",
    "Data Center Provider",
    "Managed Service Provider (MSP)",
    "IT Outsourcing Vendor",
    "Application Software Vendor",
    "Core Banking System Vendor",
    "Trading Platform Vendor",
    "Order Management System Vendor",
    "RegTech Vendor",
    "KYC / AML Screening Provider",
    "Sanctions Screening Provider",
    "Transaction Monitoring Vendor",
    "Fraud Detection Vendor",
    "Regulatory Reporting Vendor",
    "Consulting Firm",
    "Legal Advisor",
    "Audit Firm",
    "Tax Advisor",
    "Payment Processor",
    "Card Network",
    "SWIFT",
    "Correspondent Bank",
    "Custodian",
    "Sub-custodian",
    "Fund Administrator (Third Party)",
    "Transfer Agent",
    "Prime Broker",
    "Executing Broker",
    "Clearing Broker",
    "Market Data Vendor",
    "Analytics Vendor",
    "Ratings Agency",
    "Benchmark Administrator",
    "Index Provider",
    "Cybersecurity Vendor",
    "Managed Detection & Response (MDR)",
    "Backup & DR Provider",
    "Business Continuity Provider",
    "Print & Mailing Vendor",
    "Marketing Vendor",
    "Data Broker",
    "Alternative Data Provider",
    "ETL / Data Integration Vendor",
    "HR / Payroll Vendor",
    "Facilities Vendor",
    "Physical Security Vendor",
    "Contract Recruiters",
    "Offshoring Partner",
    "Nearshoring Partner",
    "Onshore Vendor",
    "Intragroup Service Provider (IGSP)",
    "Fourth-Party Provider",
    "Critical Third Party (CTP)",
    "Non-critical Third Party",
    "Concentration Risk Vendor",
]


# ---------------------------------------------------------------------------
# Field spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientProfileField:
    """One dimension in the Client Profile keyword picker.

    ``key`` is the session-state / persistence key. ``label`` and ``help``
    are user-facing. ``options`` is the curated seed catalog. By default
    the widget also accepts free-form keywords so users can type in
    anything not in the seed list; set ``allow_freeform=False`` on a
    field where the catalog is intentionally exhaustive (e.g.
    Organization Profile) so stale / invented values are quietly dropped
    and only the curated entries ever appear in the dropdown.
    """

    key: str
    label: str
    options: List[str]
    help: str
    placeholder: str
    icon: str = ""
    allow_freeform: bool = True


CLIENT_PROFILE_FIELDS: List[ClientProfileField] = [
    ClientProfileField(
        key="organization_profile",
        label="Organization Profile",
        options=list(ORGANIZATION_PROFILE_OPTIONS),
        placeholder="e.g. Group Entity, Publicly Listed, Subsidiary…",
        icon="🏢",
        allow_freeform=False,
        help=(
            "Pick one or more keywords that describe the organisation's "
            "ownership and group structure. The catalog is intentionally "
            "fixed to these six options."
        ),
    ),
    ClientProfileField(
        key="business_lines",
        label="Business Lines",
        options=list(BUSINESS_LINES_OPTIONS),
        placeholder="e.g. Retail Banking, Prime Brokerage, Fund Administration…",
        icon="🧭",
        help=(
            "Business lines the client actually operates. The regulatory "
            "interpretation, BRD, questionnaire and recommendations are "
            "scoped to these lines. Add custom keywords for niche desks."
        ),
    ),
    ClientProfileField(
        key="products_in_scope",
        label="Products in Scope",
        options=list(PRODUCTS_IN_SCOPE_OPTIONS),
        placeholder="e.g. Mortgages, OTC Derivatives, Cross-border Payments…",
        icon="📦",
        help=(
            "Specific products the regulation should be applied to. Drives "
            "obligation applicability at the product level so questions "
            "and requirements only surface for products actually offered."
        ),
    ),
    ClientProfileField(
        key="countries_of_operation",
        label="Countries of Operation",
        options=list(COUNTRIES_OF_OPERATION_OPTIONS),
        placeholder="e.g. European Union (EU), United Kingdom, Singapore…",
        icon="🌍",
        help=(
            "Jurisdictions the client operates in. Used to weight regulator "
            "context (EBA/ESMA/EIOPA/FCA/BaFin/…) and to flag cross-border "
            "considerations in the interpretation."
        ),
    ),
    ClientProfileField(
        key="legal_entities",
        label="Legal Entities",
        options=list(LEGAL_ENTITIES_OPTIONS),
        placeholder="e.g. PLC, UCITS Umbrella Fund, Branch Office…",
        icon="⚖️",
        help=(
            "The corporate / legal-entity shapes in scope for this "
            "assessment (fund vehicles, SPVs, branches, subsidiaries, …). "
            "Influences entity-level regulatory obligations."
        ),
    ),
    ClientProfileField(
        key="vendor_third_parties",
        label="Vendor & Third Parties",
        options=list(VENDOR_THIRD_PARTIES_OPTIONS),
        placeholder="e.g. Cloud Service Provider, SWIFT, Prime Broker…",
        icon="🤝",
        help=(
            "Material vendors / third parties in the client's operating "
            "chain. Used to enrich third-party-risk obligations and the "
            "critical-third-party analysis in the interpretation."
        ),
    ),
]


CLIENT_PROFILE_KEYS: List[str] = [f.key for f in CLIENT_PROFILE_FIELDS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def empty_client_profile() -> Dict[str, List[str]]:
    """Return a freshly-empty ``{field_key: []}`` map."""
    return {field.key: [] for field in CLIENT_PROFILE_FIELDS}


# Curated "reasonable starting point" for a Commercial Bank / DORA-focused
# engagement. Every value below is an entry in the corresponding
# ``*_OPTIONS`` catalog above, so the widgets render without any
# free-form fallback. Kept small (2-6 keywords per dimension) so the
# defaults act as a nudge, not a decision — reviewers can trim or extend
# in one click on Page 1.
_DEFAULT_PROFILE_SEED: Dict[str, List[str]] = {
    "organization_profile": ["Group Entity"],
    "business_lines": ["Retail Banking"],
    "products_in_scope": ["Current Accounts"],
    "countries_of_operation": ["European Union (EU)"],
    "legal_entities": ["Public Limited Company"],
    "vendor_third_parties": ["Cloud Service Provider (CSP)"],
}


def default_client_profile() -> Dict[str, List[str]]:
    """Return a pre-populated profile for a Commercial Bank / DORA engagement.

    Chosen to match ``_DEFAULT_STATE["client_roles"] == ["Commercial Bank"]``
    and ``_DEFAULT_STATE["regulation"] == "DORA"`` in ``app.py``. Every
    keyword is drawn from the curated catalog for its dimension, so no
    free-form fallback is triggered. Downstream stages treat these as
    genuine tags — reviewers should clear anything that does not fit
    their client before running Agent 1.
    """
    return normalize_client_profile(_DEFAULT_PROFILE_SEED)


def normalize_client_profile(
    profile: Optional[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Return a canonicalised ``{field_key: [keyword, …]}`` map.

    * Unknown fields are dropped (so a stale session state does not smuggle
      random keys into the pipeline).
    * Free-form keywords are preserved verbatim (whitespace-trimmed,
      duplicates removed while keeping first-seen order).
    * ``None`` or an empty mapping yields the empty profile.
    """
    out = empty_client_profile()
    if not profile:
        return out
    for field in CLIENT_PROFILE_FIELDS:
        raw = profile.get(field.key) if isinstance(profile, Mapping) else None
        if not raw:
            continue
        if isinstance(raw, str):
            raw = [raw]
        seen: List[str] = []
        seen_lower: set = set()
        for value in raw:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            lower = text.lower()
            if lower in seen_lower:
                continue
            seen.append(text)
            seen_lower.add(lower)
        out[field.key] = seen
    return out


def is_client_profile_populated(profile: Optional[Mapping[str, Any]]) -> bool:
    """True when at least one field on the profile has a value."""
    normalized = normalize_client_profile(profile)
    return any(bool(v) for v in normalized.values())


def client_profile_keyword_bag(
    profile: Optional[Mapping[str, Any]],
) -> List[str]:
    """Return the flat list of all keywords across every dimension.

    Used by the deterministic role-aware interpretation engine as extra
    signal when scoring applicability.
    """
    normalized = normalize_client_profile(profile)
    bag: List[str] = []
    for values in normalized.values():
        for value in values:
            if value and value not in bag:
                bag.append(value)
    return bag


def client_profile_context_text(
    profile: Optional[Mapping[str, Any]],
) -> str:
    """Return a compact human-readable summary of the profile.

    Suitable for use as an *extra corpus* by the deterministic role-aware
    interpretation engine (:mod:`services.client_roles`) so the profile
    keywords count as regulatory context signal.
    """
    normalized = normalize_client_profile(profile)
    parts: List[str] = []
    for field in CLIENT_PROFILE_FIELDS:
        values = normalized.get(field.key, [])
        if values:
            parts.append(f"{field.label}: {', '.join(values)}")
    return " | ".join(parts)


def client_profile_prompt_block(
    profile: Optional[Mapping[str, Any]],
    *,
    client_roles: Optional[Sequence[str]] = None,
) -> str:
    """Return the LLM prompt block used to condition BRD / interpretation.

    Emits an empty string when the profile has no populated fields *and*
    no client roles are selected, so callers can safely concatenate the
    return value without conditional logic.
    """
    normalized = normalize_client_profile(profile)
    roles = [r for r in (client_roles or []) if r]
    if not any(normalized.values()) and not roles:
        return ""

    lines: List[str] = [
        "",
        "--- CLIENT PROFILE DIRECTIVE ---",
        (
            "The regulatory analysis, BRD/FRD, RTM, questionnaire and "
            "recommendations MUST be interpreted through the following "
            "client profile. Use these keywords to (a) tune obligation "
            "applicability, (b) scope requirement wording to the actual "
            "businesses / products / jurisdictions in play, and (c) flag "
            "cross-border, entity-level and third-party considerations "
            "wherever they surface in the regulation. Preserve keywords "
            "verbatim in your output where relevant so reviewers can trace "
            "the interpretation."
        ),
    ]
    if roles:
        lines.append(f"• Institution Type(s): {', '.join(roles)}")
    for field in CLIENT_PROFILE_FIELDS:
        values = normalized.get(field.key, [])
        if values:
            lines.append(f"• {field.label}: {', '.join(values)}")
    lines.append(
        "Do NOT fabricate applicability for keywords not present in the "
        "regulation text. When a keyword cannot be traced to the "
        "regulation, state that explicitly rather than inventing a mapping."
    )
    lines.append("--- END CLIENT PROFILE DIRECTIVE ---")
    return "\n".join(lines)


__all__ = [
    "BUSINESS_LINES_OPTIONS",
    "CLIENT_PROFILE_FIELDS",
    "CLIENT_PROFILE_KEYS",
    "COUNTRIES_OF_OPERATION_OPTIONS",
    "ClientProfileField",
    "LEGAL_ENTITIES_OPTIONS",
    "ORGANIZATION_PROFILE_OPTIONS",
    "PRODUCTS_IN_SCOPE_OPTIONS",
    "VENDOR_THIRD_PARTIES_OPTIONS",
    "client_profile_context_text",
    "client_profile_keyword_bag",
    "client_profile_prompt_block",
    "default_client_profile",
    "empty_client_profile",
    "is_client_profile_populated",
    "normalize_client_profile",
]
