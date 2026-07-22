"""Regulation detector for uploaded documents.

Purpose
-------
The Streamlit app used to hard-code ``"DORA"`` as the regulation label in
``st.session_state["regulation"]``. That default leaked all the way down
the pipeline: even when a user uploaded a BRD/FRD or a regulation PDF
for a completely different regime (GDPR, MiFID II, RBI Master
Directions, Housing Finance, etc.), the app still ran the live
regulator search, native adapters, native queries, and BRD generation
against ``"DORA"``.

This module reads the raw text of an uploaded document and returns a
best-guess regulation code + display label. The detection uses two
tiers so callers can rely on it even when the GenAI client is offline:

1. **Deterministic** keyword / synonym / reference-code match against a
   curated glossary of well-known regulations (DORA, GDPR, MiFID II,
   BASEL III, PSD2, EMIR, MAR, MiCA, NIS2, CSRD, CRR/CRD, AMLD,
   Housing Finance / RBI / NHB directions, FCA/PRA policy statements,
   etc.). Fast, offline, no LLM required.
2. **LLM fallback** — when the deterministic tier finds no strong hit
   (or only a weak one) and a live
   :class:`~services.genai_service.GenAIClient` is available, we ask
   the LLM to classify the document in one structured round-trip. The
   LLM answer is grounded in the top ~28k characters of document text
   and is required to output a short canonical label rather than free
   prose, which keeps the downstream regulator search and native
   adapters stable.

Design notes
------------
* Purely additive. Callers that still want to override the detected
  label (e.g. a user typed a specific regulation in the sidebar) can
  simply ignore the return value.
* The deterministic glossary is intentionally small and biased toward
  the regulations the app already ships native adapters or reference
  data for. Adding a new regulation should be a one-line edit to
  :data:`REGULATION_KEYWORDS`.
* When both tiers fail we return :data:`DetectedRegulation.unknown()` —
  callers should keep the user's last-known label rather than blanking
  it out.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pydantic.v1 import BaseModel, Field
except Exception:  # pragma: no cover - pydantic v2 fallback
    from pydantic import BaseModel, Field  # type: ignore

logger = logging.getLogger(__name__)


# Hard cap on document text forwarded to the LLM in the fallback path.
# The rest of the app's LLM calls hover around 30k characters of
# component-specific prompt; we mirror that ceiling here so the
# detection call fits inside the same context budget.
_MAX_DOCX_TEXT_CHARS = 28_000

# Text-scan window for the deterministic tier. Scanning the full
# document is unnecessary — the regulation is almost always named in
# the title, the first heading, or the opening paragraphs. Bounding the
# scan keeps false-positives from later chapters ("appendix references
# GDPR / DORA / MiFID as related regulations") from hijacking the
# detection.
_DETERMINISTIC_SCAN_CHARS = 12_000


# ---------------------------------------------------------------------------
# Curated regulation glossary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegulationEntry:
    """One row in the deterministic detector's glossary."""

    code: str                       # canonical short code used by the app (e.g. "DORA")
    display_name: str               # human-readable label (e.g. "Digital Operational Resilience Act")
    # Case-insensitive keywords / synonyms / reference codes. Matching
    # is substring-based against the normalised text ("  " collapsed,
    # lowercase); patterns should stay short so the false-positive rate
    # is low. Every entry needs at least one high-signal keyword.
    keywords: Sequence[str] = field(default_factory=tuple)
    # Regex patterns applied against the raw text. Use these for
    # citation-shaped strings (``Regulation (EU) 2022/2554``) where a
    # bare substring match would over-fire.
    patterns: Sequence[str] = field(default_factory=tuple)
    # Weight nudges the score for stronger unique markers. Default is
    # 1.0 per keyword hit; a canonical citation match adds ``pattern_weight``.
    pattern_weight: float = 1.5
    # Ordered list of :mod:`services.search_config` regulator codes
    # that should be searched for this regulation. Populated with the
    # regulators that publish authoritative content for the regime —
    # e.g. ``("RBI", "NHB", "MOHUA", "RERA")`` for Indian Housing
    # Finance, ``("EBA", "ESMA", "EIOPA", "EUR_LEX", "ECB")`` for
    # DORA. When empty the detector produces no regulator scope hint
    # and the caller keeps the current UI selection (usually "ALL").
    #
    # This is the mechanism that stops a Housing Finance PDF from
    # being enriched with European Banking Authority publications —
    # once the detector identifies the regulation, it also tells the
    # caller which regulator domains actually matter.
    preferred_regulator_codes: Sequence[str] = field(default_factory=tuple)


# Ordering matters: the first entry with the highest score wins on tie,
# so put the most specific regulation before overlapping catch-alls.
REGULATION_KEYWORDS: List[RegulationEntry] = [
    RegulationEntry(
        code="DORA",
        display_name="Digital Operational Resilience Act (DORA)",
        keywords=(
            "digital operational resilience act",
            "digital operational resilience",
            "dora regulation",
            " dora ",
            "regulation (eu) 2022/2554",
            "eu 2022/2554",
        ),
        patterns=(
            r"\bRegulation\s*\(EU\)\s*2022/2554\b",
            r"\b32022R2554\b",
        ),
        preferred_regulator_codes=("EBA", "ESMA", "EIOPA", "EUR_LEX", "ECB"),
    ),
    RegulationEntry(
        code="GDPR",
        display_name="General Data Protection Regulation (GDPR)",
        keywords=(
            "general data protection regulation",
            " gdpr ",
            "regulation (eu) 2016/679",
        ),
        patterns=(
            r"\bRegulation\s*\(EU\)\s*2016/679\b",
            r"\b32016R0679\b",
        ),
        preferred_regulator_codes=("EUR_LEX", "EBA", "ESMA", "EIOPA"),
    ),
    RegulationEntry(
        code="MiFID II",
        display_name="Markets in Financial Instruments Directive II (MiFID II)",
        keywords=(
            "markets in financial instruments directive",
            "markets in financial instruments",
            "mifid ii",
            "mifid2",
            " mifid ",
            "directive 2014/65/eu",
        ),
        patterns=(r"\bDirective\s*2014/65/EU\b",),
        preferred_regulator_codes=("ESMA", "EUR_LEX", "FCA"),
    ),
    RegulationEntry(
        code="MiCA",
        display_name="Markets in Crypto-Assets Regulation (MiCA)",
        keywords=(
            "markets in crypto-assets",
            "markets in cryptoassets",
            " mica ",
            "regulation (eu) 2023/1114",
        ),
        patterns=(r"\bRegulation\s*\(EU\)\s*2023/1114\b",),
        preferred_regulator_codes=("ESMA", "EUR_LEX", "EBA"),
    ),
    RegulationEntry(
        code="EMIR",
        display_name="European Market Infrastructure Regulation (EMIR)",
        keywords=(
            "european market infrastructure regulation",
            " emir ",
            "regulation (eu) no 648/2012",
        ),
        patterns=(r"\bRegulation\s*\(EU\)\s*(?:No\s*)?648/2012\b",),
        preferred_regulator_codes=("ESMA", "EUR_LEX", "EBA"),
    ),
    RegulationEntry(
        code="MAR",
        display_name="Market Abuse Regulation (MAR)",
        keywords=(
            "market abuse regulation",
            "regulation (eu) no 596/2014",
        ),
        patterns=(r"\bRegulation\s*\(EU\)\s*(?:No\s*)?596/2014\b",),
        preferred_regulator_codes=("ESMA", "EUR_LEX"),
    ),
    RegulationEntry(
        code="NIS2",
        display_name="Network and Information Security Directive 2 (NIS2)",
        keywords=(
            "network and information security directive",
            "nis 2 directive",
            " nis2 ",
            "directive (eu) 2022/2555",
        ),
        patterns=(r"\bDirective\s*\(EU\)\s*2022/2555\b",),
        preferred_regulator_codes=("EUR_LEX", "EBA"),
    ),
    RegulationEntry(
        code="CSRD",
        display_name="Corporate Sustainability Reporting Directive (CSRD)",
        keywords=(
            "corporate sustainability reporting",
            " csrd ",
            "directive (eu) 2022/2464",
        ),
        patterns=(r"\bDirective\s*\(EU\)\s*2022/2464\b",),
        preferred_regulator_codes=("EUR_LEX", "ESMA"),
    ),
    RegulationEntry(
        code="PSD2",
        display_name="Revised Payment Services Directive (PSD2)",
        keywords=(
            "revised payment services directive",
            "payment services directive 2",
            " psd2 ",
            "directive (eu) 2015/2366",
        ),
        patterns=(r"\bDirective\s*\(EU\)\s*2015/2366\b",),
        preferred_regulator_codes=("EBA", "EUR_LEX", "ECB"),
    ),
    RegulationEntry(
        code="BASEL III",
        display_name="Basel III Framework",
        keywords=(
            "basel iii framework",
            " basel iii ",
            "basel 3 ",
            "basel committee on banking supervision",
        ),
        preferred_regulator_codes=("EBA", "ECB", "BAFIN", "DNB", "PRA"),
    ),
    RegulationEntry(
        code="CRR",
        display_name="Capital Requirements Regulation (CRR)",
        keywords=(
            "capital requirements regulation",
            " crr ",
            "regulation (eu) no 575/2013",
        ),
        patterns=(r"\bRegulation\s*\(EU\)\s*(?:No\s*)?575/2013\b",),
        preferred_regulator_codes=("EBA", "EUR_LEX", "ECB"),
    ),
    RegulationEntry(
        code="CRD",
        display_name="Capital Requirements Directive (CRD)",
        keywords=(
            "capital requirements directive",
            " crd iv ",
            "directive 2013/36/eu",
        ),
        patterns=(r"\bDirective\s*2013/36/EU\b",),
        preferred_regulator_codes=("EBA", "EUR_LEX", "ECB"),
    ),
    RegulationEntry(
        code="AMLD",
        display_name="Anti-Money Laundering Directive (AMLD)",
        keywords=(
            "anti-money laundering directive",
            "anti money laundering directive",
            " amld ",
            "6th aml directive",
            "sixth aml directive",
        ),
        preferred_regulator_codes=("AMLA", "EBA", "EUR_LEX"),
    ),
    RegulationEntry(
        code="AI ACT",
        display_name="EU Artificial Intelligence Act",
        keywords=(
            "artificial intelligence act",
            "eu ai act",
            " ai act ",
            "regulation (eu) 2024/1689",
        ),
        patterns=(r"\bRegulation\s*\(EU\)\s*2024/1689\b",),
        preferred_regulator_codes=("EUR_LEX", "DG_FISMA"),
    ),
    RegulationEntry(
        code="HOUSING FINANCE",
        display_name="Housing Finance / RBI-NHB Directions (India)",
        keywords=(
            # Strong markers - specific HFC / NHB references
            "housing finance company",
            "housing finance companies",
            "national housing bank",
            "nhb directions",
            "rbi master direction - non-banking financial company",
            "rbi (housing finance",
            "hfc directions",
            # RBI Master Circular / general housing finance titles.
            # The RBI publishes a "Master Circular - Housing Finance"
            # (DBR.No.DIR.BC.13/08.12.001/...) that consolidates the
            # housing-finance rules for scheduled commercial banks.
            # Without these markers the detector missed the entire
            # circular family (the docs never mention "HFC" or the
            # NHB in their title).
            "master circular - housing finance",
            "master circular – housing finance",
            "master circular housing finance",
            " housing finance ",
            "housing loans under priority sector",
            "housing finance allocation",
            "national housing finance policy",
        ),
        patterns=(
            # RBI reference codes on the master circular / directions
            # follow the ``RBI/2015-16/46`` and ``DBR.No.DIR.BC.13/
            # 08.12.001/2015-16`` styles. Presence of *either* is a
            # very high-signal marker that the doc is an RBI-issued
            # circular; combined with a housing-finance keyword above
            # it comfortably clears the 0.6 confidence threshold.
            r"\bRBI/\d{4}-\d{2}/\d{1,4}\b",
            r"\bDBR\.[A-Z]{2,5}\.[A-Z]{2,5}\.\d{1,4}/\d{2}\.\d{2}\.\d{3}/\d{4}-\d{2}\b",
        ),
        # Indian housing-finance scope: RBI publishes the circulars,
        # NHB supervises HFCs, MOHUA owns the housing policy layer
        # (PMAY etc.), and State RERA covers the developer /
        # homebuyer side. European regulators (EBA / ESMA / etc.)
        # never publish authoritative content on Indian housing
        # finance, so we exclude them from the scope.
        preferred_regulator_codes=("RBI", "NHB", "MOHUA", "RERA", "CERSAI", "DFS"),
    ),
    RegulationEntry(
        code="RBI MASTER DIRECTION",
        display_name="RBI Master Direction / Master Circular",
        keywords=(
            "rbi master direction",
            "reserve bank of india master direction",
            "master direction - reserve bank",
            # RBI's older consolidation instrument is the Master
            # Circular (still widely in force). The circulars follow
            # the same regulatory-force convention as directions, so
            # they belong under the same code.
            "rbi master circular",
            "reserve bank of india master circular",
            "master circular - reserve bank",
            "master circular – reserve bank",
            "department of banking regulation",
            "reserve bank of india",
            "banking regulation act, 1949",
            "banking regulation act 1949",
        ),
        patterns=(
            r"\bRBI/\d{4}-\d{2}/\d{1,4}\b",
        ),
        preferred_regulator_codes=("RBI", "NHB", "SEBI", "FIU_IND", "IBBI"),
    ),
    RegulationEntry(
        code="SEBI",
        display_name="SEBI Regulations (India)",
        keywords=(
            "securities and exchange board of india",
            "sebi (listing obligations",
            "sebi (mutual funds",
            "sebi regulations",
        ),
        preferred_regulator_codes=("SEBI", "RBI", "IBBI"),
    ),
    RegulationEntry(
        code="FCA HANDBOOK",
        display_name="FCA Handbook (UK)",
        keywords=(
            "fca handbook",
            "financial conduct authority handbook",
            "sysc ",
            "cobs ",
        ),
        preferred_regulator_codes=("FCA", "PRA"),
    ),
    RegulationEntry(
        code="PRA SS",
        display_name="PRA Supervisory Statement (UK)",
        keywords=(
            "prudential regulation authority supervisory statement",
            "pra supervisory statement",
            "pra ss",
        ),
        patterns=(r"\bSS\d{1,3}/\d{2}\b",),
        preferred_regulator_codes=("PRA", "FCA"),
    ),
]


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class DetectedRegulation:
    """Outcome of a detection attempt.

    Attributes
    ----------
    code
        Canonical short code (``"DORA"``, ``"GDPR"``, ...) that callers
        should push into ``st.session_state["regulation"]``. Empty when
        no regulation could be identified.
    display_name
        Human-readable label. Falls back to ``code`` when the glossary
        does not carry a friendlier form.
    confidence
        ``0.0 - 1.0`` heuristic score. ``>=0.6`` is treated as "high
        confidence" by callers; anything below is a hint and should be
        surfaced to the user for confirmation.
    method
        ``"keyword"`` when the deterministic tier settled it,
        ``"llm"`` when the LLM fallback made the call, ``"none"``
        when both tiers failed.
    matched_terms
        Human-readable list of the keywords / patterns that fired.
        Used by the UI to explain the detection to the user.
    preferred_regulator_codes
        Ordered list of :mod:`services.search_config` regulator codes
        the caller should search for this regulation. Populated from
        the matching glossary entry (see
        :class:`RegulationEntry.preferred_regulator_codes`) so an
        Indian Housing Finance detection never triggers a European
        Banking Authority search and vice versa. Empty when the LLM
        tier produced the detection — the LLM cannot infer a
        regulator scope on its own, so the caller keeps whatever
        selection was already active.
    """

    code: str = ""
    display_name: str = ""
    confidence: float = 0.0
    method: str = "none"
    matched_terms: List[str] = field(default_factory=list)
    preferred_regulator_codes: List[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return bool(self.code) and self.confidence >= 0.6

    @property
    def is_known(self) -> bool:
        return bool(self.code)

    @classmethod
    def unknown(cls) -> "DetectedRegulation":
        return cls()


# ---------------------------------------------------------------------------
# Deterministic tier
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace + lowercase for stable substring matching.

    We surround the text with spaces so ``" dora "`` (with word
    boundaries) still matches when the acronym opens or closes the
    document.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text.lower())
    return f" {collapsed} "


def _score_entry(entry: RegulationEntry, normalised: str, raw: str) -> Tuple[float, List[str]]:
    """Score one glossary entry against the document text.

    Returns ``(score, matched_terms)``. Score is ``sum(keyword_hits) +
    pattern_weight * pattern_hits``. A missing entry returns
    ``(0.0, [])``.
    """
    matches: List[str] = []
    score = 0.0

    for kw in entry.keywords:
        key = kw.lower()
        if key.strip() and key in normalised:
            occurrences = normalised.count(key)
            # First occurrence counts full, subsequent occurrences add a
            # smaller amount so a document that mentions the regulation
            # many times still ranks above one with a single mention -
            # but not so much that a spammy footer citation dominates.
            score += 1.0 + 0.2 * (occurrences - 1)
            matches.append(kw.strip())

    for pat in entry.patterns:
        try:
            hits = re.findall(pat, raw, flags=re.IGNORECASE)
        except re.error:
            hits = []
        if hits:
            score += entry.pattern_weight * len(hits)
            matches.append(hits[0] if isinstance(hits[0], str) else str(hits[0]))

    return score, matches


def _deterministic_detect(text: str) -> DetectedRegulation:
    """Score every glossary entry and return the best-scoring hit.

    Confidence mapping (empirical, tuned on the app's sample BRD):

    * score >= 5    -> 0.95 (very strong: multiple keyword hits and/or
      a canonical citation reference)
    * score >= 3    -> 0.85
    * score >= 2    -> 0.75
    * score >= 1    -> 0.60 (borderline; still returned but callers may
      want to confirm)
    * score < 1     -> unknown
    """
    scan = (text or "")[:_DETERMINISTIC_SCAN_CHARS]
    if not scan.strip():
        return DetectedRegulation.unknown()

    normalised = _normalise(scan)

    best: Optional[Tuple[RegulationEntry, float, List[str]]] = None
    for entry in REGULATION_KEYWORDS:
        score, matches = _score_entry(entry, normalised, scan)
        if score <= 0:
            continue
        if best is None or score > best[1]:
            best = (entry, score, matches)

    if best is None:
        return DetectedRegulation.unknown()

    entry, score, matches = best
    if score >= 5:
        confidence = 0.95
    elif score >= 3:
        confidence = 0.85
    elif score >= 2:
        confidence = 0.75
    else:
        confidence = 0.60

    logger.info(
        "Regulation detector (keyword tier): code=%s score=%.2f confidence=%.2f "
        "matched_terms=%s",
        entry.code, score, confidence, matches[:5],
    )

    return DetectedRegulation(
        code=entry.code,
        display_name=entry.display_name,
        confidence=confidence,
        method="keyword",
        matched_terms=matches,
        preferred_regulator_codes=list(entry.preferred_regulator_codes),
    )


# ---------------------------------------------------------------------------
# LLM fallback tier
# ---------------------------------------------------------------------------

class _LLMRegulationHint(BaseModel):
    """Structured response the LLM returns for the fallback tier."""

    regulation_code: str = Field(
        default="",
        description=(
            "Short canonical acronym or code for the regulation the "
            "document is about (e.g. 'DORA', 'GDPR', 'MiFID II', "
            "'Basel III', 'RBI Master Direction'). Empty when the "
            "document does not focus on a single identifiable "
            "regulation."
        ),
    )
    regulation_full_name: str = Field(
        default="",
        description=(
            "Full name of the regulation ('Digital Operational "
            "Resilience Act', 'General Data Protection Regulation', "
            "'Markets in Financial Instruments Directive II'). Empty "
            "when the code alone is enough."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description=(
            "0.0-1.0 confidence that regulation_code is correct. Use "
            "0.9+ only when the document explicitly names the "
            "regulation multiple times; use 0.3-0.5 when you are "
            "guessing from indirect signals."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "One-sentence justification citing the strongest phrase "
            "you saw in the document text."
        ),
    )


_LLM_SYSTEM_INSTRUCTION = (
    "You are a regulatory analyst. Given the raw text of a document "
    "(a business/functional requirements document, a regulation "
    "excerpt, a policy statement, or similar), identify the SINGLE "
    "regulation the document is primarily about. Return a short "
    "canonical acronym / code (DORA, GDPR, MiFID II, EMIR, NIS2, "
    "Basel III, CRR, CRD, PSD2, MiCA, CSRD, AMLD, AI Act, RBI Master "
    "Direction, SEBI, FCA Handbook, PRA Supervisory Statement, etc.). "
    "Do NOT invent a regulation that is not clearly evidenced in the "
    "text. When the document is generic or covers no specific "
    "regulation, return an empty regulation_code and a low "
    "confidence."
)


def _llm_detect(text: str, client: Any) -> Optional[DetectedRegulation]:
    """LLM fallback tier. Returns ``None`` when unavailable / on failure."""
    if client is None or not text or not text.strip():
        return None

    excerpt = text.strip()
    if len(excerpt) > _MAX_DOCX_TEXT_CHARS:
        excerpt = excerpt[:_MAX_DOCX_TEXT_CHARS]
        logger.info(
            "Regulation detector (LLM tier): text truncated for prompt. "
            "kept_chars=%d", len(excerpt),
        )

    prompt = (
        "Identify the single regulation this document is primarily "
        "about. Populate the _LLMRegulationHint schema.\n\n"
        "------- BEGIN DOCUMENT TEXT -------\n"
        f"{excerpt}\n"
        "------- END DOCUMENT TEXT -------"
    )

    try:
        hint: _LLMRegulationHint = client.generate(
            schema_model=_LLMRegulationHint,
            component_name="Regulation detection from uploaded document",
            component_instruction=prompt,
            context="",
            system_instruction=_LLM_SYSTEM_INSTRUCTION,
        )
    except Exception:
        logger.exception("Regulation detector LLM call FAILED.")
        return None

    if hint is None:
        return None

    code = (hint.regulation_code or "").strip()
    if not code:
        return None

    display = (hint.regulation_full_name or "").strip() or code
    confidence = float(hint.confidence or 0.0)
    if confidence <= 0.0:
        confidence = 0.6
    if confidence > 1.0:
        confidence = 1.0

    # If the LLM's code lines up with a known glossary entry (case-
    # insensitive), inherit the regulator-scope hint from the entry so
    # the caller can still pin the search to the right jurisdiction.
    # This is essential for the LLM path — the LLM knows the
    # regulation is (say) DORA but has no way to name the regulator
    # scope codes the app understands.
    scoped_regulators: List[str] = []
    for entry in REGULATION_KEYWORDS:
        if entry.code.lower() == code.lower():
            scoped_regulators = list(entry.preferred_regulator_codes)
            break

    logger.info(
        "Regulation detector (LLM tier): code=%s confidence=%.2f "
        "regulator_scope=%s reasoning=%s",
        code, confidence, scoped_regulators, (hint.reasoning or "")[:120],
    )

    return DetectedRegulation(
        code=code,
        display_name=display,
        confidence=confidence,
        method="llm",
        matched_terms=[hint.reasoning] if hint.reasoning else [],
        preferred_regulator_codes=scoped_regulators,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def detect_regulation_from_text(
    text: str,
    *,
    client: Any = None,
    prefer_llm_below_confidence: float = 0.6,
) -> DetectedRegulation:
    """Detect the regulation named in ``text``.

    Parameters
    ----------
    text
        Raw document text. Only the first
        :data:`_DETERMINISTIC_SCAN_CHARS` characters are inspected by
        the keyword tier; the LLM tier consumes up to
        :data:`_MAX_DOCX_TEXT_CHARS`.
    client
        Optional live :class:`~services.genai_service.GenAIClient`. When
        supplied and the keyword tier is uncertain
        (``< prefer_llm_below_confidence``), we escalate to the LLM.
    prefer_llm_below_confidence
        Threshold used to decide whether the keyword-tier hit is trusted
        as-is or the LLM is consulted for a second opinion. Setting this
        to ``0.0`` disables the LLM escalation entirely.
    """
    deterministic = _deterministic_detect(text)

    if deterministic.confidence >= prefer_llm_below_confidence:
        return deterministic

    llm_result = _llm_detect(text, client)
    if llm_result is None:
        return deterministic

    if not deterministic.is_known:
        return llm_result

    # Both tiers produced a hint. Prefer the one with the higher
    # confidence; on ties, prefer the deterministic hit because it is
    # anchored on an exact keyword match rather than model judgment.
    if llm_result.confidence > deterministic.confidence + 0.1:
        return llm_result
    return deterministic


def detect_regulation_from_docx(
    path: str | Path,
    *,
    client: Any = None,
) -> DetectedRegulation:
    """Convenience wrapper around :func:`detect_regulation_from_text` for a DOCX path.

    Uses :func:`utils.docx_parser.extract_full_text` so paragraphs and
    flattened table cells (Requirement / Detailed Requirement / etc.)
    are all considered. Falls back to an empty result when the file
    cannot be read.
    """
    from utils.docx_parser import extract_full_text  # local import to keep import graph flat

    try:
        text = extract_full_text(str(path), include_tables=True)
    except Exception:
        logger.exception("Regulation detector: DOCX read failed. path=%s", path)
        return DetectedRegulation.unknown()

    return detect_regulation_from_text(text, client=client)


def detect_regulation_from_pdf(
    path: str | Path,
    *,
    client: Any = None,
) -> DetectedRegulation:
    """Convenience wrapper for a PDF path.

    Uses the project's existing PyMuPDF-backed helper
    :func:`utils.pdf_parser.extract_pdf_text` — the same backend that
    the BRD generator uses to consume uploaded regulation PDFs. This
    keeps a single PDF dependency (PyMuPDF / ``fitz``) across the
    codebase and avoids introducing a second one. Returns an empty
    result when the file cannot be opened, is encrypted / scanned, or
    yields no extractable text.
    """
    p = Path(path)
    if not p.exists():
        return DetectedRegulation.unknown()

    try:
        from utils.pdf_parser import extract_pdf_text  # local import to keep the import graph flat
    except Exception:
        logger.exception(
            "Regulation detector: utils.pdf_parser unavailable. "
            "Cannot extract text from %s.", p,
        )
        return DetectedRegulation.unknown()

    try:
        # ``_MAX_DOCX_TEXT_CHARS`` matches the LLM tier's budget; the
        # deterministic tier only scans ``_DETERMINISTIC_SCAN_CHARS``
        # anyway, so capping here saves memory on 500-page consultation
        # papers without affecting detection accuracy.
        result = extract_pdf_text(p, max_chars=_MAX_DOCX_TEXT_CHARS)
    except Exception:
        logger.exception(
            "Regulation detector: PDF text extraction failed. path=%s", p,
        )
        return DetectedRegulation.unknown()

    if result.warning_message:
        logger.info(
            "Regulation detector: PDF extraction warning for %s: %s",
            p.name, result.warning_message,
        )
    if result.is_empty:
        logger.info(
            "Regulation detector: no extractable text in PDF (pages=%d "
            "encrypted=%s). Path=%s",
            result.page_count, result.is_encrypted, p,
        )
        return DetectedRegulation.unknown()

    return detect_regulation_from_text(result.text, client=client)


def detect_regulation_from_upload(
    path: str | Path,
    *,
    client: Any = None,
) -> DetectedRegulation:
    """Auto-dispatch to the right reader based on the file extension.

    Supported: ``.docx``, ``.pdf``. Unknown extensions return
    :meth:`DetectedRegulation.unknown`.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        return detect_regulation_from_docx(p, client=client)
    if ext == ".pdf":
        return detect_regulation_from_pdf(p, client=client)
    return DetectedRegulation.unknown()


__all__ = [
    "DetectedRegulation",
    "REGULATION_KEYWORDS",
    "RegulationEntry",
    "detect_regulation_from_docx",
    "detect_regulation_from_pdf",
    "detect_regulation_from_text",
    "detect_regulation_from_upload",
]
