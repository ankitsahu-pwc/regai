"""Canonical severity ladder for the app.

Historical duplication
----------------------
The four-band severity ladder was redefined in at least six places:

* ``services/scoring_engine.cxo_status``     — score -> (label, action)
* ``services/questionnaire_enhancer._severity_rank``       — label -> rank
* ``services/questionnaire_enhancer._weight_from_severity`` — label -> weight
* ``app._severity_class``           — readiness score -> CSS class
* ``app._impact_class``             — impact score  -> CSS class
* ``app._severity_label_from_status`` — label -> CSS class

They agreed most of the time but not always (labels drifted between
"At risk" / "High" / "at risk" / etc.). This module is now the single
source of truth. All existing helpers become thin wrappers over it, so
behaviour is preserved bit-for-bit and future changes only need one edit.

Two label vocabularies
----------------------
The app uses two label sets that describe the **same** four bands:

* Readiness ladder: ``Ready > Watch > At risk > Critical``
* Impact ladder:    ``Low   > Medium > High   > Critical``

Both are supported by :func:`from_label`.

Bands
-----
``CRITICAL`` (worst) - ``AT_RISK`` - ``WATCH`` - ``READY`` (best).

Thresholds
----------
Readiness / compliance scores use the ``>=75 / >=50 / >=25 / <25`` bands
(higher = better). Impact scores use the mirror ladder (higher = worse):
``>=75 impact -> Critical``, ``>=50 -> At risk``, ``>=25 -> Watch``,
``<25 -> Ready``. A readiness of 71.5% (impact 28.5%) therefore reads
as ``Watch`` on both axes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


class SeverityBand(Enum):
    """One of the four canonical severity bands.

    Ordered from worst (``CRITICAL``) to best (``READY``). The enum value
    also serves as the sort ordinal (0 = worst, 3 = best) so callers can
    sort with ``key=lambda x: band.value`` without an extra lookup.
    """

    CRITICAL = 0
    AT_RISK = 1
    WATCH = 2
    READY = 3


# ---------------------------------------------------------------------------
# Score-band boundaries
# ---------------------------------------------------------------------------

#: Readiness / compliance thresholds (higher = better).
READINESS_THRESHOLDS: Tuple[float, float, float] = (75.0, 50.0, 25.0)


def readiness_band(score: Optional[float]) -> Optional[SeverityBand]:
    """Return the severity band for a **readiness / compliance** score.

    Returns ``None`` when the score is missing or non-numeric so callers
    can distinguish "no data" from "worst band".
    """
    if score is None:
        return None
    try:
        val = float(score)
    except (TypeError, ValueError):
        return None
    hi, mid, lo = READINESS_THRESHOLDS
    if val >= hi:
        return SeverityBand.READY
    if val >= mid:
        return SeverityBand.WATCH
    if val >= lo:
        return SeverityBand.AT_RISK
    return SeverityBand.CRITICAL


def impact_band(impact: Optional[float]) -> Optional[SeverityBand]:
    """Return the severity band for an **impact %** score (higher = worse)."""
    if impact is None:
        return None
    try:
        val = float(impact)
    except (TypeError, ValueError):
        return None
    hi, mid, lo = READINESS_THRESHOLDS
    if val >= hi:
        return SeverityBand.CRITICAL
    if val >= mid:
        return SeverityBand.AT_RISK
    if val >= lo:
        return SeverityBand.WATCH
    return SeverityBand.READY


# ---------------------------------------------------------------------------
# Label / band conversion
# ---------------------------------------------------------------------------

#: Canonical readiness-ladder labels (worst -> best).
READINESS_LABELS: Tuple[str, str, str, str] = ("Critical", "At risk", "Watch", "Ready")

#: Canonical impact-ladder labels (worst -> best).
IMPACT_LABELS: Tuple[str, str, str, str] = ("Critical", "High", "Medium", "Low")


# All accepted spellings mapped to a canonical band. Kept case-insensitive
# so callers do not need to normalise before looking up.
_LABEL_TO_BAND = {
    # Critical
    "critical":     SeverityBand.CRITICAL,
    # At risk / High
    "at risk":      SeverityBand.AT_RISK,
    "at_risk":      SeverityBand.AT_RISK,
    "high":         SeverityBand.AT_RISK,
    # Watch / Medium
    "watch":        SeverityBand.WATCH,
    "medium":       SeverityBand.WATCH,
    "med":          SeverityBand.WATCH,
    # Ready / Low
    "ready":        SeverityBand.READY,
    "low":          SeverityBand.READY,
}


def from_label(label: Optional[str]) -> Optional[SeverityBand]:
    """Return the canonical band for ``label`` (readiness or impact vocabulary).

    Returns ``None`` for unknown / empty labels. Case-insensitive, tolerant
    of surrounding whitespace.
    """
    if not label:
        return None
    return _LABEL_TO_BAND.get(str(label).strip().lower())


def readiness_label(band: Optional[SeverityBand]) -> str:
    """Return the canonical readiness-ladder label for a band."""
    if band is None:
        return ""
    return {
        SeverityBand.CRITICAL: "Critical",
        SeverityBand.AT_RISK:  "At risk",
        SeverityBand.WATCH:    "Watch",
        SeverityBand.READY:    "Ready",
    }[band]


def impact_label(band: Optional[SeverityBand]) -> str:
    """Return the canonical impact-ladder label for a band."""
    if band is None:
        return ""
    return {
        SeverityBand.CRITICAL: "Critical",
        SeverityBand.AT_RISK:  "High",
        SeverityBand.WATCH:    "Medium",
        SeverityBand.READY:    "Low",
    }[band]


# ---------------------------------------------------------------------------
# Sort rank + weight + CSS + action
# ---------------------------------------------------------------------------

def band_rank(band: Optional[SeverityBand]) -> int:
    """Return a numeric urgency rank (higher = more urgent).

    Mirrors the historical mapping used by
    :mod:`services.questionnaire_enhancer` (Critical=4, Watch/Medium=2, ...).
    Unknown bands fall back to the middle band's rank so unclassified
    items still sort predictably.
    """
    if band is None:
        return 2
    return {
        SeverityBand.CRITICAL: 4,
        SeverityBand.AT_RISK:  3,
        SeverityBand.WATCH:    2,
        SeverityBand.READY:    0,
    }[band]


def weight_from_band(band: Optional[SeverityBand]) -> int:
    """Return the composite scoring weight for a band (1..5).

    Preserves the historical mapping used by
    :func:`services.questionnaire_enhancer._weight_from_severity`:
    Critical=5, High/At risk=4, Medium/Watch=3, Low/Ready=2.
    """
    if band is None:
        return 2
    return {
        SeverityBand.CRITICAL: 5,
        SeverityBand.AT_RISK:  4,
        SeverityBand.WATCH:    3,
        SeverityBand.READY:    2,
    }[band]


def css_class(band: Optional[SeverityBand]) -> str:
    """Return the dashboard CSS class name for a band.

    Preserves the historical class names used across ``app.py``:
    ``crit`` / ``risk`` / ``watch`` / ``ready`` / ``none``.
    """
    if band is None:
        return "none"
    return {
        SeverityBand.CRITICAL: "crit",
        SeverityBand.AT_RISK:  "risk",
        SeverityBand.WATCH:    "watch",
        SeverityBand.READY:    "ready",
    }[band]


def action_for_readiness(band: Optional[SeverityBand]) -> str:
    """Return the executive action string for a readiness band.

    Mirrors :func:`services.scoring_engine.cxo_status` verbatim so the
    UI copy stays identical.
    """
    if band is None:
        return ""
    return {
        SeverityBand.CRITICAL: "Escalate to governance and define funded remediation.",
        SeverityBand.AT_RISK:  "Prioritise remediation plan, owners and evidence.",
        SeverityBand.WATCH:    "Resolve targeted gaps before executive sign-off.",
        SeverityBand.READY:    "Maintain evidence and periodic validation.",
    }[band]


__all__ = [
    "SeverityBand",
    "READINESS_THRESHOLDS",
    "READINESS_LABELS",
    "IMPACT_LABELS",
    "readiness_band",
    "impact_band",
    "from_label",
    "readiness_label",
    "impact_label",
    "band_rank",
    "weight_from_band",
    "css_class",
    "action_for_readiness",
]
