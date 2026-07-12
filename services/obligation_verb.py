"""Regulatory obligation-verb classifier.

Distinguishes the five canonical strength verbs used in regulatory text
(**Must / Shall / Should / May / Can**), independent of the MoSCoW
priority ladder used for BRD requirement management. The verb captures
what the regulator actually *said*; MoSCoW captures how the delivery team
plans to *sequence* the work.

Ladder (strongest -> weakest):

* ``Must``   — hard mandatory obligation.
* ``Shall``  — mandatory in regulatory drafting convention (equivalent
  to ``Must`` for enforceability, kept distinct so we preserve the
  source drafter's word choice).
* ``Should`` — strong recommendation; departure requires justification.
* ``May``    — permissive / discretionary language; the trigger for
  risk analysis (RAG rating + industry practice + consequences of
  taking the option).
* ``Can``    — capability statement (usually informational, sometimes
  synonymous with ``May`` depending on the drafter).

The classifier is intentionally rule-based (word-boundary regex on a
lowercased haystack). It is fast, deterministic, and cheap enough to
run on every obligation without a GenAI call. Callers can override
with an LLM classification later if the deterministic verdict is
``""`` (no match).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple


#: Canonical verbs ordered from strongest to weakest. Preserved as the
#: authoritative ordering so downstream code can compare / sort by strength.
CANONICAL_VERBS: Tuple[str, ...] = ("Must", "Shall", "Should", "May", "Can")


#: Word-boundary regex patterns keyed by canonical verb. Kept as a module-level
#: constant so re.compile happens once at import time.
_VERB_PATTERNS = {
    verb: re.compile(rf"\b{verb.lower()}\b", flags=re.IGNORECASE)
    for verb in CANONICAL_VERBS
}


#: Lowercased canonical verb spellings, plus common inflections we want to
#: normalise to a canonical entry. ``"required to"``/``"is required"``/
#: ``"is obliged to"`` etc. are treated as ``Must``; ``"has the option to"``
#: is treated as ``May``. Kept small on purpose so the classifier stays
#: conservative — a low-precision classifier would silently downgrade
#: mandatory language.
_INFLECTION_HINTS = (
    ("must", "Must"),
    ("required to", "Must"),
    ("is required", "Must"),
    ("are required", "Must"),
    ("obliged to", "Must"),
    ("is obligated to", "Must"),
    ("shall", "Shall"),
    ("should", "Should"),
    ("is expected to", "Should"),
    ("are expected to", "Should"),
    ("recommended to", "Should"),
    ("may", "May"),
    ("has the option to", "May"),
    ("at its discretion", "May"),
    ("can", "Can"),
    ("is able to", "Can"),
    ("are able to", "Can"),
)


def classify_verb(text: Optional[str]) -> str:
    """Return the strongest canonical verb detected in ``text``.

    Search order is strongest -> weakest so a clause that contains both
    "must" and "may" is classified as ``Must``. Inflected phrases like
    "required to" / "is expected to" are recognised as their canonical
    equivalent.

    Returns the empty string when no verb is detected. Callers that need
    a default should supply their own fallback (typically ``"Should"``
    to preserve the historical MoSCoW default without overstating the
    obligation).
    """
    if not text:
        return ""
    haystack = str(text).lower()

    # Fast path: direct word-boundary hit on a canonical verb.
    for verb in CANONICAL_VERBS:
        if _VERB_PATTERNS[verb].search(haystack):
            return verb

    # Slow path: inflected phrase hint.
    for phrase, canonical in _INFLECTION_HINTS:
        if phrase in haystack:
            return canonical

    return ""


def classify_verb_from_sources(texts: Iterable[Optional[str]]) -> str:
    """Return the strongest verb detected across a sequence of texts.

    Useful when the caller has multiple candidate strings (obligation
    title, compliance requirement, source snippet). Any one of them
    carrying "must" wins over a "should" elsewhere.
    """
    strongest_idx: Optional[int] = None
    for text in texts:
        verb = classify_verb(text)
        if not verb:
            continue
        idx = CANONICAL_VERBS.index(verb)
        if strongest_idx is None or idx < strongest_idx:
            strongest_idx = idx
        # No point continuing once we've seen the strongest possible verb.
        if strongest_idx == 0:
            break
    return CANONICAL_VERBS[strongest_idx] if strongest_idx is not None else ""


def is_mandatory(verb: Optional[str]) -> bool:
    """Return True when the verb signals a hard obligation (Must / Shall)."""
    return (verb or "").strip().lower() in {"must", "shall"}


def is_discretionary(verb: Optional[str]) -> bool:
    """Return True when the verb signals discretionary language (May / Can).

    Used by the risk-analysis pipeline to decide which obligations need a
    risk / RAG / consequences enrichment. (Note: the risk enrichment
    itself is a separate feature and is currently parked.)
    """
    return (verb or "").strip().lower() in {"may", "can"}


__all__ = [
    "CANONICAL_VERBS",
    "classify_verb",
    "classify_verb_from_sources",
    "is_mandatory",
    "is_discretionary",
]
