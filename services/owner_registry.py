"""Shared owner-by-function registry.

Historical duplication
----------------------
Prior to this module, the impacted-function -> suggested-owner mapping was
duplicated in two places:

* ``services/recommendation_service.py`` (legacy compact recommendations)
* ``services/rich_recommendation_service.py`` (consulting-grade recommendations)

The two copies had drifted (one contained ``"DORA Programme Manager"``, the
other ``"Programme Manager"``). This module is now the single source of
truth so both recommendation services stay in lock-step.

Adding a new impacted function only requires editing this file.
"""

from __future__ import annotations

from typing import Dict, Optional


#: Canonical impacted-function -> suggested-owner mapping.
#:
#: The mapping is intentionally regulation-agnostic ("Programme Manager"
#: rather than "DORA Programme Manager") so the same catalogue works for
#: any regulation the platform ingests.
OWNER_BY_FUNCTION: Dict[str, str] = {
    "Execution / Client Activity":       "Front Office / Business Owner",
    "Risk Management":                   "Chief Risk Officer",
    "Compliance & Legal":                "Chief Compliance Officer",
    "Technology / IT Operations":        "Chief Technology Officer",
    "Cyber Security":                    "Chief Information Security Officer",
    "Business Continuity / Resilience":  "Business Continuity Manager",
    "Incident Management":               "ICT Incident Response Lead",
    "Vendor / Third-Party Management":   "Head of Vendor / Third-Party Risk",
    "Data Governance / Reporting":       "Chief Data Officer",
    "Internal Audit / Assurance":        "Head of Internal Audit",
    "Operations / Settlement":           "Head of Operations",
    "Programme Management":              "Programme Manager",
    "Human Resources / Training":        "Head of HR / Talent",
}


def owner_for(
    function: Optional[str],
    *,
    fallback: str = "Compliance / Programme Owner",
) -> str:
    """Return the suggested owner for ``function``.

    Parameters
    ----------
    function
        Impacted business function label. When ``None`` or empty the
        ``fallback`` is returned directly (no lookup is attempted).
    fallback
        String returned when ``function`` is empty or absent from the
        catalogue. Defaults to the compact-recommendation fallback
        (``"Compliance / Programme Owner"``); callers that prefer a
        different sentinel (e.g. the rich recommendation service uses
        ``"Executive Sponsor"``) can override it explicitly.
    """
    if not function:
        return fallback
    return OWNER_BY_FUNCTION.get(function, fallback)


__all__ = ["OWNER_BY_FUNCTION", "owner_for"]
