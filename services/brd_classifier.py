"""AI-based classifier for uploaded BRD/FRD documents.

Purpose
-------
The strict table parser in :mod:`services.questionnaire_generator`
(``read_docx_requirements``) requires an app-generated BRD with a very
specific column layout (``ID | Category | Requirement | Detailed Requirement
| <regulation> Alignment | Priority | Acceptance Criteria``). Real-world
BRDs — especially those authored by hand in Word, exported from third-party
tools, or written before this app existed — rarely conform to that exact
schema. When the strict parser returns zero rows the questionnaire pipeline
has nothing to work with and Agent 3 cannot fire.

This module fills that gap. Given the full text of an uploaded BRD DOCX
and a live :class:`~services.genai_service.GenAIClient`, it asks the LLM
to classify the document's content into two structured buckets:

* **requirements** — concrete business/functional requirements the BRD
  imposes on the delivery team (what must be built or configured).
* **obligations** — the underlying regulatory duties the BRD is aligning
  to (what the regulator is asking for). These are what Agent 1 normally
  produces, but a BRD authored downstream of Agent 1 typically restates
  them in its own words, so we can recover a workable obligation list
  from the uploaded document itself.

The output plugs straight into the existing questionnaire pipeline via
:func:`services.questionnaire_generator.build_questionnaire_package` — no
schema changes are required downstream.

Design notes
------------
* Purely opt-in fallback. When the strict table parser finds even one
  requirement row we skip the LLM entirely (cheap and deterministic
  wins).
* When the client is unavailable (offline mode, missing API key,
  network failure) the function returns ``None`` and the caller falls
  back to the strict parser's original ``ValueError`` behaviour. We
  never fabricate requirements without an LLM in the loop.
* Single structured LLM roundtrip. The prompt asks for both requirements
  and obligations in one shot to keep latency and cost predictable.
* Output records are Pydantic models (v1 compat via the same shim used
  elsewhere in the codebase), which the caller converts to the module-
  local ``Requirement`` dataclass / ``Obligation`` shape before passing
  them to the deterministic parts of the pipeline.

This module is intentionally small — it does one thing (structure a
free-form BRD) and delegates every downstream concern (impact-pair
derivation, question generation, scoring, role-filtering) to the
existing pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

try:
    from pydantic.v1 import BaseModel, Field
except Exception:  # pragma: no cover - pydantic v2 fallback
    from pydantic import BaseModel, Field  # type: ignore

logger = logging.getLogger(__name__)


# Hard cap on how much DOCX text we forward to the LLM in a single call.
# The rest of the app's LLM calls stay well under ~30k characters of
# component-specific prompt, so we mirror that ceiling here. Very long
# BRDs get truncated at the character boundary — the earlier text tends
# to contain the executive summary + main requirement sections which is
# what we most want to classify. We warn on truncation so callers can
# see it in the logs.
_MAX_DOCX_TEXT_CHARS = 28_000


class ClassifiedRequirement(BaseModel):
    """One requirement recovered from a free-form BRD.

    Fields mirror the module-local
    :class:`services.questionnaire_generator.Requirement` dataclass so a
    ``ClassifiedRequirement`` can be lifted into that shape by the
    caller without any translation layer beyond field-name mapping.
    Optional fields default to empty strings so the LLM can omit them
    when the BRD text does not contain the corresponding signal.
    """

    source_id: str = Field(
        default="",
        description=(
            "Short human-readable identifier for this requirement — either "
            "quoted verbatim from the BRD (e.g. 'FR-001', '3.2.1') or, when "
            "the BRD does not carry explicit IDs, a synthetic short token "
            "the LLM invents (e.g. 'INCIDENT-01'). Kept for traceability "
            "back to the source BRD."
        ),
    )
    category: str = Field(
        default="",
        description=(
            "Category label from the BRD such as 'Governance', 'Incident "
            "Management', 'Third-Party Risk'. Empty when not stated."
        ),
    )
    requirement: str = Field(
        ...,
        description=(
            "One-line requirement title — imperative voice, plain English, "
            "under 20 words. This is the short handle downstream agents use "
            "when labelling generated questions."
        ),
    )
    detail: str = Field(
        default="",
        description=(
            "Full detail sentence(s) explaining what the delivery team must "
            "do to satisfy the requirement. This is what Agent 3 uses to "
            "generate probing questions."
        ),
    )
    alignment: str = Field(
        default="",
        description=(
            "How this requirement aligns to the regulation — quote the "
            "specific article/section number when the BRD states it "
            "(e.g. 'DORA Art. 5', 'GDPR Art. 32'). Empty when the BRD does "
            "not tie the requirement to a specific regulatory reference."
        ),
    )
    priority: str = Field(
        default="Should",
        description=(
            "MoSCoW priority — one of Must / Should / Could / Won't. "
            "Default to 'Should' when the BRD does not state a priority."
        ),
    )
    acceptance: str = Field(
        default="",
        description=(
            "Acceptance criteria from the BRD in plain English. Empty when "
            "the BRD does not specify how compliance is verified."
        ),
    )


class ClassifiedObligation(BaseModel):
    """One regulatory obligation recovered from a free-form BRD.

    Field names align with the ``Obligation`` dataclass in
    :mod:`models.workflow_models` so downstream code that already reads
    those attributes on Agent 1 output can consume classified
    obligations without change.
    """

    obligation_id: str = Field(
        default="",
        description=(
            "Short identifier for the obligation — either quoted from the "
            "BRD (e.g. 'OBL-01', 'REQ-3.2') or invented as a short "
            "kebab-case token (e.g. 'incident-report'). Used to cross-link "
            "obligations to the requirements they justify."
        ),
    )
    title: str = Field(
        ...,
        description=(
            "One-line obligation title in the regulator's voice ('The firm "
            "shall...'). Under 20 words."
        ),
    )
    theme: str = Field(
        default="",
        description=(
            "Regulatory theme (e.g. 'ICT Risk Management', 'Data "
            "Protection', 'Incident Reporting'). Empty when no theme is "
            "evident."
        ),
    )
    compliance_requirement: str = Field(
        default="",
        description=(
            "Full description of what compliance looks like — what the "
            "firm must do or evidence to be in scope for this obligation."
        ),
    )
    impacted_area: str = Field(
        default="",
        description=(
            "Business area primarily impacted (e.g. 'Operations', "
            "'Technology', 'Compliance'). Empty when not derivable."
        ),
    )
    impacted_function: str = Field(
        default="",
        description=(
            "Business function primarily impacted (e.g. 'Third-Party "
            "Management', 'Incident Response'). Empty when not derivable."
        ),
    )
    control_expectations: List[str] = Field(
        default_factory=list,
        description=(
            "Short list (0-5 entries) of the specific control activities "
            "the regulator expects — e.g. 'Maintain a register of ICT "
            "third-party providers'. Kept short and imperative."
        ),
    )
    evidence_needs: List[str] = Field(
        default_factory=list,
        description=(
            "Short list (0-5 entries) of evidence artefacts that would "
            "demonstrate compliance — e.g. 'Signed incident-response "
            "runbook', 'Board minutes approving policy'."
        ),
    )
    priority: str = Field(
        default="Should",
        description=(
            "MoSCoW priority derived from the regulator's language "
            "(mandatory verbs -> Must, expected -> Should, encouraged -> "
            "Could). Default 'Should'."
        ),
    )


class BRDClassification(BaseModel):
    """Container returned by :func:`classify_brd_document`.

    Deliberately kept flat so the caller can iterate the two lists
    without a nested unpack step.
    """

    requirements: List[ClassifiedRequirement] = Field(
        default_factory=list,
        description=(
            "All requirements the LLM could extract from the BRD text. "
            "Empty when the document contains no distinguishable "
            "requirement statements (e.g. it is a pure introduction / "
            "cover page)."
        ),
    )
    obligations: List[ClassifiedObligation] = Field(
        default_factory=list,
        description=(
            "All regulatory obligations the LLM could infer from the BRD "
            "text. Empty when the BRD does not reference any specific "
            "regulatory duties."
        ),
    )


_SYSTEM_INSTRUCTION = (
    "You are a regulatory business analyst. You will be given the raw text "
    "of a Business/Functional Requirements Document (BRD/FRD) that has "
    "been uploaded by a compliance practitioner. Your job is to classify "
    "the document's content into two structured lists so that a downstream "
    "questionnaire generator can build assessment questions from it.\n\n"
    "1. REQUIREMENTS: concrete things the delivery team must build, "
    "configure, or operationalise. Each requirement is a discrete unit of "
    "work — split multi-part statements into separate items. Preserve any "
    "IDs the BRD carries (e.g. 'FR-001', '3.2.1'); when none exist, "
    "invent a short kebab-case token that describes the requirement.\n\n"
    "2. OBLIGATIONS: the underlying regulatory duties the requirements are "
    "aligning to (what the regulator is asking for, not what the delivery "
    "team is building). These are typically expressed in the regulator's "
    "voice: 'The firm shall...', 'Institutions must...'. If the BRD "
    "quotes or paraphrases the regulation, extract those statements as "
    "obligations. If the BRD only describes the solution without "
    "referencing the underlying duty, you may leave the obligations list "
    "empty rather than fabricate one.\n\n"
    "Strict rules:\n"
    " - Do NOT invent requirements or obligations that are not present in "
    "the text. Missing information is fine — return fewer, higher-quality "
    "items rather than padding with speculation.\n"
    " - Do NOT copy huge blocks of text verbatim into the 'detail' or "
    "'compliance_requirement' fields. Rephrase concisely in plain "
    "English (2-4 sentences maximum per field).\n"
    " - Requirements come from the delivery team's perspective "
    "(imperative voice: 'Implement...', 'Configure...', 'Maintain...'). "
    "Obligations come from the regulator's perspective (mandatory "
    "modal verbs: 'shall', 'must', 'is required to').\n"
    " - When a requirement clearly aligns to a specific regulation "
    "article/section (BRD text says 'per Article 5' or 'as required by "
    "GDPR Art. 32'), quote that reference in the 'alignment' field. "
    "Otherwise leave it empty.\n"
    " - If the BRD text is empty, purely a cover page, or contains no "
    "distinguishable requirement statements, return both lists empty. "
    "That is a valid outcome."
)


def classify_brd_document(
    client: Any,
    *,
    docx_text: str,
    regulation: str,
) -> Optional[BRDClassification]:
    """Classify a BRD's free-form text into requirements + obligations.

    Parameters
    ----------
    client
        A live :class:`~services.genai_service.GenAIClient`. When
        ``None`` we return ``None`` immediately so the caller can fall
        back to its error path. We never fabricate output without the
        LLM in the loop.
    docx_text
        Full text of the uploaded DOCX, as produced by
        :func:`utils.docx_parser.extract_full_text`. Truncated to
        :data:`_MAX_DOCX_TEXT_CHARS` when longer to keep the prompt
        within the LLM's context budget.
    regulation
        The regulation code/name the BRD is meant to align to (e.g.
        ``"DORA"``, ``"HOUSING FINANCE"``). Used to prime the LLM's
        expectations about which regulatory vocabulary to look for.

    Returns
    -------
    Optional[BRDClassification]
        The parsed classification, or ``None`` when the client is
        unavailable or the LLM call fails. Callers must treat ``None``
        as "fallback unavailable" and surface the original
        strict-parser error to the user.
    """
    text = (docx_text or "").strip()
    if client is None or not text:
        return None

    if len(text) > _MAX_DOCX_TEXT_CHARS:
        logger.warning(
            "BRD classifier: DOCX text truncated for LLM. "
            "original_chars=%d kept_chars=%d",
            len(text), _MAX_DOCX_TEXT_CHARS,
        )
        text = text[:_MAX_DOCX_TEXT_CHARS]

    reg = (regulation or "").strip() or "the selected regulation"
    prompt = (
        f"Regulation the BRD is aligning to: {reg}\n\n"
        f"Raw BRD/FRD text (paragraphs + flattened table cells, in "
        f"document order):\n"
        f"------- BEGIN BRD -------\n{text}\n------- END BRD -------\n\n"
        f"Populate the BRDClassification schema. Extract every distinct "
        f"requirement and every distinct regulatory obligation you can "
        f"identify from the text above. Follow the rules in the system "
        f"instruction — quality over quantity, no fabrication."
    )

    try:
        result: BRDClassification = client.generate(
            schema_model=BRDClassification,
            component_name="BRD classification (requirements + obligations)",
            component_instruction=prompt,
            context="",
            system_instruction=_SYSTEM_INSTRUCTION,
            regulation=regulation,
        )
    except Exception:
        logger.exception(
            "BRD classifier LLM call FAILED. regulation=%s text_chars=%d",
            regulation, len(text),
        )
        return None

    if result is None:
        return None

    reqs = list(result.requirements or [])
    obls = list(result.obligations or [])
    logger.info(
        "BRD classifier extracted %d requirements and %d obligations "
        "from uploaded DOCX. regulation=%s text_chars=%d",
        len(reqs), len(obls), regulation, len(text),
    )
    return result


__all__ = [
    "BRDClassification",
    "ClassifiedObligation",
    "ClassifiedRequirement",
    "classify_brd_document",
]
