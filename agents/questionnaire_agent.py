"""Agent 3 — Questionnaire Generation.

Role in the pipeline
--------------------
Takes the :class:`BRDArtifact` (and optionally :class:`RTMArtifact`) produced
by Agent 2 and emits a :class:`~models.workflow_models.QuestionnairePackage`.

The questionnaire is grouped by impacted area / function / control theme and
each question carries the scoring metadata required by the deterministic
Python Rules Engine (scoring weight, mapped requirement IDs, regulatory basis,
confidence).

Implementation strategy
-----------------------
* Reuses :mod:`services.questionnaire_generator` end-to-end so the output
  contract (package dict) remains identical to the existing schema validated
  by :mod:`utils.json_utils`. This keeps backward compatibility with the
  bundled sample package and any saved questionnaire JSON.
* Supports three input modes:

    1. ``from_report``    — closed-loop path from a generated BRD model.
    2. ``from_docx``      — parses an uploaded BRD/FRD DOCX.
    3. ``from_package``   — validates and loads an externally-supplied JSON
                            package (e.g. uploaded sample).

The bundled offline fallback is preserved because all three paths ultimately
delegate to :mod:`services.questionnaire_generator`, which works without
GenAI access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from models.workflow_models import BRDArtifact, QuestionnairePackage
from services.questionnaire_generator import (
    build_package_from_report,
    build_questionnaire_package,
)
from utils.json_utils import validate_package_schema

DocxPath = Union[str, Path]


class QuestionnaireAgent:
    """Agent 3 implementation."""

    def from_report(
        self,
        brd: BRDArtifact,
        *,
        regulation: str = "DORA",
        name: Optional[str] = None,
    ) -> QuestionnairePackage:
        """Build a questionnaire from a generated BRD in-memory model.

        Forwards the BRD's ``source_references_by_item`` map so each generated
        question carries the citation of its anchor BRD requirement on its
        explainability bundle (``explainability.source_references``).
        """
        if brd is None or brd.report is None:
            raise ValueError(
                "BRD artefact does not contain an in-memory report. "
                "Generate the BRD first or use from_docx() / from_package()."
            )
        source_refs = (brd.metadata or {}).get("source_references_by_item") or {}
        package = build_package_from_report(
            brd.report, regulation=regulation, source_refs_by_item=source_refs,
        )
        return QuestionnairePackage(
            package=package,
            source="generated_brd",
            name=name or f"{regulation} — generated from BRD",
        )

    def from_docx(
        self,
        path: DocxPath,
        *,
        regulation: str = "DORA",
        name: Optional[str] = None,
    ) -> QuestionnairePackage:
        """Parse a BRD/FRD DOCX and build a questionnaire."""
        package = build_questionnaire_package(str(path), regulation=regulation)
        return QuestionnairePackage(
            package=package,
            source="uploaded_brd",
            name=name or f"{regulation} — from {Path(path).stem}",
        )

    def from_package(
        self,
        package: Mapping[str, Any],
        *,
        source: str = "uploaded_json",
        name: Optional[str] = None,
    ) -> QuestionnairePackage:
        """Validate then load an existing questionnaire-package dict / JSON."""
        issues = validate_package_schema(package)
        if issues:
            raise ValueError(
                "Questionnaire package failed schema validation:\n  - "
                + "\n  - ".join(issues)
            )
        return QuestionnairePackage(
            package=dict(package),
            source=source,
            name=name,
        )


__all__ = ["QuestionnaireAgent"]
