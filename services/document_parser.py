"""Unified document parser used by the Document Parser pipeline stage.

This module is the single entry point for turning an on-disk regulation, BRD,
or FRD file (PDF or DOCX) into a :class:`~models.workflow_models.ParsedDocument`
that downstream agents can consume.

Reuses existing logic; does not duplicate:
- :mod:`utils.pdf_parser` for PDF text extraction (PyMuPDF).
- :mod:`utils.docx_parser` for DOCX text extraction.

The functions here intentionally fail soft: if a PDF is encrypted or a DOCX
cannot be opened, they return a ``ParsedDocument`` with ``warning_message`` set
and an empty body, so the orchestrator can continue with the deterministic
offline fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from models.workflow_models import ParsedDocument
from utils.docx_parser import DocxSource, extract_full_text
from utils.pdf_parser import PdfSource, extract_pdf_text

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Low-level helpers (PDF / DOCX) — thin wrappers around utils.*
# ---------------------------------------------------------------------------

def parse_pdf(source: PdfSource, *, name: Optional[str] = None,
              kind: str = "regulation") -> ParsedDocument:
    """Extract text from a PDF source into a ``ParsedDocument``."""
    result = extract_pdf_text(source)
    return ParsedDocument(
        name=name or _name_from_source(source),
        kind=kind,
        text=result.text,
        source_path=str(source) if isinstance(source, (str, Path)) else None,
        page_count=result.page_count,
        mime="application/pdf",
        warning_message=result.warning_message,
        metadata={"is_encrypted": result.is_encrypted},
    )


def parse_docx(source: DocxSource, *, name: Optional[str] = None,
               kind: str = "brd") -> ParsedDocument:
    """Extract paragraphs + flattened table cells from a DOCX into a ``ParsedDocument``."""
    text = extract_full_text(source, include_tables=True)
    return ParsedDocument(
        name=name or _name_from_source(source),
        kind=kind,
        text=text,
        source_path=str(source) if isinstance(source, (str, Path)) else None,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        warning_message=None if text.strip() else "DOCX appears to contain no extractable text.",
    )


# ---------------------------------------------------------------------------
# High-level dispatch
# ---------------------------------------------------------------------------

def parse_document(path: PathLike, *, kind: str = "regulation") -> ParsedDocument:
    """Detect file type by extension and dispatch to the right parser.

    Falls back to an empty ``ParsedDocument`` (with a warning) for unsupported
    extensions, so the orchestrator can decide whether to continue.
    """
    p = Path(path)
    if not p.exists():
        return ParsedDocument(
            name=p.name,
            kind=kind,
            text="",
            source_path=str(p),
            warning_message=f"File not found: {p}",
        )

    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            return parse_pdf(p, name=p.name, kind=kind)
        if suffix == ".docx":
            return parse_docx(p, name=p.name, kind=kind)
    except Exception as exc:  # pragma: no cover - defensive
        return ParsedDocument(
            name=p.name,
            kind=kind,
            text="",
            source_path=str(p),
            warning_message=f"Failed to parse {p.name}: {exc}",
        )

    return ParsedDocument(
        name=p.name,
        kind=kind,
        text="",
        source_path=str(p),
        warning_message=f"Unsupported file type for parser: {suffix}",
    )


def _name_from_source(source) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return "uploaded_document"


__all__ = [
    "parse_document",
    "parse_pdf",
    "parse_docx",
    "ParsedDocument",
]
