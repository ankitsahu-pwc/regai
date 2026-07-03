"""PDF text extraction utilities backed by PyMuPDF (``fitz``).

Used by Phase 7's Page 1 ("Upload regulation document"). The extracted text is
fed back as additional context to the BRD/FRD generator in Phase 4 — per the
Phase 1 design choice.

PyMuPDF is the dependency named in the project's requirements. We keep these
helpers small and fault-tolerant so a scanned/encrypted PDF degrades to an
empty string with a captured ``warning_message`` rather than crashing the app.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - the requirements file enforces availability.
    raise ImportError(
        "PyMuPDF is required for utils.pdf_parser. Install it with `pip install PyMuPDF`."
    ) from exc

PdfSource = Union[str, Path, bytes, io.IOBase]


@dataclass
class PdfExtractionResult:
    """The structured outcome of a PDF parse."""

    text: str
    page_count: int
    is_encrypted: bool
    warning_message: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _open_document(source: PdfSource) -> "fitz.Document":
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return fitz.open(str(path))
    if isinstance(source, bytes):
        return fitz.open(stream=source, filetype="pdf")
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return fitz.open(stream=data, filetype="pdf")
    raise TypeError(f"Unsupported PDF source type: {type(source).__name__}")


def extract_pdf_pages(source: PdfSource) -> List[str]:
    """Return a list of page texts. Empty strings are kept so indices align with page numbers."""
    doc = _open_document(source)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            return []
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()


def extract_pdf_text(source: PdfSource, max_chars: Optional[int] = None) -> PdfExtractionResult:
    """Return the full text of the PDF (page-joined) with simple diagnostics.

    ``max_chars`` is an optional cap useful when piping text into a token-bounded
    LLM context window.
    """
    doc = _open_document(source)
    try:
        encrypted = bool(doc.is_encrypted)
        if encrypted and not doc.authenticate(""):
            return PdfExtractionResult(
                text="",
                page_count=doc.page_count,
                is_encrypted=True,
                warning_message="PDF is password-protected; no text extracted.",
            )

        pages: List[str] = []
        for page in doc:
            try:
                pages.append(page.get_text("text"))
            except Exception as exc:  # pragma: no cover - defensive
                pages.append("")
                warning = f"Failed to extract page {page.number}: {exc}"
                return PdfExtractionResult(
                    text="\n\n".join(pages),
                    page_count=doc.page_count,
                    is_encrypted=encrypted,
                    warning_message=warning,
                )

        text = "\n\n".join(p.strip() for p in pages if p.strip())
        warning: Optional[str] = None
        if not text:
            warning = (
                "PDF appears to contain no extractable text. It may be a scanned image. "
                "Consider OCR (e.g. ocrmypdf) before re-uploading."
            )
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return PdfExtractionResult(
            text=text,
            page_count=doc.page_count,
            is_encrypted=encrypted,
            warning_message=warning,
        )
    finally:
        doc.close()


__all__ = [
    "PdfExtractionResult",
    "PdfSource",
    "extract_pdf_pages",
    "extract_pdf_text",
]
