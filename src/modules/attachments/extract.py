"""Text extraction for uploads (PR 15 / decision 14).

Dispatch by suffix: txt/md/csv via stdlib, pdf via pypdf, docx via
python-docx, html via trafilatura. All parsers are sync and CPU-bound —
callers run them through `asyncio.to_thread` (repo convention). Every
extraction is capped to ATTACHMENT_MAX_CHARS; a file that yields
nothing usable is an error the upload route reports per file.
"""

import csv
import io
import logging
from pathlib import Path

from ...tools.settings import settings
from .model import (
    DOCX_SUFFIXES,
    HTML_SUFFIXES,
    PDF_SUFFIXES,
    TEXT_SUFFIXES,
)

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """The file could not be parsed into text."""


def _cap(text: str) -> str:
    return text[: settings.ATTACHMENT_MAX_CHARS]


def _extract_text_types(data: bytes) -> str:
    try:
        return _cap(data.decode("utf-8"))
    except UnicodeDecodeError:
        return _cap(data.decode("utf-8", errors="replace"))


def _extract_csv(data: bytes) -> str:
    """CSV → markdown-ish pipe table so the chunker keeps structure."""
    reader = csv.reader(io.StringIO(_extract_text_types(data)))
    rows = [" | ".join(row) for row in reader]
    return _cap("\n".join(rows))


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [
        (page.extract_text() or "") for page in reader.pages
    ]
    return _cap("\n\n".join(pages))


def _extract_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    return _cap("\n".join(parts))


def _extract_html(data: bytes) -> str:
    import trafilatura

    extracted = trafilatura.extract(
        _extract_text_types(data).encode("utf-8"),
        include_comments=False,
        include_tables=True,
        output_format="markdown",
    )
    if not extracted:
        raise ExtractionError("No main content could be extracted.")
    return _cap(extracted)


def extract_text(filename: str, data: bytes) -> str:
    """Extract capped text from one upload; raises ExtractionError."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            return _extract_csv(data)
        if suffix in TEXT_SUFFIXES:
            return _extract_text_types(data)
        if suffix in PDF_SUFFIXES:
            return _extract_pdf(data)
        if suffix in DOCX_SUFFIXES:
            return _extract_docx(data)
        if suffix in HTML_SUFFIXES:
            return _extract_html(data)
    except ExtractionError:
        raise
    except Exception as exc:
        logger.debug("Extraction failed for %s: %s", filename, exc)
        raise ExtractionError(
            f"'{filename}' could not be parsed ({type(exc).__name__})."
        ) from exc
    raise ExtractionError(
        f"'{filename}': unsupported type. Allowed: txt, md, csv, pdf, "
        f"docx, html."
    )
