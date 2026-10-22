"""PDF loading: one record per page with file name and 1-based page number."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from app.core.errors import RetrievalError


@dataclass
class PageText:
    file: str
    page: int
    text: str


def _page_text(page) -> str:
    """Text blocks in reading order, separated by blank lines so paragraph structure survives."""
    blocks = page.get_text("blocks", sort=True)
    return "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())


def list_pdfs(documents_dir: Path) -> list[Path]:
    if not documents_dir.is_dir():
        raise RetrievalError(f"documents directory not found: {documents_dir}")
    return sorted(documents_dir.glob("*.pdf"))


def load_pdf(path: Path) -> list[PageText]:
    try:
        with fitz.open(path) as doc:
            return [PageText(file=path.name, page=i + 1, text=_page_text(page)) for i, page in enumerate(doc)]
    except Exception as exc:  # PyMuPDF raises various types for corrupt files
        raise RetrievalError(f"could not read PDF {path.name}: {exc}") from exc


def load_documents(documents_dir: Path) -> list[PageText]:
    pdfs = list_pdfs(documents_dir)
    if not pdfs:
        raise RetrievalError(f"no PDF files found in {documents_dir}")
    pages: list[PageText] = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    return pages
