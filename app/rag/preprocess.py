"""Text cleaning for extracted PDF pages.

Steps (each is deterministic and unit-tested):
1. normalise unicode (NFKC) and line endings; drop control characters
2. re-join words hyphenated across line breaks ("subscrip-\\ntion" -> "subscription")
3. drop lines that are only a page number ("3", "Page 3", "3 / 10")
4. drop header/footer lines repeated on at least half of a document's pages (needs >= 3 pages)
5. collapse intra-paragraph line breaks into spaces, keep paragraph breaks, squeeze whitespace
6. discard pages whose cleaned text is too short to carry information
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.rag.loader import PageText

MIN_PAGE_CHARS = 40

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text)


def _repeated_lines(pages: list[PageText]) -> set[str]:
    """Lines that appear on >= half the pages of one file (running headers / footers)."""
    by_file: dict[str, list[PageText]] = {}
    for p in pages:
        by_file.setdefault(p.file, []).append(p)
    repeated: set[str] = set()
    for file_pages in by_file.values():
        if len(file_pages) < 3:
            continue
        counts: Counter[str] = Counter()
        for p in file_pages:
            counts.update({ln.strip() for ln in _normalise(p.text).split("\n") if ln.strip()})
        threshold = max(2, len(file_pages) / 2)
        repeated |= {ln for ln, c in counts.items() if c >= threshold and len(ln) < 120}
    return repeated


def clean_text(text: str, drop_lines: set[str] | None = None) -> str:
    drop_lines = drop_lines or set()
    text = _HYPHEN_BREAK.sub(r"\1\2", _normalise(text))
    kept = [ln for ln in text.split("\n") if not _PAGE_NUMBER_LINE.match(ln) and ln.strip() not in drop_lines]
    # Paragraphs are separated by blank lines; single newlines inside them are layout wrapping.
    paragraphs, current = [], []
    for ln in kept:
        if ln.strip():
            current.append(ln.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(re.sub(r"[ \t]+", " ", p) for p in paragraphs).strip()


def preprocess_pages(pages: list[PageText]) -> list[PageText]:
    drop = _repeated_lines(pages)
    cleaned = []
    for p in pages:
        text = clean_text(p.text, drop)
        if len(text) >= MIN_PAGE_CHARS:
            cleaned.append(PageText(file=p.file, page=p.page, text=text))
    return cleaned
