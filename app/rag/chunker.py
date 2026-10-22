"""Sentence-aware chunking that never crosses a page boundary, so every chunk has an exact
(file, page) citation. Size and overlap are measured in characters and are configurable."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.loader import PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_HEADING = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S.{0,80}$")


@dataclass
class Chunk:
    chunk_id: str
    file: str
    page: int
    section: str | None
    text: str

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: the section title adds retrieval context."""
        return f"{self.section}. {self.text}" if self.section else self.text


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    return "\n" not in p and len(p) <= 90 and bool(_HEADING.match(p)) and not p.endswith(".")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _hard_wrap(sentence: str, size: int) -> list[str]:
    """Break an over-long 'sentence' on spaces so no chunk blows the size budget."""
    words, out, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_page(page: PageText, chunk_size: int, overlap: int, section: str | None, start_index: int = 0):
    """Return (chunks, section_in_effect_at_page_end)."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")

    sentences: list[tuple[str, str | None]] = []  # (sentence, section)
    for para in re.split(r"\n{2,}", page.text):
        para = para.strip()
        if not para:
            continue
        if is_heading(para):
            section = para
            continue
        for s in split_sentences(para):
            for piece in ([s] if len(s) <= chunk_size else _hard_wrap(s, chunk_size)):
                sentences.append((piece, section))

    stem = Path(page.file).stem
    chunks: list[Chunk] = []
    current: list[tuple[str, str | None]] = []

    def flush():
        if not current:
            return
        idx = start_index + len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{stem}-p{page.page}-c{idx}", file=page.file, page=page.page,
            section=current[-1][1], text=" ".join(s for s, _ in current),
        ))

    def length(items):
        return sum(len(s) for s, _ in items) + max(0, len(items) - 1)

    for item in sentences:
        if current and length(current) + 1 + len(item[0]) > chunk_size:
            flush()
            # carry trailing sentences as overlap, but always make progress
            carried: list[tuple[str, str | None]] = []
            for prev in reversed(current):
                if length(carried + [prev]) > overlap:
                    break
                carried.insert(0, prev)
            current = carried
            if current and length(current) + 1 + len(item[0]) > chunk_size:
                current = []
        current.append(item)
    flush()
    return chunks, section


def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    section_by_file: dict[str, str | None] = {}
    counter_by_file: dict[str, int] = {}
    for page in pages:
        file_chunks, end_section = chunk_page(
            page, chunk_size, overlap, section_by_file.get(page.file), counter_by_file.get(page.file, 0)
        )
        section_by_file[page.file] = end_section
        counter_by_file[page.file] = counter_by_file.get(page.file, 0) + len(file_chunks)
        chunks.extend(file_chunks)
    return chunks
