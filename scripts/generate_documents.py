"""Generate the sample PDFs from scripts/document_sources.py (reproducible)."""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from app.core.config import get_settings
from scripts.document_sources import DOCUMENTS


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor="#555555", spaceAfter=14)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=8)

    written = []
    for filename, doc in DOCUMENTS.items():
        path = output_dir / filename
        pdf = SimpleDocTemplate(
            str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
            topMargin=2.2 * cm, bottomMargin=2.2 * cm, title=doc["title"], author="Acme Analytics",
            invariant=True,  # deterministic output for reproducible builds
        )
        story = []
        for i, (heading, paragraphs) in enumerate(doc["sections"]):
            if i:
                story.append(PageBreak())
            if i == 0:
                story += [Paragraph(doc["title"], h1), Paragraph(doc["version"], meta)]
            story.append(Paragraph(heading, h2))
            for text in paragraphs:
                story.append(Paragraph(text, body))
            story.append(Spacer(1, 6))
        pdf.build(story)
        written.append(path)
    return written


def main() -> int:
    out = get_settings().documents_dir
    for p in generate(out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
