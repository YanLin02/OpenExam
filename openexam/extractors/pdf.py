from __future__ import annotations

from pathlib import Path

from openexam.extractors.base import ExtractionError
from openexam.models import ExtractedSection


def extract_pdf(path: Path) -> list[ExtractedSection]:
    try:
        import fitz
    except ImportError as exc:
        raise ExtractionError("PyMuPDF is not installed. Install pymupdf to parse PDF files.") from exc

    sections: list[ExtractedSection] = []
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                sections.append(
                    ExtractedSection(
                        source_path=path,
                        file_name=path.name,
                        location_type="page",
                        location_label=f"p.{page_index}",
                        page_number=page_index,
                        text=text,
                    )
                )
    except Exception as exc:
        raise ExtractionError(f"Failed to parse PDF: {exc}") from exc
    return sections
