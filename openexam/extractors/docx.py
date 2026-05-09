from __future__ import annotations

from pathlib import Path

from openexam.extractors.base import ExtractionError
from openexam.models import ExtractedSection


def extract_docx(path: Path) -> list[ExtractedSection]:
    try:
        from docx import Document
    except ImportError as exc:
        raise ExtractionError("python-docx is not installed. Install python-docx to parse DOCX files.") from exc

    sections: list[ExtractedSection] = []
    try:
        document = Document(path)
        paragraph_index = 0
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            paragraph_index += 1
            sections.append(
                ExtractedSection(
                    source_path=path,
                    file_name=path.name,
                    location_type="paragraph",
                    location_label=f"para.{paragraph_index}",
                    paragraph_index=paragraph_index,
                    text=text,
                )
            )
    except Exception as exc:
        raise ExtractionError(f"Failed to parse DOCX: {exc}") from exc
    return sections
