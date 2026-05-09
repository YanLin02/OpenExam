from __future__ import annotations

from pathlib import Path

from openexam.config import SUPPORTED_EXTENSIONS
from openexam.extractors.base import ExtractionError
from openexam.extractors.docx import extract_docx
from openexam.extractors.pdf import extract_pdf
from openexam.extractors.pptx import extract_pptx
from openexam.extractors.text import extract_text_file
from openexam.models import ExtractedSection


def extract_file(path: Path) -> list[ExtractedSection]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        return extract_text_file(path)
    if suffix == ".docx":
        return extract_docx(path)
    if suffix == ".pptx":
        return extract_pptx(path)
    raise ExtractionError(f"Unsupported file extension: {suffix}")
