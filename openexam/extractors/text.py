from __future__ import annotations

from pathlib import Path

from openexam.models import ExtractedSection
from openexam.priority_sources import is_exam_answer_bank_path, split_answer_bank_markdown_sections


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def extract_text_file(path: Path) -> list[ExtractedSection]:
    text = _read_text(path)
    if path.suffix.lower() in {".md", ".txt"} and is_exam_answer_bank_path(str(path)):
        answer_sections = split_answer_bank_markdown_sections(text)
        if answer_sections:
            return [
                ExtractedSection(
                    source_path=path,
                    file_name=path.name,
                    location_type="section",
                    location_label=f"section.{index}",
                    paragraph_index=index,
                    text=section,
                )
                for index, section in enumerate(answer_sections, start=1)
            ]

    paragraphs = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    sections: list[ExtractedSection] = []
    paragraph_index = 0
    for block in paragraphs:
        if not block:
            continue
        paragraph_index += 1
        sections.append(
            ExtractedSection(
                source_path=path,
                file_name=path.name,
                location_type="paragraph",
                location_label=f"para.{paragraph_index}",
                paragraph_index=paragraph_index,
                text=block,
            )
        )
    if not sections and text.strip():
        sections.append(
            ExtractedSection(
                source_path=path,
                file_name=path.name,
                location_type="paragraph",
                location_label="para.1",
                paragraph_index=1,
                text=text.strip(),
            )
        )
    return sections
