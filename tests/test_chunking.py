from __future__ import annotations

from pathlib import Path

from openexam.chunking import chunk_sections
from openexam.models import ExtractedSection


def test_chunking_preserves_source_and_overlap() -> None:
    text = "A" * 500 + "Transformer attention " + "B" * 700
    section = ExtractedSection(
        source_path=Path("/tmp/course.pdf"),
        file_name="course.pdf",
        location_type="page",
        location_label="p.3",
        page_number=3,
        text=text,
    )

    chunks = chunk_sections([section], chunk_size=800, overlap=120)

    assert len(chunks) == 2
    assert chunks[0].source_path == Path("/tmp/course.pdf")
    assert chunks[0].file_name == "course.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_id
    assert chunks[1].char_start is not None
    assert chunks[0].char_end is not None
    assert chunks[1].char_start < chunks[0].char_end
