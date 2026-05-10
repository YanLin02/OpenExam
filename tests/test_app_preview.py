from __future__ import annotations

from openexam.app import should_show_pdf_preview


def test_should_show_pdf_preview_only_for_pdf_pages() -> None:
    assert should_show_pdf_preview("/tmp/sample.pdf", 1)
    assert should_show_pdf_preview("/tmp/SAMPLE.PDF", 2)
    assert not should_show_pdf_preview("/tmp/sample.pdf", None)
    assert not should_show_pdf_preview("/tmp/sample.docx", 1)
    assert not should_show_pdf_preview("/tmp/sample.pptx", 1)
    assert not should_show_pdf_preview("/tmp/sample.txt", 1)
    assert not should_show_pdf_preview("/tmp/sample.md", 1)
