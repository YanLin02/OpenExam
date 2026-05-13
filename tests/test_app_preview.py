from __future__ import annotations

from openexam.app import should_show_pdf_preview


def test_app_source_keeps_only_search_and_ask_modes() -> None:
    source = __import__("pathlib").Path(__import__("openexam.app").app.__file__).read_text(encoding="utf-8")

    assert 'options=["Search", "Ask local AI"]' in source
    assert "Solve exam problem" not in source
    assert "选择目录" in source
    assert 'st.session_state["data_dir"]' in source
    assert "重建语义索引" in source


def test_should_show_pdf_preview_only_for_pdf_pages() -> None:
    assert should_show_pdf_preview("/tmp/sample.pdf", 1)
    assert should_show_pdf_preview("/tmp/SAMPLE.PDF", 2)
    assert not should_show_pdf_preview("/tmp/sample.pdf", None)
    assert not should_show_pdf_preview("/tmp/sample.docx", 1)
    assert not should_show_pdf_preview("/tmp/sample.pptx", 1)
    assert not should_show_pdf_preview("/tmp/sample.txt", 1)
    assert not should_show_pdf_preview("/tmp/sample.md", 1)
