from __future__ import annotations

import pytest

from openexam.pdf_preview import PdfPreviewError, render_pdf_page


def make_pdf(path):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "first page")
    page = document.new_page()
    page.insert_text((72, 72), "second page")
    document.save(path)
    document.close()


def test_render_pdf_page_returns_png_bytes(tmp_path) -> None:
    path = tmp_path / "中文 课件+Transformer (OCR).pdf"
    make_pdf(path)

    data = render_pdf_page(str(path), 1, zoom=1.0)

    assert data.startswith(b"\x89PNG")


def test_render_pdf_page_uses_one_based_page_number(tmp_path, monkeypatch) -> None:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "sample.pdf"
    make_pdf(path)
    loaded_pages: list[int] = []
    original_open = fitz.open

    class WrappedDocument:
        def __init__(self, document):
            self._document = document

        @property
        def page_count(self):
            return self._document.page_count

        def load_page(self, index):
            loaded_pages.append(index)
            return self._document.load_page(index)

        def close(self):
            self._document.close()

    def fake_open(*args, **kwargs):
        return WrappedDocument(original_open(*args, **kwargs))

    monkeypatch.setattr("openexam.pdf_preview.fitz.open", fake_open)

    render_pdf_page(str(path), 2, zoom=1.0)

    assert loaded_pages == [1]


def test_render_pdf_page_out_of_range_has_clear_error(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    make_pdf(path)

    with pytest.raises(PdfPreviewError, match="Page 3 is out of range. PDF has 2 pages."):
        render_pdf_page(str(path), 3)


def test_render_pdf_page_missing_file_has_clear_error(tmp_path) -> None:
    with pytest.raises(PdfPreviewError, match="PDF file does not exist"):
        render_pdf_page(str(tmp_path / "missing.pdf"), 1)


def test_render_pdf_page_bad_pdf_wraps_mupdf_error(tmp_path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(PdfPreviewError, match="PDF page preview failed. Use Open File"):
        render_pdf_page(str(path), 1)


def test_render_pdf_page_runtime_error_is_wrapped(tmp_path, monkeypatch) -> None:
    path = tmp_path / "sample.pdf"
    make_pdf(path)

    class BrokenPage:
        def get_pixmap(self, *args, **kwargs):
            raise RuntimeError("MuPDF error: format error: No common ancestor in structure tree")

    class BrokenDocument:
        page_count = 1

        def load_page(self, index):
            return BrokenPage()

        def close(self):
            pass

    monkeypatch.setattr("openexam.pdf_preview.fitz.open", lambda *args, **kwargs: BrokenDocument())

    with pytest.raises(PdfPreviewError) as exc_info:
        render_pdf_page(str(path), 1)

    message = str(exc_info.value)
    assert "PDF page preview failed. Use Open File" in message
    assert "MuPDF error" not in message
