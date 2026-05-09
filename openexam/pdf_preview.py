from __future__ import annotations

from pathlib import Path

import fitz


class PdfPreviewError(RuntimeError):
    pass


def render_pdf_page(path: str, page_number: int, zoom: float = 1.5) -> bytes:
    target = Path(path).expanduser()
    if not target.exists():
        raise PdfPreviewError(f"PDF file does not exist: {target}")
    if page_number < 1:
        raise PdfPreviewError(f"Page number must be 1 or greater: {page_number}")

    document = fitz.open(target)
    try:
        page_index = page_number - 1
        if page_index >= document.page_count:
            raise PdfPreviewError(f"Page {page_number} is out of range. PDF has {document.page_count} pages.")
        page = document.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()
