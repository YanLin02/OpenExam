from __future__ import annotations

from pathlib import Path

import fitz


class PdfPreviewError(RuntimeError):
    pass


def _fitz_preview_exceptions() -> tuple[type[BaseException], ...]:
    names = ("FileDataError", "EmptyFileError", "FileNotFoundError")
    exceptions: list[type[BaseException]] = [RuntimeError, ValueError]
    for name in names:
        exc_type = getattr(fitz, name, None)
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            exceptions.append(exc_type)
    return tuple(exceptions)


def _preview_error(exc: BaseException) -> PdfPreviewError:
    summary = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    summary = summary.replace("MuPDF error: ", "")
    return PdfPreviewError(f"PDF page preview failed. Use Open File to view the document. {summary}")


def render_pdf_page(path: str, page_number: int, zoom: float = 1.5) -> bytes:
    target = Path(path).expanduser()
    if not target.exists():
        raise PdfPreviewError(f"PDF file does not exist: {target}")
    if page_number < 1:
        raise PdfPreviewError(f"Page number must be 1 or greater: {page_number}")

    try:
        document = fitz.open(target)
    except _fitz_preview_exceptions() as exc:
        raise _preview_error(exc) from exc
    try:
        page_index = page_number - 1
        if page_index >= document.page_count:
            raise PdfPreviewError(f"Page {page_number} is out of range. PDF has {document.page_count} pages.")
        try:
            page = document.load_page(page_index)
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            return pixmap.tobytes("png")
        except _fitz_preview_exceptions() as exc:
            raise _preview_error(exc) from exc
    finally:
        document.close()
