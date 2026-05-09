from __future__ import annotations

from openexam.file_utils import file_uri


def test_file_uri_handles_chinese_spaces_plus_and_page(tmp_path) -> None:
    path = tmp_path / "中文 课件+Transformer (OCR).pdf"
    path.write_text("x", encoding="utf-8")

    uri = file_uri(path, page_number=37)

    assert uri.startswith("file://")
    assert "%E4%B8%AD%E6%96%87%20%E8%AF%BE%E4%BB%B6%2BTransformer%20%28OCR%29.pdf" in uri
    assert uri.endswith("#page=37")
