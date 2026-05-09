from __future__ import annotations

from openexam.file_utils import file_uri, open_local_file, open_pdf_page_in_chrome, reveal_local_file


def test_file_uri_handles_chinese_spaces_plus_and_page(tmp_path) -> None:
    path = tmp_path / "中文 课件+Transformer (OCR).pdf"
    path.write_text("x", encoding="utf-8")

    uri = file_uri(path, page_number=37)

    assert uri.startswith("file://")
    assert "%E4%B8%AD%E6%96%87%20%E8%AF%BE%E4%BB%B6%2BTransformer%20%28OCR%29.pdf" in uri
    assert uri.endswith("#page=37")


def test_open_local_file_is_non_blocking(monkeypatch, tmp_path) -> None:
    path = tmp_path / "中文 课件.pdf"
    path.write_text("x", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    ok, message = open_local_file(path)

    assert ok
    assert message == "Opening file."
    assert seen["args"] == ["open", str(path)]
    assert seen["kwargs"]["start_new_session"] is True


def test_open_local_file_reports_missing_file(tmp_path) -> None:
    ok, message = open_local_file(tmp_path / "missing.pdf")

    assert not ok
    assert "File does not exist" in message


def test_reveal_local_file_uses_finder_reveal(monkeypatch, tmp_path) -> None:
    path = tmp_path / "中文 课件.pdf"
    path.write_text("x", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    ok, message = reveal_local_file(path)

    assert ok
    assert message == "Revealing file in Finder."
    assert seen["args"] == ["open", "-R", str(path)]


def test_open_pdf_page_in_chrome_is_non_blocking(monkeypatch, tmp_path) -> None:
    path = tmp_path / "中文 课件+Transformer.pdf"
    path.write_text("x", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakePopen:
        def __init__(self, args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    ok, message = open_pdf_page_in_chrome(path, 37)

    assert ok
    assert message == "Opening PDF page in Google Chrome."
    assert seen["args"][0:3] == ["open", "-a", "Google Chrome"]
    assert str(seen["args"][3]).startswith("file://")
    assert str(seen["args"][3]).endswith("#page=37")
    assert "%E4%B8%AD%E6%96%87%20%E8%AF%BE%E4%BB%B6%2BTransformer.pdf" in str(seen["args"][3])
    assert seen["kwargs"]["start_new_session"] is True


def test_open_pdf_page_in_chrome_reports_missing_chrome(monkeypatch, tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_text("x", encoding="utf-8")

    def fake_popen(*args, **kwargs):
        raise OSError("application not found")

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    ok, message = open_pdf_page_in_chrome(path, 1)

    assert not ok
    assert message == "未找到 Google Chrome，请使用页内预览或普通打开文件。"
