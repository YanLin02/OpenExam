from __future__ import annotations

from openexam.file_utils import file_uri, open_local_file, reveal_local_file


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
