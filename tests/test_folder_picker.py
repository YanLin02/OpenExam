from __future__ import annotations

import subprocess
from types import SimpleNamespace

from openexam.folder_picker import pick_folder, pick_folder_macos


def test_pick_folder_macos_success(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        assert args[0][:2] == ["osascript", "-e"]
        assert kwargs["shell"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 60
        return SimpleNamespace(returncode=0, stdout="/tmp/data\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path == "/tmp/data"
    assert result.cancelled is False
    assert result.error is None


def test_pick_folder_macos_cancelled(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="User canceled.")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path is None
    assert result.cancelled is True
    assert result.error is None


def test_pick_folder_macos_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=60)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path is None
    assert result.cancelled is False
    assert result.error


def test_pick_folder_non_macos(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")

    result = pick_folder()

    assert result.selected_path is None
    assert result.cancelled is False
    assert "Please enter the path manually" in (result.error or "")
