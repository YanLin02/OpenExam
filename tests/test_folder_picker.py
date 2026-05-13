from __future__ import annotations

import subprocess

from openexam.folder_picker import pick_folder, pick_folder_macos


def test_pick_folder_macos_returns_selected_path(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="/Users/example/Documents\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path == "/Users/example/Documents"
    assert not result.cancelled
    assert result.error is None


def test_pick_folder_macos_detects_cancel(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="User canceled. (-128)")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path is None
    assert result.cancelled
    assert result.error is None


def test_pick_folder_macos_reports_subprocess_error(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="osascript failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = pick_folder_macos()

    assert result.selected_path is None
    assert not result.cancelled
    assert result.error == "osascript failed"


def test_pick_folder_non_macos_prompts_manual_input(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")

    result = pick_folder()

    assert result.selected_path is None
    assert not result.cancelled
    assert result.error is not None
    assert "manually" in result.error
