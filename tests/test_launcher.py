from __future__ import annotations

from pathlib import Path

from openexam import launcher
from openexam.__main__ import build_parser


def test_launch_streamlit_uses_python_module_command(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda name: object() if name == "streamlit" else None)
    monkeypatch.setattr(launcher.subprocess, "call", fake_call)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")

    assert launcher.launch_streamlit(address="127.0.0.1", port=8501) == 0

    assert calls == [
        [
            "/usr/bin/python3",
            "-m",
            "streamlit",
            "run",
            str(Path(launcher.__file__).resolve().with_name("app.py")),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
        ]
    ]
    assert Path(calls[0][4]).is_absolute()
    assert calls[0][4].endswith("openexam/app.py")


def test_openexam_parser_has_ui_subcommand() -> None:
    args = build_parser().parse_args(["ui", "--address", "127.0.0.1", "--port", "8501", "--headless"])

    assert args.command == "ui"
    assert args.address == "127.0.0.1"
    assert args.port == 8501
    assert args.headless is True
