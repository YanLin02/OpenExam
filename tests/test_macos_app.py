from __future__ import annotations

import json
from pathlib import Path
import signal
import socket
from types import SimpleNamespace

from openexam import macos_app


def test_app_support_dir_uses_macos_application_support(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert macos_app.app_support_dir() == tmp_path / "Library" / "Application Support" / "OpenExam"


def test_find_available_port_skips_busy_preferred_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((macos_app.DEFAULT_ADDRESS, 0))
        busy_port = int(sock.getsockname()[1])

        selected_port = macos_app.find_available_port(busy_port)

    assert selected_port != busy_port


def test_write_and_read_server_state(tmp_path) -> None:
    state_path = tmp_path / "server.json"

    state = macos_app.write_server_state(
        state_path,
        port=8501,
        url="http://127.0.0.1:8501",
        ready=False,
        pid=123,
        server_pid=456,
    )

    loaded = macos_app.read_server_state(state_path)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded == raw
    assert state["pid"] == 123
    assert state["port"] == 8501
    assert state["ready"] is False
    assert state["url"] == "http://127.0.0.1:8501"
    assert state["server_pid"] == 456


def test_run_streamlit_app_uses_expected_streamlit_args(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_main() -> None:
        calls.append(macos_app.sys.argv[:])

    monkeypatch.setattr(macos_app, "streamlit_app_path", lambda: Path("/tmp/openexam/app.py"))
    cli_module = SimpleNamespace(main=fake_main)

    assert macos_app.run_streamlit_app(4567, cli_module=cli_module) == 0

    assert calls == [
        [
            "streamlit",
            "run",
            "/tmp/openexam/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "4567",
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
            "--global.developmentMode",
            "false",
        ]
    ]


def test_launch_app_reuses_existing_reachable_server(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "server.json"
    state_path.write_text(
        json.dumps({"pid": 123, "port": 8501, "url": "http://127.0.0.1:8501"}),
        encoding="utf-8",
    )
    opened_urls: list[str] = []

    def fail_run_streamlit(port: int, address: str) -> int:
        raise AssertionError("launch_app should reuse the existing service")

    monkeypatch.setattr(macos_app, "pid_exists", lambda pid: pid == 123)

    result = macos_app.launch_app(
        app_dir=tmp_path,
        open_window_func=opened_urls.append,
        start_server_func=fail_run_streamlit,
        url_checker=lambda url: True,
    )

    assert result == 0
    assert opened_urls == []


def test_launch_app_starts_embedded_window_and_cleans_up(monkeypatch, tmp_path) -> None:
    class FakeProcess:
        pid = 456

        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = FakeProcess()
    opened_urls: list[str] = []
    started: list[tuple[Path, int, str, int]] = []
    stopped: list[FakeProcess] = []

    monkeypatch.setattr(macos_app.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(macos_app, "install_shutdown_handlers", lambda *args, **kwargs: None)

    result = macos_app.launch_app(
        app_dir=tmp_path,
        start_server_func=lambda app_dir, port, address, parent_pid: started.append((app_dir, port, address, parent_pid))
        or process,
        stop_server_func=stopped.append,
        open_window_func=opened_urls.append,
        url_checker=lambda url: True,
    )

    assert result == 0
    assert len(started) == 1
    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("http://127.0.0.1:")
    assert stopped == [process]
    assert not (tmp_path / "server.json").exists()


def test_cleanup_owned_runtime_clears_state_and_stops_owned_ollama(tmp_path) -> None:
    state_path = tmp_path / "server.json"
    macos_app.write_server_state(
        state_path,
        port=8501,
        url="http://127.0.0.1:8501",
        ready=True,
        pid=123,
    )
    calls: list[Path] = []
    stopped_processes: list[object] = []

    def fake_stop_ollama(*, log_path: Path):
        calls.append(log_path)
        return SimpleNamespace(stopped=True, message="stopped")

    process = object()
    macos_app.cleanup_owned_runtime(
        tmp_path,
        state_path,
        pid=123,
        server_process=process,
        stop_server_func=stopped_processes.append,
        stop_ollama_func=fake_stop_ollama,
    )

    assert not state_path.exists()
    assert calls == [tmp_path / "ollama.log"]
    assert stopped_processes == [process]


def test_install_shutdown_handlers_registers_interrupt_and_terminate(monkeypatch, tmp_path) -> None:
    registered: dict[signal.Signals, object] = {}

    def fake_signal(signum, handler):
        registered[signum] = handler

    monkeypatch.setattr(macos_app.signal, "signal", fake_signal)

    macos_app.install_shutdown_handlers(tmp_path, tmp_path / "server.json", pid=123)

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
