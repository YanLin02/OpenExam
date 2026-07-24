from __future__ import annotations

import atexit
from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


APP_NAME = "OpenExam"
DEFAULT_ADDRESS = "127.0.0.1"
DEFAULT_PORT = 8501
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
SERVER_ARG = "--openexam-streamlit-server"
SERVER_STATE_NAME = "server.json"
LOG_NAME = "OpenExam.log"


def app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME


def log_path(app_dir: Path | None = None) -> Path:
    return (app_dir or app_support_dir()) / LOG_NAME


def ollama_log_path(app_dir: Path | None = None) -> Path:
    return (app_dir or app_support_dir()) / "ollama.log"


def server_state_path(app_dir: Path | None = None) -> Path:
    return (app_dir or app_support_dir()) / SERVER_STATE_NAME


def configure_logging(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path(app_dir).open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(log_file)],
        force=True,
    )


def build_url(port: int, address: str = DEFAULT_ADDRESS) -> str:
    return f"http://{address}:{port}"


def is_port_available(port: int, address: str = DEFAULT_ADDRESS) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return False
        return True


def find_available_port(preferred_port: int = DEFAULT_PORT, address: str = DEFAULT_ADDRESS) -> int:
    if is_port_available(preferred_port, address=address):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((address, 0))
        return int(sock.getsockname()[1])


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_pid(pid: int, wait_seconds: float = 3.0, poll_interval: float = 0.1) -> bool:
    if not pid_exists(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not pid_exists(pid):
            return True
        time.sleep(poll_interval)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def is_url_reachable(url: str, timeout: float = 0.5) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, URLError, ValueError):
        return False


def read_server_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def write_server_state(
    path: Path,
    *,
    port: int,
    url: str,
    ready: bool,
    pid: int | None = None,
    server_pid: int | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "app": APP_NAME,
        "pid": pid or os.getpid(),
        "port": port,
        "url": url,
        "ready": ready,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if server_pid is not None:
        state["server_pid"] = server_pid
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def clear_server_state_if_owned(path: Path, pid: int | None = None) -> None:
    state = read_server_state(path)
    if not state or state.get("pid") != (pid or os.getpid()):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def stop_openexam_owned_ollama(
    app_dir: Path,
    stop_ollama_func: Callable[..., Any] | None = None,
) -> None:
    if stop_ollama_func is None:
        from openexam.ollama_utils import stop_ollama_server as stop_ollama_func

    status = stop_ollama_func(log_path=ollama_log_path(app_dir))
    if getattr(status, "stopped", False):
        logging.info("Stopped OpenExam-started Ollama: %s", getattr(status, "message", ""))
    else:
        logging.info("Ollama shutdown skipped: %s", getattr(status, "message", ""))


def cleanup_owned_runtime(
    app_dir: Path,
    state_path: Path,
    pid: int | None = None,
    server_process: subprocess.Popen[Any] | None = None,
    stop_server_func: Callable[..., Any] | None = None,
    stop_ollama_func: Callable[..., Any] | None = None,
) -> None:
    if server_process is not None:
        stopper = stop_server_func or stop_streamlit_server_process
        stopper(server_process)
    clear_server_state_if_owned(state_path, pid=pid)
    stop_openexam_owned_ollama(app_dir, stop_ollama_func=stop_ollama_func)


def install_shutdown_handlers(
    app_dir: Path,
    state_path: Path,
    pid: int | None = None,
    server_process: subprocess.Popen[Any] | None = None,
    stop_server_func: Callable[..., Any] | None = None,
    stop_ollama_func: Callable[..., Any] | None = None,
) -> None:
    owner_pid = pid or os.getpid()

    def handle_shutdown(signum: int, _frame: Any) -> None:
        logging.info("Received shutdown signal %s", signum)
        cleanup_owned_runtime(
            app_dir,
            state_path,
            pid=owner_pid,
            server_process=server_process,
            stop_server_func=stop_server_func,
            stop_ollama_func=stop_ollama_func,
        )
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_shutdown)


def streamlit_app_path() -> Path:
    candidates = [Path(__file__).resolve().with_name("app.py")]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "openexam" / "app.py")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def streamlit_argv(port: int, address: str = DEFAULT_ADDRESS) -> list[str]:
    return [
        "streamlit",
        "run",
        str(streamlit_app_path()),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]


def run_streamlit_app(port: int, address: str = DEFAULT_ADDRESS, cli_module: Any | None = None) -> int:
    if cli_module is None:
        from streamlit.web import cli as cli_module

    old_argv = sys.argv[:]
    sys.argv = streamlit_argv(port, address=address)
    try:
        cli_module.main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 0 if exc.code is None else 1
    finally:
        sys.argv = old_argv
    return 0


def wait_for_url(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    interval_seconds: float = 0.5,
    url_checker: Callable[[str], bool] = is_url_reachable,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if url_checker(url):
            return True
        time.sleep(interval_seconds)
    return False


def start_parent_monitor(app_dir: Path, parent_pid: int, poll_interval: float = 0.5) -> None:
    state_path = server_state_path(app_dir)

    def target() -> None:
        while True:
            if not pid_exists(parent_pid):
                logging.info("OpenExam parent process %s exited; stopping Streamlit server.", parent_pid)
                clear_server_state_if_owned(state_path, pid=parent_pid)
                os._exit(0)
            time.sleep(poll_interval)

    threading.Thread(target=target, name="openexam-parent-monitor", daemon=True).start()


def start_streamlit_server_process(
    app_dir: Path,
    port: int,
    address: str = DEFAULT_ADDRESS,
    parent_pid: int | None = None,
) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["OPENEXAM_INDEX_DIR"] = str(app_dir)
    env["PYTHONUNBUFFERED"] = "1"
    app_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path(app_dir).open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, SERVER_ARG, str(app_dir), address, str(port), str(parent_pid or os.getpid())],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()


def stop_streamlit_server_process(
    process: subprocess.Popen[Any],
    wait_seconds: float = 5.0,
) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=wait_seconds)


def open_app_window(url: str) -> None:
    import webview

    webview.create_window(
        APP_NAME,
        url,
        width=1280,
        height=860,
        min_size=(960, 640),
        text_select=True,
    )
    webview.start(debug=False)


def launch_app(
    *,
    app_dir: Path | None = None,
    preferred_port: int = DEFAULT_PORT,
    address: str = DEFAULT_ADDRESS,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    open_window_func: Callable[[str], None] = open_app_window,
    start_server_func: Callable[..., subprocess.Popen[Any]] = start_streamlit_server_process,
    stop_server_func: Callable[..., Any] = stop_streamlit_server_process,
    url_checker: Callable[[str], bool] = is_url_reachable,
) -> int:
    app_dir = app_dir or app_support_dir()
    app_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENEXAM_INDEX_DIR"] = str(app_dir)

    state_path = server_state_path(app_dir)
    state = read_server_state(state_path)
    existing_url = state.get("url") if state else None
    if isinstance(existing_url, str) and url_checker(existing_url):
        existing_pid = state.get("pid")
        if isinstance(existing_pid, int) and pid_exists(existing_pid):
            logging.info("%s is already running at %s", APP_NAME, existing_url)
            return 0
        stale_server_pid = state.get("server_pid")
        if isinstance(stale_server_pid, int):
            logging.info("Stopping stale OpenExam server process %s", stale_server_pid)
            terminate_pid(stale_server_pid)
        clear_server_state_if_owned(state_path, pid=existing_pid if isinstance(existing_pid, int) else None)

    port = find_available_port(preferred_port, address=address)
    url = build_url(port, address=address)
    owner_pid = os.getpid()
    server_process = start_server_func(app_dir, port, address, owner_pid)
    write_server_state(state_path, port=port, url=url, ready=False, pid=owner_pid, server_pid=server_process.pid)
    atexit.register(cleanup_owned_runtime, app_dir, state_path, owner_pid, server_process, stop_server_func)
    install_shutdown_handlers(app_dir, state_path, owner_pid, server_process, stop_server_func)

    try:
        if not wait_for_url(url, timeout_seconds=startup_timeout_seconds, url_checker=url_checker):
            logging.error("Timed out waiting for %s to become reachable", url)
            return 1
        write_server_state(state_path, port=port, url=url, ready=True, pid=owner_pid, server_pid=server_process.pid)
        open_window_func(url)
        return 0
    finally:
        cleanup_owned_runtime(app_dir, state_path, owner_pid, server_process, stop_server_func)


def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == SERVER_ARG:
        app_dir = Path(sys.argv[2])
        address = sys.argv[3]
        port = int(sys.argv[4])
        parent_pid = int(sys.argv[5])
        configure_logging(app_dir)
        os.environ["OPENEXAM_INDEX_DIR"] = str(app_dir)
        start_parent_monitor(app_dir, parent_pid)
        logging.info("Starting %s Streamlit server on %s:%s", APP_NAME, address, port)
        return run_streamlit_app(port=port, address=address)

    app_dir = app_support_dir()
    configure_logging(app_dir)
    logging.info("Starting %s", APP_NAME)
    return launch_app(app_dir=app_dir)


if __name__ == "__main__":
    raise SystemExit(main())
