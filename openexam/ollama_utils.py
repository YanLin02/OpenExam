from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from openexam.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class OllamaStatus:
    reachable: bool
    started: bool
    models: list[str]
    message: str
    log_path: Path | None = None


@dataclass(frozen=True)
class OllamaStopStatus:
    stopped: bool
    message: str
    pid: int | None = None


OLLAMA_PATH_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)

OLLAMA_APP_EXECUTABLES = (
    Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    Path.home() / "Applications/Ollama.app/Contents/Resources/ollama",
)


def _tags_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/api/tags"


def _pid_path(log_path: Path | None = None) -> Path:
    if log_path is not None:
        return log_path.parent / "ollama.pid"
    return DEFAULT_CONFIG.index_dir / "ollama.pid"


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_command(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _is_ollama_pid(pid: int) -> bool:
    command = _pid_command(pid)
    if not command:
        return True
    return Path(command).name == "ollama"


def is_ollama_reachable(base_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_tags_url(base_url), timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_ollama_models(base_url: str, timeout: float = 2.0) -> list[str]:
    try:
        with urllib.request.urlopen(_tags_url(base_url), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for item in payload.get("models", []):
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(set(names))


def _extended_path() -> str:
    existing_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    parts = [*existing_parts]
    for path_dir in OLLAMA_PATH_DIRS:
        if path_dir not in parts:
            parts.append(path_dir)
    return os.pathsep.join(parts)


def find_ollama_executable(extra_candidates: Iterable[Path | str] = ()) -> str | None:
    executable = shutil.which("ollama", path=_extended_path())
    if executable:
        return executable

    for candidate in [*extra_candidates, *OLLAMA_APP_EXECUTABLES]:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def start_ollama_server(log_path: Path | None = None) -> tuple[bool, str, Path | None]:
    executable = find_ollama_executable()
    if executable is None:
        return (
            False,
            "Ollama executable not found in PATH or standard macOS install locations. "
            "Install Ollama, or start it manually with `ollama serve`.",
            log_path,
        )

    effective_log_path = log_path or (DEFAULT_CONFIG.index_dir / "ollama.log")
    effective_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = effective_log_path.open("ab")
    env = os.environ.copy()
    env["PATH"] = _extended_path()
    try:
        process = subprocess.Popen(
            [executable, "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        log_file.close()
        return False, f"Failed to start Ollama: {exc}", effective_log_path
    log_file.close()
    _pid_path(effective_log_path).write_text(str(process.pid), encoding="utf-8")
    return True, f"Started Ollama with: ollama serve. Log: {effective_log_path}", effective_log_path


def stop_ollama_server(log_path: Path | None = None, wait_seconds: float = 3.0, poll_interval: float = 0.1) -> OllamaStopStatus:
    pid_path = _pid_path(log_path)
    if not pid_path.exists():
        return OllamaStopStatus(
            False,
            "OpenExam did not start this Ollama process. Stop it manually or use brew services stop ollama.",
        )
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return OllamaStopStatus(False, "OpenExam Ollama pid file was invalid and has been removed.")

    if not _pid_exists(pid):
        pid_path.unlink(missing_ok=True)
        return OllamaStopStatus(False, "OpenExam Ollama pid file was stale and has been removed.", pid)
    if not _is_ollama_pid(pid):
        pid_path.unlink(missing_ok=True)
        return OllamaStopStatus(False, "OpenExam Ollama pid did not match an Ollama process; pid file removed.", pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return OllamaStopStatus(False, "Ollama process was already stopped.", pid)
    except PermissionError as exc:
        return OllamaStopStatus(False, f"Permission denied while stopping Ollama: {exc}", pid)

    deadline = time.perf_counter() + wait_seconds
    while time.perf_counter() < deadline:
        if not _pid_exists(pid):
            pid_path.unlink(missing_ok=True)
            return OllamaStopStatus(True, "Stopped OpenExam-started Ollama.", pid)
        time.sleep(poll_interval)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return OllamaStopStatus(True, "Stopped OpenExam-started Ollama.", pid)
    except PermissionError as exc:
        return OllamaStopStatus(False, f"Permission denied while force-stopping Ollama: {exc}", pid)
    pid_path.unlink(missing_ok=True)
    return OllamaStopStatus(True, "Force-stopped OpenExam-started Ollama.", pid)


def ensure_ollama_running(
    base_url: str,
    auto_start: bool = True,
    log_path: Path | None = None,
    wait_seconds: float = 20.0,
    poll_interval: float = 0.5,
) -> OllamaStatus:
    if is_ollama_reachable(base_url):
        return OllamaStatus(True, False, list_ollama_models(base_url), "Ollama is running.", log_path)
    if not auto_start:
        return OllamaStatus(False, False, [], f"Ollama is not reachable after waiting 0s. Start it with: ollama serve", log_path)

    started, message, effective_log_path = start_ollama_server(log_path=log_path)
    if not started:
        return OllamaStatus(False, False, [], message, effective_log_path)

    deadline = time.perf_counter() + wait_seconds
    while time.perf_counter() < deadline:
        if is_ollama_reachable(base_url):
            return OllamaStatus(True, True, list_ollama_models(base_url), "Ollama started successfully.", effective_log_path)
        time.sleep(poll_interval)
    return OllamaStatus(False, True, [], f"Ollama did not become reachable after waiting {wait_seconds:.0f}s. Check log: {effective_log_path}", effective_log_path)


def choose_default_llm_model(models: list[str], preferred: str = "qwen3:8b", embedding_model: str = "bge-m3") -> str | None:
    if preferred in models:
        return preferred
    embedding_base = embedding_model.split(":", maxsplit=1)[0]
    for model in models:
        model_base = model.split(":", maxsplit=1)[0]
        if model != embedding_model and model_base != embedding_base and "embed" not in model.lower():
            return model
    return None
