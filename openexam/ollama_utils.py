from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
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


def _tags_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/api/tags"


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


def start_ollama_server(log_path: Path | None = None) -> tuple[bool, str, Path | None]:
    executable = shutil.which("ollama")
    if executable is None:
        return False, "Ollama executable not found in PATH. Install Ollama or start it manually.", log_path

    effective_log_path = log_path or (DEFAULT_CONFIG.index_dir / "ollama.log")
    effective_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = effective_log_path.open("ab")
    try:
        subprocess.Popen(
            [executable, "serve"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_file.close()
        return False, f"Failed to start Ollama: {exc}", effective_log_path
    log_file.close()
    return True, f"Started Ollama with: ollama serve. Log: {effective_log_path}", effective_log_path


def ensure_ollama_running(
    base_url: str,
    auto_start: bool = True,
    log_path: Path | None = None,
    wait_seconds: float = 8.0,
    poll_interval: float = 0.5,
) -> OllamaStatus:
    if is_ollama_reachable(base_url):
        return OllamaStatus(True, False, list_ollama_models(base_url), "Ollama is running.", log_path)
    if not auto_start:
        return OllamaStatus(False, False, [], "Ollama is not reachable. Start it with: ollama serve", log_path)

    started, message, effective_log_path = start_ollama_server(log_path=log_path)
    if not started:
        return OllamaStatus(False, False, [], message, effective_log_path)

    deadline = time.perf_counter() + wait_seconds
    while time.perf_counter() < deadline:
        if is_ollama_reachable(base_url):
            return OllamaStatus(True, True, list_ollama_models(base_url), "Ollama started successfully.", effective_log_path)
        time.sleep(poll_interval)
    return OllamaStatus(False, True, [], f"Ollama did not become reachable after {wait_seconds:.0f}s. Check log: {effective_log_path}", effective_log_path)


def choose_default_llm_model(models: list[str], preferred: str = "qwen3:8b", embedding_model: str = "bge-m3") -> str | None:
    if preferred in models:
        return preferred
    embedding_base = embedding_model.split(":", maxsplit=1)[0]
    for model in models:
        model_base = model.split(":", maxsplit=1)[0]
        if model != embedding_model and model_base != embedding_base and "embed" not in model.lower():
            return model
    return None
