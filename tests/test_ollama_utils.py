from __future__ import annotations

import json
from io import BytesIO

from openexam.ollama_utils import choose_default_llm_model, ensure_ollama_running, is_ollama_reachable, list_ollama_models, stop_ollama_server


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_list_ollama_models(monkeypatch) -> None:
    def fake_urlopen(*args, **kwargs):
        return FakeResponse(json.dumps({"models": [{"name": "qwen3:8b"}, {"model": "bge-m3"}]}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert is_ollama_reachable("http://127.0.0.1:11434")
    assert list_ollama_models("http://127.0.0.1:11434") == ["bge-m3", "qwen3:8b"]


def test_choose_default_llm_model_skips_embedding() -> None:
    assert choose_default_llm_model(["bge-m3", "qwen3:8b"]) == "qwen3:8b"
    assert choose_default_llm_model(["bge-m3", "llama3.2:latest"]) == "llama3.2:latest"
    assert choose_default_llm_model(["bge-m3"]) is None


def test_auto_start_ollama_uses_mock_subprocess(monkeypatch, tmp_path) -> None:
    calls = {"reachable": 0, "popen": 0}

    def fake_reachable(base_url, timeout=1.0):
        calls["reachable"] += 1
        return calls["reachable"] >= 2

    def fake_models(base_url, timeout=2.0):
        return ["qwen3:8b"]

    class FakePopen:
        def __init__(self, *args, **kwargs):
            calls["popen"] += 1
            self.pid = 12345

    monkeypatch.setattr("openexam.ollama_utils.is_ollama_reachable", fake_reachable)
    monkeypatch.setattr("openexam.ollama_utils.list_ollama_models", fake_models)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr("subprocess.Popen", FakePopen)

    status = ensure_ollama_running("http://127.0.0.1:11434", auto_start=True, log_path=tmp_path / "ollama.log", wait_seconds=1)

    assert status.reachable
    assert status.started
    assert status.models == ["qwen3:8b"]
    assert calls["popen"] == 1
    assert (tmp_path / "ollama.pid").read_text(encoding="utf-8") == "12345"


def test_stop_ollama_without_openexam_pid_file_is_conservative(tmp_path) -> None:
    status = stop_ollama_server(log_path=tmp_path / "ollama.log")

    assert not status.stopped
    assert "did not start" in status.message


def test_stop_ollama_removes_stale_pid_file(monkeypatch, tmp_path) -> None:
    pid_path = tmp_path / "ollama.pid"
    pid_path.write_text("999999", encoding="utf-8")
    monkeypatch.setattr("openexam.ollama_utils._pid_exists", lambda pid: False)

    status = stop_ollama_server(log_path=tmp_path / "ollama.log")

    assert not status.stopped
    assert "stale" in status.message
    assert not pid_path.exists()
