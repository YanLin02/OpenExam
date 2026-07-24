from __future__ import annotations

from pathlib import Path

from openexam.config import build_default_config


def test_build_default_config_uses_project_index_dir_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENEXAM_INDEX_DIR", raising=False)

    config = build_default_config()

    assert config.index_dir == Path(".openexam")


def test_build_default_config_allows_index_dir_env_override(monkeypatch, tmp_path) -> None:
    index_dir = tmp_path / "OpenExam Data"
    monkeypatch.setenv("OPENEXAM_INDEX_DIR", str(index_dir))

    config = build_default_config()

    assert config.index_dir == index_dir
