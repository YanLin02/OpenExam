from __future__ import annotations

import json

import numpy as np

from openexam.__main__ import build_parser, cmd_ingest
from openexam.config import AppConfig
from openexam.embeddings import EmbedResult, EmbedStats, build_embeddings, embed_if_needed, embedding_status
from openexam.ingest import ingest_directory
from openexam.ollama_utils import OllamaStatus


def test_build_embeddings_writes_numpy_and_metadata(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text("Transformer attention.\n\nCNN convolution.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    def fake_embed(texts, config):
        return np.asarray([[float(index + 1), 1.0] for index, _ in enumerate(texts)], dtype=np.float32)

    monkeypatch.setattr("openexam.embeddings.ollama_embed", fake_embed)
    stats = build_embeddings(config)

    assert stats.chunk_count == 2
    assert stats.vector_dim == 2
    assert config.embeddings_npy_path.name == "embeddings_bge-m3.npy"
    assert config.embeddings_json_path.name == "embeddings_bge-m3.json"
    assert config.embeddings_npy_path.exists()
    assert config.embeddings_json_path.exists()

    vectors = np.load(config.embeddings_npy_path)
    assert vectors.shape == (2, 2)
    metadata = json.loads(config.embeddings_json_path.read_text(encoding="utf-8"))
    first = metadata["chunks"][0]
    assert first["chunk_id"]
    assert first["document_id"] == 1
    assert first["embedding_model"] == "bge-m3"
    assert first["text_hash"]
    assert first["created_at"]

    status = embedding_status(config)
    assert status.valid
    assert status.vector_count == 2


def test_embedding_status_detects_changed_chunks(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    notes = data_dir / "notes.md"
    notes.write_text("Original Transformer note.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    def fake_embed(texts, config):
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("openexam.embeddings.ollama_embed", fake_embed)
    build_embeddings(config)

    notes.write_text("Changed Transformer note.", encoding="utf-8")
    ingest_directory(data_dir, config=config, rebuild=True)
    status = embedding_status(config)

    assert status.exists
    assert status.stale
    assert not status.valid
    assert "Run embed again" in status.message


def test_embed_if_needed_skips_when_ready(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text("Transformer attention.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    def fake_embed(texts, config):
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("openexam.embeddings.ollama_embed", fake_embed)
    build_embeddings(config)

    def fail_build(config):
        raise AssertionError("build_embeddings should not be called")

    monkeypatch.setattr("openexam.embeddings.build_embeddings", fail_build)
    result = embed_if_needed(config, auto_start_ollama=False)

    assert result.status == "skipped"
    assert result.chunks_embedded == 0
    assert result.message == "already_ready"


def test_embed_if_needed_builds_when_missing(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text("Transformer attention.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)
    called: dict[str, bool] = {}

    monkeypatch.setattr(
        "openexam.embeddings.ensure_ollama_running",
        lambda *args, **kwargs: OllamaStatus(True, False, ["bge-m3"], "Ollama is running."),
    )

    def fake_build(config):
        called["build"] = True
        return EmbedStats(
            model=config.embedding_model,
            chunk_count=1,
            vector_dim=3,
            npy_path=config.embeddings_npy_path,
            json_path=config.embeddings_json_path,
        )

    monkeypatch.setattr("openexam.embeddings.build_embeddings", fake_build)
    result = embed_if_needed(config)

    assert called["build"] is True
    assert result.status == "ready"
    assert result.chunks_embedded == 1


def test_embed_if_needed_handles_ollama_unreachable(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text("Transformer attention.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    monkeypatch.setattr(
        "openexam.embeddings.ensure_ollama_running",
        lambda *args, **kwargs: OllamaStatus(False, False, [], "Ollama is not reachable. Start it with: ollama serve"),
    )
    result = embed_if_needed(config)

    assert result.status == "failed"
    assert "Ollama is not reachable" in result.message


def test_ingest_parser_supports_embed_flags() -> None:
    parser = build_parser()

    embed_args = parser.parse_args(["ingest", "/tmp/data", "--embed"])
    no_embed_args = parser.parse_args(["ingest", "/tmp/data", "--no-embed"])

    assert embed_args.embed is True
    assert no_embed_args.embed is False


def test_ingest_embed_calls_embed_if_needed(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.md").write_text("Transformer attention.", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    called: dict[str, object] = {}

    monkeypatch.setattr("openexam.__main__.DEFAULT_CONFIG", config)

    def fake_embed_if_needed(config, auto_start_ollama=True, force=False):
        called["config"] = config
        called["auto_start_ollama"] = auto_start_ollama
        return EmbedResult(
            status="ready",
            chunks_embedded=1,
            model=config.embedding_model,
            elapsed_time_ms=1.0,
            message="Semantic index is ready.",
        )

    monkeypatch.setattr("openexam.__main__.embed_if_needed", fake_embed_if_needed)
    args = build_parser().parse_args(["ingest", str(data_dir), "--embed", "--no-auto-start-ollama"])
    exit_code = cmd_ingest(args)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["config"] == config
    assert called["auto_start_ollama"] is False
    assert "Embed status: ready" in output
