from __future__ import annotations

import json

import numpy as np

from openexam.config import AppConfig
from openexam.embeddings import build_embeddings, embedding_status
from openexam.ingest import ingest_directory


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
