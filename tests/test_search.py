from __future__ import annotations

from pathlib import Path

from openexam.config import AppConfig
from openexam.db import connect
from openexam.embeddings import EmbeddingError
from openexam.ingest import ingest_directory
from openexam.search import search_index


def test_ingest_and_search_txt(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "deep learning notes.txt").write_text(
        "CNN uses convolution kernels.\n\nTransformer uses attention and backpropagation.",
        encoding="utf-8",
    )
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)

    stats = ingest_directory(data_dir, config=config, rebuild=True)
    results = search_index("Transformer", top_k=5, config=config, mode="hybrid")

    assert stats.indexed_files == 1
    assert stats.chunks_indexed == 2
    assert results
    assert results[0].file_name == "deep learning notes.txt"
    assert "Transformer" in results[0].snippet

    conn = connect(config.db_path)
    try:
        documents = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        fts = conn.execute("SELECT COUNT(*) AS count FROM chunks_fts").fetchone()["count"]
    finally:
        conn.close()
    assert documents == 1
    assert chunks == 2
    assert fts == 2


def test_fuzzy_search_supplements_fts(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "attention.md").write_text("Transformer attention mechanism", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    results = search_index("attenton", top_k=3, config=config, mode="fuzzy")

    assert results
    assert results[0].match_type == "fuzzy"


def test_chinese_query_uses_substring_fallback(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "Chapter+7-正则化与优化.md").write_text(
        "正则化可以降低过拟合风险。\n\n优化算法包括梯度下降法。",
        encoding="utf-8",
    )
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    results = search_index("正则化 优化", top_k=5, config=config, mode="hybrid")

    assert results
    assert results[0].substring_score > 0
    assert results[0].mode == "hybrid"
    assert "**正则化**" in results[0].snippet or "**优化**" in results[0].snippet


def test_keyword_mode_returns_chinese_exact_phrase(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "gan.md").write_text(
        "生成对抗网络由生成器和判别器组成。\n\n卷积神经网络适合图像任务。",
        encoding="utf-8",
    )
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    results = search_index("生成对抗网络", top_k=3, config=config, mode="keyword")

    assert results
    assert results[0].substring_score == 1.0
    assert "substring" in results[0].match_type
    assert "**生成对抗网络**" in results[0].snippet


def test_fuzzy_mode_handles_chinese_typo(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cnn.md").write_text("卷积神经网络包含卷积层、池化层和全连接层。", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    results = search_index("卷积神经网路", top_k=3, config=config, mode="fuzzy")

    assert results
    assert results[0].file_name == "cnn.md"
    assert results[0].fuzzy_text_score > 0.8


def test_semantic_mode_uses_semantic_scores(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cnn.md").write_text("卷积神经网络依靠局部连接和权值共享。", encoding="utf-8")
    (data_dir / "gan.md").write_text("生成对抗网络包含生成器和判别器。", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    conn = connect(config.db_path)
    try:
        cnn_id = conn.execute("SELECT id FROM chunks WHERE file_name = ?", ("cnn.md",)).fetchone()["id"]
    finally:
        conn.close()

    def fake_semantic_scores(query, config, limit):
        return {cnn_id: 0.93}

    monkeypatch.setattr("openexam.search.semantic_scores", fake_semantic_scores)
    results = search_index("局部连接 权值共享", top_k=3, config=config, mode="semantic")

    assert results
    assert results[0].file_name == "cnn.md"
    assert results[0].semantic_score == 0.93
    assert results[0].match_type == "semantic"


def test_hybrid_falls_back_when_semantic_unavailable(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "transformer.md").write_text("Transformer attention mechanism", encoding="utf-8")
    config = AppConfig(index_dir=tmp_path / ".openexam", chunk_size=800, chunk_overlap=120)
    ingest_directory(data_dir, config=config, rebuild=True)

    def broken_semantic_scores(query, config, limit):
        raise EmbeddingError("Ollama is not reachable")

    monkeypatch.setattr("openexam.search.semantic_scores", broken_semantic_scores)
    results = search_index("Transformer", top_k=3, config=config, mode="hybrid")

    assert results
    assert results[0].file_name == "transformer.md"
    assert results[0].semantic_score == 0.0
