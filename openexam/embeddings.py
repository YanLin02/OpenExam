from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.ollama_utils import ensure_ollama_running
from openexam.text_utils import sha256_text


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingStatus:
    exists: bool
    valid: bool
    stale: bool
    model: str
    chunk_count: int
    vector_count: int
    message: str
    npy_path: Path
    json_path: Path


@dataclass(frozen=True)
class EmbedStats:
    model: str
    chunk_count: int
    vector_dim: int
    npy_path: Path
    json_path: Path


@dataclass(frozen=True)
class EmbedResult:
    status: str
    chunks_embedded: int
    elapsed_time_ms: float
    message: str
    stats: EmbedStats | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 or "not found" in body.lower() or "pull" in body.lower():
            raise EmbeddingError("Ollama model is not available. While online, run: ollama pull bge-m3") from exc
        raise EmbeddingError(f"Ollama embedding request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError("Ollama is not reachable at http://127.0.0.1:11434. Start it with: ollama serve") from exc


def ollama_embed(texts: list[str], config: AppConfig = DEFAULT_CONFIG) -> np.ndarray:
    if config.embedding_provider != "ollama":
        raise EmbeddingError(f"Unsupported embedding provider: {config.embedding_provider}")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    url = config.ollama_base_url.rstrip("/") + "/api/embed"
    response = _post_json(url, {"model": config.embedding_model, "input": texts})
    embeddings = response.get("embeddings")
    if embeddings is None:
        legacy_url = config.ollama_base_url.rstrip("/") + "/api/embeddings"
        vectors = []
        for text in texts:
            legacy_response = _post_json(legacy_url, {"model": config.embedding_model, "prompt": text})
            vector = legacy_response.get("embedding")
            if vector is None:
                raise EmbeddingError("Ollama response did not include embeddings.")
            vectors.append(vector)
        embeddings = vectors
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise EmbeddingError("Ollama returned embeddings with an unexpected shape.")
    return array


def _load_chunks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id AS chunk_db_id, document_id, chunk_id, text, text_norm
            FROM chunks
            ORDER BY id
            """
        )
    )


def _count_chunks(config: AppConfig) -> int:
    if not config.db_path.exists():
        return 0
    conn = sqlite3.connect(config.db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        conn.close()


def build_embeddings(config: AppConfig = DEFAULT_CONFIG) -> EmbedStats:
    config.index_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_chunks(conn)
    finally:
        conn.close()
    if not rows:
        raise EmbeddingError("No chunks found. Run ingest before embed.")

    vectors: list[np.ndarray] = []
    for start in range(0, len(rows), config.embedding_batch_size):
        batch = rows[start : start + config.embedding_batch_size]
        vectors.append(ollama_embed([row["text"] for row in batch], config=config))
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    metadata = {
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "ollama_base_url": config.ollama_base_url,
        "created_at": utc_now(),
        "vector_dim": int(matrix.shape[1]),
        "chunks": [
            {
                "chunk_db_id": int(row["chunk_db_id"]),
                "chunk_id": row["chunk_id"],
                "document_id": int(row["document_id"]),
                "embedding_model": config.embedding_model,
                "text_hash": sha256_text(row["text"]),
                "created_at": utc_now(),
            }
            for row in rows
        ],
    }

    np.save(config.embeddings_npy_path, matrix)
    config.embeddings_json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return EmbedStats(
        model=config.embedding_model,
        chunk_count=len(rows),
        vector_dim=int(matrix.shape[1]),
        npy_path=config.embeddings_npy_path,
        json_path=config.embeddings_json_path,
    )


def embed_if_needed(
    config: AppConfig = DEFAULT_CONFIG,
    auto_start_ollama: bool = True,
    force: bool = False,
) -> EmbedResult:
    started_at = time.perf_counter()
    status = embedding_status(config)
    if status.valid and not force:
        return EmbedResult(
            status="skipped",
            chunks_embedded=0,
            elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
            message="Semantic index is already ready.",
        )
    if not config.db_path.exists():
        return EmbedResult(
            status="failed",
            chunks_embedded=0,
            elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
            message="SQLite index not found. Run ingest before embed.",
        )
    chunk_count = _count_chunks(config)
    if chunk_count == 0:
        return EmbedResult(
            status="failed",
            chunks_embedded=0,
            elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
            message="No chunks found. Run ingest before embed.",
        )

    ollama_status = ensure_ollama_running(
        config.ollama_base_url,
        auto_start=auto_start_ollama,
        log_path=config.index_dir / "ollama.log",
    )
    if not ollama_status.reachable:
        return EmbedResult(
            status="failed",
            chunks_embedded=0,
            elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
            message=f"Embedding failed: {ollama_status.message}",
        )

    try:
        stats = build_embeddings(config)
    except EmbeddingError as exc:
        return EmbedResult(
            status="failed",
            chunks_embedded=0,
            elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
            message=f"Embedding failed: {exc}",
        )
    return EmbedResult(
        status="ready",
        chunks_embedded=stats.chunk_count,
        elapsed_time_ms=(time.perf_counter() - started_at) * 1000,
        message="Semantic index is ready.",
        stats=stats,
    )


def load_embedding_metadata(config: AppConfig = DEFAULT_CONFIG) -> dict[str, Any] | None:
    if not config.embeddings_json_path.exists():
        return None
    try:
        return json.loads(config.embeddings_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def embedding_status(config: AppConfig = DEFAULT_CONFIG) -> EmbeddingStatus:
    metadata = load_embedding_metadata(config)
    npy_exists = config.embeddings_npy_path.exists()
    if metadata is None or not npy_exists:
        return EmbeddingStatus(
            exists=False,
            valid=False,
            stale=False,
            model=config.embedding_model,
            chunk_count=0,
            vector_count=0,
            message="Semantic index not found. Run: python3 -m openexam embed",
            npy_path=config.embeddings_npy_path,
            json_path=config.embeddings_json_path,
        )
    if not config.db_path.exists():
        return EmbeddingStatus(True, False, True, config.embedding_model, 0, 0, "SQLite index not found. Run ingest and embed again.", config.embeddings_npy_path, config.embeddings_json_path)

    chunks = metadata.get("chunks", [])
    model = metadata.get("embedding_model", "")
    if model != config.embedding_model:
        return EmbeddingStatus(True, False, True, model or config.embedding_model, 0, len(chunks), "Embedding model changed. Run embed again.", config.embeddings_npy_path, config.embeddings_json_path)

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_chunks(conn)
    finally:
        conn.close()
    if len(rows) != len(chunks):
        return EmbeddingStatus(True, False, True, model, len(rows), len(chunks), "Chunk count changed. Run embed again.", config.embeddings_npy_path, config.embeddings_json_path)

    for row, item in zip(rows, chunks, strict=True):
        if int(row["chunk_db_id"]) != int(item.get("chunk_db_id", -1)) or row["chunk_id"] != item.get("chunk_id") or sha256_text(row["text"]) != item.get("text_hash"):
            return EmbeddingStatus(True, False, True, model, len(rows), len(chunks), "Chunk content changed. Run embed again.", config.embeddings_npy_path, config.embeddings_json_path)

    return EmbeddingStatus(True, True, False, model, len(rows), len(chunks), "Semantic index is ready.", config.embeddings_npy_path, config.embeddings_json_path)


def semantic_scores(query: str, config: AppConfig = DEFAULT_CONFIG, limit: int = 100) -> dict[int, float]:
    status = embedding_status(config)
    if not status.valid:
        return {}
    metadata = load_embedding_metadata(config)
    if metadata is None:
        return {}
    matrix = np.load(config.embeddings_npy_path)
    query_vector = ollama_embed([query], config=config)[0]
    query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    scores = matrix @ query_vector.astype(np.float32)
    if scores.size == 0:
        return {}
    top_indices = np.argsort(scores)[::-1][:limit]
    chunks = metadata.get("chunks", [])
    return {int(chunks[index]["chunk_db_id"]): float(max(scores[index], 0.0)) for index in top_indices}
