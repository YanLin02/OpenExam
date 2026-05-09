from __future__ import annotations

import hashlib
from pathlib import Path

from openexam.chunking import chunk_sections
from openexam.config import AppConfig, DEFAULT_CONFIG, SUPPORTED_EXTENSIONS
from openexam.db import clear_index, connect, delete_document_chunks, get_document_by_path, insert_chunks, upsert_document
from openexam.extractors import extract_file
from openexam.extractors.base import ExtractionError
from openexam.models import IngestStats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def ingest_directory(root: Path, config: AppConfig = DEFAULT_CONFIG, rebuild: bool = False) -> IngestStats:
    root = root.expanduser().resolve()
    stats = IngestStats()
    conn = connect(config.db_path)
    try:
        if rebuild:
            clear_index(conn)

        for path in scan_files(root):
            stats.scanned_files += 1
            try:
                file_stat = path.stat()
                file_hash = sha256_file(path)
                existing = get_document_by_path(conn, path)
                if (
                    existing is not None
                    and not rebuild
                    and existing["sha256"] == file_hash
                    and existing["status"] == "indexed"
                ):
                    stats.skipped_files += 1
                    continue

                document_id = upsert_document(
                    conn,
                    path,
                    size_bytes=file_stat.st_size,
                    mtime=file_stat.st_mtime,
                    sha256=file_hash,
                    status="indexing",
                    error=None,
                )
                delete_document_chunks(conn, document_id)

                sections = extract_file(path)
                chunks = chunk_sections(sections, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
                if not chunks:
                    message = "Warning: no extractable text found. Scanned PDF or empty document may require OCR, which is not implemented."
                    upsert_document(
                        conn,
                        path,
                        size_bytes=file_stat.st_size,
                        mtime=file_stat.st_mtime,
                        sha256=file_hash,
                        status="failed",
                        error=message,
                    )
                    stats.failed_files += 1
                    stats.errors.append((str(path), message))
                    conn.commit()
                    continue

                inserted = insert_chunks(conn, document_id, chunks)
                upsert_document(
                    conn,
                    path,
                    size_bytes=file_stat.st_size,
                    mtime=file_stat.st_mtime,
                    sha256=file_hash,
                    status="indexed",
                    error=None,
                )
                conn.commit()
                stats.indexed_files += 1
                stats.chunks_indexed += inserted
            except (ExtractionError, OSError, ValueError) as exc:
                message = str(exc)
                try:
                    file_stat = path.stat()
                    file_hash = sha256_file(path)
                    upsert_document(
                        conn,
                        path,
                        size_bytes=file_stat.st_size,
                        mtime=file_stat.st_mtime,
                        sha256=file_hash,
                        status="failed",
                        error=message,
                    )
                    conn.commit()
                except OSError:
                    conn.rollback()
                stats.failed_files += 1
                stats.errors.append((str(path), message))
        return stats
    finally:
        conn.close()
