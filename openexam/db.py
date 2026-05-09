from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openexam.models import ChunkRecord


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  filename TEXT NOT NULL,
  ext TEXT NOT NULL,
  size_bytes INTEGER,
  mtime REAL,
  sha256 TEXT,
  indexed_at TEXT,
  status TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  chunk_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  source_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  location_type TEXT NOT NULL,
  location_label TEXT NOT NULL,
  page_number INTEGER NULL,
  paragraph_index INTEGER NULL,
  slide_number INTEGER NULL,
  text TEXT NOT NULL,
  text_norm TEXT NOT NULL,
  char_start INTEGER NULL,
  char_end INTEGER NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text_norm,
  file_name,
  source_path,
  content='chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


def clear_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks_fts")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.commit()


def get_document_by_path(conn: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE path = ?", (str(path),)).fetchone()


def upsert_document(
    conn: sqlite3.Connection,
    path: Path,
    size_bytes: int,
    mtime: float,
    sha256: str,
    status: str = "indexed",
    error: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO documents(path, filename, ext, size_bytes, mtime, sha256, indexed_at, status, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          filename=excluded.filename,
          ext=excluded.ext,
          size_bytes=excluded.size_bytes,
          mtime=excluded.mtime,
          sha256=excluded.sha256,
          indexed_at=excluded.indexed_at,
          status=excluded.status,
          error=excluded.error
        """,
        (str(path), path.name, path.suffix.lower(), size_bytes, mtime, sha256, utc_now(), status, error),
    )
    row = get_document_by_path(conn, path)
    if row is None:
        raise RuntimeError(f"Failed to upsert document: {path}")
    return int(row["id"])


def delete_document_chunks(conn: sqlite3.Connection, document_id: int) -> None:
    rowids = [row["id"] for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
    if rowids:
        conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(rowid,) for rowid in rowids])
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))


def insert_chunks(conn: sqlite3.Connection, document_id: int, chunks: Iterable[ChunkRecord]) -> int:
    count = 0
    for chunk in chunks:
        cursor = conn.execute(
            """
            INSERT INTO chunks(
              document_id, chunk_id, chunk_index, source_path, file_name,
              location_type, location_label, page_number, paragraph_index, slide_number,
              text, text_norm, char_start, char_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk.chunk_id,
                chunk.chunk_index,
                str(chunk.source_path),
                chunk.file_name,
                chunk.location_type,
                chunk.location_label,
                chunk.page_number,
                chunk.paragraph_index,
                chunk.slide_number,
                chunk.text,
                chunk.text_norm,
                chunk.char_start,
                chunk.char_end,
            ),
        )
        rowid = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text_norm, file_name, source_path) VALUES (?, ?, ?, ?)",
            (rowid, chunk.text_norm, chunk.file_name, str(chunk.source_path)),
        )
        count += 1
    return count


def index_stats(conn: sqlite3.Connection) -> dict[str, int | str | None]:
    doc_count = conn.execute("SELECT COUNT(*) AS count FROM documents WHERE status = 'indexed'").fetchone()["count"]
    failed_count = conn.execute("SELECT COUNT(*) AS count FROM documents WHERE status = 'failed'").fetchone()["count"]
    chunk_count = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
    latest = conn.execute("SELECT MAX(indexed_at) AS latest FROM documents").fetchone()["latest"]
    return {
        "documents": int(doc_count),
        "failed_documents": int(failed_count),
        "chunks": int(chunk_count),
        "latest_indexed_at": latest,
    }


def failed_documents(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT path, error FROM documents WHERE status = 'failed' ORDER BY indexed_at DESC LIMIT ?",
            (limit,),
        )
    )
