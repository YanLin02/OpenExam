from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ExtractedSection(BaseModel):
    source_path: Path
    file_name: str
    location_type: str
    location_label: str
    page_number: int | None = None
    paragraph_index: int | None = None
    slide_number: int | None = None
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    source_path: Path
    file_name: str
    location_type: str
    location_label: str
    page_number: int | None = None
    paragraph_index: int | None = None
    slide_number: int | None = None
    chunk_index: int
    text: str
    text_norm: str
    char_start: int | None = None
    char_end: int | None = None


class IngestStats(BaseModel):
    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    chunks_indexed: int = 0
    errors: list[tuple[str, str]] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_db_id: int
    document_id: int
    file_name: str
    source_path: str
    location_type: str
    location_label: str
    page_number: int | None = None
    paragraph_index: int | None = None
    slide_number: int | None = None
    chunk_id: str
    text: str
    snippet: str
    score: float
    fts_score: float = 0.0
    substring_score: float = 0.0
    fuzzy_text_score: float = 0.0
    fuzzy_filename_score: float = 0.0
    semantic_score: float = 0.0
    match_type: str = "fts"
    mode: str = "hybrid"
