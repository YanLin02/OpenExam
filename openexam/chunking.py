from __future__ import annotations

import hashlib

from openexam.models import ChunkRecord, ExtractedSection
from openexam.text_utils import normalize_text


def _window_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    windows: list[tuple[int, int, str]] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
            if boundary > start + chunk_size * 0.6:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            windows.append((start, end, chunk))
        if end >= len(text):
            break
        start = max(0, end - overlap)
        if start >= len(text):
            break
        if len(windows) > 0 and start < windows[-1][1] - step - overlap:
            start = windows[-1][1] - overlap
    return windows


def make_chunk_id(section: ExtractedSection, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{section.source_path}|{section.location_label}|{chunk_index}|{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def chunk_sections(
    sections: list[ExtractedSection],
    chunk_size: int = 1000,
    overlap: int = 120,
) -> list[ChunkRecord]:
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for section in sections:
        for start, end, text in _window_text(section.text, chunk_size=chunk_size, overlap=overlap):
            chunk_index += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(section, chunk_index, text),
                    source_path=section.source_path,
                    file_name=section.file_name,
                    location_type=section.location_type,
                    location_label=section.location_label,
                    page_number=section.page_number,
                    paragraph_index=section.paragraph_index,
                    slide_number=section.slide_number,
                    chunk_index=chunk_index,
                    text=text,
                    text_norm=normalize_text(text),
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks
