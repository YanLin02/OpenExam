from __future__ import annotations

import sqlite3
import re
from pathlib import Path

from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.db import connect
from openexam.models import SearchResult


EXAM_ANSWER_BANK_PATTERNS = (
    "深度学习简答题_开卷检索版",
    "近五年真题",
    "附录名词术语详解",
    "《深度学习》附录名词术语详解",
)

_ANSWER_BANK_LABELS = (
    ("深度学习简答题_开卷检索版", "short_answer_bank"),
    ("近五年真题", "past_exam_bank"),
    ("《深度学习》附录名词术语详解", "terminology_bank"),
    ("附录名词术语详解", "terminology_bank"),
)


def _path_text(path: str) -> str:
    return str(path).casefold()


def is_exam_answer_bank_path(path: str) -> bool:
    text = _path_text(path)
    return any(pattern.casefold() in text for pattern in EXAM_ANSWER_BANK_PATTERNS)


def priority_source_label(path: str) -> str | None:
    text = _path_text(path)
    for pattern, label in _ANSWER_BANK_LABELS:
        if pattern.casefold() in text:
            return label
    return None


def is_exam_answer_bank_result(result: SearchResult) -> bool:
    return is_exam_answer_bank_path(result.source_path) or is_exam_answer_bank_path(result.file_name)


_MARKDOWN_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")


def split_answer_bank_markdown_sections(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        section = "\n".join(current).strip()
        current = []
        if not section:
            return
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if len(lines) <= 1 and lines and lines[0].startswith("###"):
            return
        sections.append(section)

    for line in normalized.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            if level <= 3:
                flush()
                current = [line] if level == 3 else []
                continue
        if current:
            current.append(line)
    flush()
    return sections


def is_heading_only_answer_bank_chunk(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("###"):
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) <= 1:
        return True
    body = "\n".join(lines[1:]).strip()
    return len(body) < 12


def merge_priority_results(
    results: list[SearchResult],
    top_k: int,
    priority_enabled: bool,
) -> list[SearchResult]:
    if not priority_enabled:
        return results[:top_k]

    seen: set[tuple[int, str]] = set()
    priority: list[SearchResult] = []
    regular: list[SearchResult] = []
    for result in results:
        key = (result.chunk_db_id, result.source_path)
        if key in seen:
            continue
        seen.add(key)
        if is_exam_answer_bank_result(result):
            if is_heading_only_answer_bank_chunk(result.text):
                continue
            priority.append(result)
        else:
            regular.append(result)
    return [*priority, *regular][:top_k]


def find_indexed_answer_bank_sources(config: AppConfig = DEFAULT_CONFIG) -> list[str]:
    if not config.db_path.exists():
        return []
    conn = connect(config.db_path)
    try:
        rows = conn.execute(
            """
            SELECT path, filename
            FROM documents
            WHERE status = 'indexed'
            ORDER BY filename
            """
        ).fetchall()
        sources: list[str] = []
        seen: set[str] = set()
        for row in rows:
            path = str(row["path"])
            filename = str(row["filename"])
            if not (is_exam_answer_bank_path(path) or is_exam_answer_bank_path(filename)):
                continue
            label = str(Path(path).name)
            if label not in seen:
                sources.append(label)
                seen.add(label)
        return sources
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def exam_answer_bank_label_for_result(result: SearchResult) -> str | None:
    return priority_source_label(result.source_path) or priority_source_label(result.file_name)
