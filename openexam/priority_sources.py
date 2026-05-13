from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.db import connect
from openexam.models import SearchResult


DEFAULT_PRIORITY_PATTERNS: tuple[str, ...] = ()
DEFAULT_ANSWER_BANK_DIR_NAMES = (
    "answer_bank",
    "exam_answer_bank",
    "priority_sources",
    "易考",
    "重点",
    "答案库",
)

_DEFAULT_DIR_LABELS = {
    "answer_bank": "directory_answer_bank",
    "exam_answer_bank": "directory_answer_bank",
    "priority_sources": "directory_answer_bank",
    "答案库": "directory_answer_bank",
    "易考": "exam_focus_bank",
    "重点": "exam_focus_bank",
}


@dataclass(frozen=True)
class PrioritySourceConfig:
    answer_bank_dirs: tuple[str, ...]
    priority_patterns: tuple[str, ...]
    labels: dict[str, str]
    warnings: tuple[str, ...] = ()


def _dedupe(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        key = stripped.casefold()
        if not stripped or key in seen:
            continue
        seen.add(key)
        ordered.append(stripped)
    return tuple(ordered)


def default_priority_source_config() -> PrioritySourceConfig:
    return PrioritySourceConfig(
        answer_bank_dirs=DEFAULT_ANSWER_BANK_DIR_NAMES,
        priority_patterns=DEFAULT_PRIORITY_PATTERNS,
        labels=dict(_DEFAULT_DIR_LABELS),
    )


def load_priority_source_config(config: AppConfig = DEFAULT_CONFIG) -> PrioritySourceConfig:
    default = default_priority_source_config()
    config_path = config.index_dir / "priority_sources.json"
    if not config_path.exists():
        return default
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("priority_sources.json must contain a JSON object")
        extra_dirs = payload.get("answer_bank_dirs", [])
        extra_patterns = payload.get("priority_patterns", [])
        extra_labels = payload.get("labels", {})
        if not isinstance(extra_dirs, list):
            raise ValueError("answer_bank_dirs must be a list")
        if not isinstance(extra_patterns, list):
            raise ValueError("priority_patterns must be a list")
        if not isinstance(extra_labels, dict):
            raise ValueError("labels must be an object")
        labels = dict(default.labels)
        labels.update({str(key): str(value) for key, value in extra_labels.items()})
        return PrioritySourceConfig(
            answer_bank_dirs=_dedupe([*default.answer_bank_dirs, *extra_dirs]),
            priority_patterns=_dedupe([*default.priority_patterns, *extra_patterns]),
            labels=labels,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PrioritySourceConfig(
            answer_bank_dirs=default.answer_bank_dirs,
            priority_patterns=default.priority_patterns,
            labels=default.labels,
            warnings=(f"Failed to read {config_path}: {exc}",),
        )


def _path_text(path: str) -> str:
    return str(path).casefold()


def _path_parts(path: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", _path_text(path)) if part]


def _configured(config: AppConfig | PrioritySourceConfig | None) -> PrioritySourceConfig:
    if config is None:
        return load_priority_source_config(DEFAULT_CONFIG)
    if isinstance(config, PrioritySourceConfig):
        return config
    return load_priority_source_config(config)


def is_exam_answer_bank_path(path: str, config: AppConfig | PrioritySourceConfig | None = None) -> bool:
    priority_config = _configured(config)
    text = _path_text(path)
    if any(pattern.casefold() in text for pattern in priority_config.priority_patterns):
        return True
    parts = _path_parts(path)
    configured_dirs = {directory.casefold() for directory in priority_config.answer_bank_dirs}
    return any(part in configured_dirs for part in parts[:-1])


def priority_source_label(path: str, config: AppConfig | PrioritySourceConfig | None = None) -> str | None:
    priority_config = _configured(config)
    text = _path_text(path)
    for pattern in priority_config.priority_patterns:
        if pattern.casefold() in text:
            return priority_config.labels.get(pattern, "custom_answer_bank")
    parts = _path_parts(path)
    for part in parts[:-1]:
        for directory in priority_config.answer_bank_dirs:
            if part == directory.casefold():
                return priority_config.labels.get(directory, _DEFAULT_DIR_LABELS.get(directory, "directory_answer_bank"))
    return None


def is_exam_answer_bank_result(result: SearchResult, config: AppConfig | PrioritySourceConfig | None = None) -> bool:
    return is_exam_answer_bank_path(result.source_path, config) or is_exam_answer_bank_path(result.file_name, config)


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
    config: AppConfig | PrioritySourceConfig | None = None,
) -> list[SearchResult]:
    if not priority_enabled:
        return results[:top_k]

    priority_config = _configured(config)
    seen: set[tuple[int, str]] = set()
    priority: list[SearchResult] = []
    regular: list[SearchResult] = []
    for result in results:
        key = (result.chunk_db_id, result.source_path)
        if key in seen:
            continue
        seen.add(key)
        if is_exam_answer_bank_result(result, priority_config):
            if is_heading_only_answer_bank_chunk(result.text):
                continue
            priority.append(result)
        else:
            regular.append(result)
    return [*priority, *regular][:top_k]


def find_indexed_answer_bank_sources(config: AppConfig = DEFAULT_CONFIG) -> list[str]:
    if not config.db_path.exists():
        return []
    priority_config = load_priority_source_config(config)
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
            label = priority_source_label(path, priority_config) or priority_source_label(filename, priority_config)
            if label is None:
                continue
            display = f"{Path(path).name} [{label}]"
            if display not in seen:
                sources.append(display)
                seen.add(display)
        return sources
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def exam_answer_bank_label_for_result(result: SearchResult, config: AppConfig | PrioritySourceConfig | None = None) -> str | None:
    return priority_source_label(result.source_path, config) or priority_source_label(result.file_name, config)
