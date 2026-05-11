from __future__ import annotations

import json
from typing import Any


def build_ask_signature(
    *,
    query: str,
    answer_mode: str = "ask",
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
    llm_model: str,
    evidence_policy: str,
    detail: str,
) -> str:
    payload: dict[str, Any] = {
        "query": query,
        "answer_mode": answer_mode,
        "mode": mode,
        "scope": scope,
        "prefer": prefer,
        "per_file_cap": int(per_file_cap),
        "top_k": int(top_k),
        "llm_model": llm_model,
        "evidence_policy": evidence_policy,
        "detail": detail,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_search_signature(
    *,
    query: str,
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
) -> str:
    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "scope": scope,
        "prefer": prefer,
        "per_file_cap": int(per_file_cap),
        "top_k": int(top_k),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def preview_state_key(prefix: str, chunk_db_id: int, source_path: str, page_number: int | None) -> str:
    return f"{prefix}:preview:{chunk_db_id}:{source_path}:{page_number}"


def preview_toggle_label(is_visible: bool) -> str:
    return "隐藏预览" if is_visible else "预览该页"


def compact_index_status(index_exists: bool, documents: int, chunks: int, semantic_ready: bool, embeddings: int) -> str:
    index_label = "ready" if index_exists else "missing"
    semantic_label = "ready" if semantic_ready else "missing"
    return f"Index: {index_label} | docs {documents} | chunks {chunks} | semantic {semantic_label} | embeddings {embeddings}"


def compact_source_status(lecture: int, textbook_ocr: int, other: int) -> str:
    return f"Source: lecture {lecture} | textbook_ocr {textbook_ocr} | other {other}"


def compact_ollama_status(reachable: bool, models: list[str]) -> str:
    status = "running" if reachable else "not reachable"
    if not models:
        return f"Ollama: {status}"
    return f"Ollama: {status} | Models: {', '.join(models)}"
