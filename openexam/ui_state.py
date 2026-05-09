from __future__ import annotations

import json
from typing import Any


def build_ask_signature(
    *,
    query: str,
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
