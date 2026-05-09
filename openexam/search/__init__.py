from __future__ import annotations

import sqlite3
from typing import Literal

from openexam.config import DEFAULT_CONFIG, AppConfig
from openexam.models import SearchResult
from openexam.search.fts import all_chunks_for_fuzzy, fts_search, normalized_query
from openexam.search.fuzzy import fuzzy_score
from openexam.search.rank import (
    combine_fuzzy_scores,
    combine_hybrid_scores,
    combine_keyword_scores,
    normalize_fts_scores,
)
from openexam.text_utils import make_snippet, substring_score


SearchMode = Literal["keyword", "fuzzy", "hybrid"]


def _match_type(fts_value: float, substring_value: float, fuzzy_value: float, mode: SearchMode) -> str:
    parts: list[str] = []
    if mode in {"keyword", "hybrid"} and fts_value > 0:
        parts.append("fts")
    if mode in {"keyword", "hybrid"} and substring_value > 0:
        parts.append("substring")
    if mode in {"fuzzy", "hybrid"} and fuzzy_value > 0:
        parts.append("fuzzy")
    return "+".join(parts) if parts else mode


def search_index(
    query: str,
    top_k: int = 10,
    config: AppConfig = DEFAULT_CONFIG,
    mode: SearchMode = "hybrid",
) -> list[SearchResult]:
    if mode not in {"keyword", "fuzzy", "hybrid"}:
        raise ValueError("mode must be one of: keyword, fuzzy, hybrid")

    query_norm = normalized_query(query)
    if not query_norm:
        return []

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        fts_rows = fts_search(conn, query, config.fts_candidate_limit) if mode in {"keyword", "hybrid"} else []
        fts_raw = {int(row["chunk_db_id"]): float(row["rank"]) for row in fts_rows}
        fts_norm = normalize_fts_scores(fts_raw)

        candidates: dict[int, sqlite3.Row] = {int(row["chunk_db_id"]): row for row in fts_rows}
        all_rows = all_chunks_for_fuzzy(conn, config.fuzzy_scan_limit)

        substring_scores: dict[int, float] = {}
        if mode in {"keyword", "hybrid"}:
            for row in all_rows:
                score = substring_score(query_norm, row["text_norm"])
                if score > 0:
                    chunk_db_id = int(row["chunk_db_id"])
                    substring_scores[chunk_db_id] = score
                    candidates.setdefault(chunk_db_id, row)

        fuzzy_scores: dict[int, tuple[float, float]] = {}
        fuzzy_ranked: list[tuple[float, sqlite3.Row]] = []
        if mode in {"fuzzy", "hybrid"}:
            for row in all_rows:
                chunk_db_id = int(row["chunk_db_id"])
                text_score = fuzzy_score(query_norm, row["text_norm"])
                filename_score = fuzzy_score(query_norm, normalized_query(row["file_name"]))
                fuzzy_scores[chunk_db_id] = (text_score, filename_score)
                combined = max(text_score, filename_score * 0.8)
                if combined > 0:
                    fuzzy_ranked.append((combined, row))
            fuzzy_ranked.sort(key=lambda item: item[0], reverse=True)
            for _, row in fuzzy_ranked[: config.fuzzy_candidate_limit]:
                candidates.setdefault(int(row["chunk_db_id"]), row)

        results: list[SearchResult] = []
        for chunk_db_id, row in candidates.items():
            fts_score = fts_norm.get(chunk_db_id, 0.0)
            sub_score = substring_scores.get(chunk_db_id, 0.0)
            text_score, filename_score = fuzzy_scores.get(chunk_db_id, (0.0, 0.0))
            if mode == "keyword":
                score = combine_keyword_scores(fts_score, sub_score)
            elif mode == "fuzzy":
                score = combine_fuzzy_scores(text_score, filename_score)
            else:
                score = combine_hybrid_scores(fts_score, sub_score, text_score, filename_score)
            if score <= 0:
                continue
            match_type = _match_type(fts_score, sub_score, text_score, mode)
            results.append(
                SearchResult(
                    chunk_db_id=chunk_db_id,
                    document_id=int(row["document_id"]),
                    file_name=row["file_name"],
                    source_path=row["source_path"],
                    location_type=row["location_type"],
                    location_label=row["location_label"],
                    page_number=row["page_number"],
                    paragraph_index=row["paragraph_index"],
                    slide_number=row["slide_number"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    snippet=make_snippet(row["text"], query),
                    score=score,
                    fts_score=round(fts_score, 4),
                    substring_score=round(sub_score, 4),
                    fuzzy_text_score=round(text_score, 4),
                    fuzzy_filename_score=round(filename_score, 4),
                    match_type=match_type,
                    mode=mode,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
    finally:
        conn.close()
