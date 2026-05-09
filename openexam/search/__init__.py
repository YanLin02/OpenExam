from __future__ import annotations

import sqlite3
import re
import time
from typing import Literal

from openexam.config import DEFAULT_CONFIG, AppConfig
from openexam.embeddings import EmbeddingError, semantic_scores
from openexam.models import SearchResult
from openexam.search.fts import all_chunks_for_fuzzy, fts_search, normalized_query
from openexam.search.fuzzy import fuzzy_score
from openexam.search.rank import (
    combine_fuzzy_scores,
    combine_hybrid_scores,
    combine_keyword_scores,
    normalize_fts_scores,
)
from openexam.sources import SearchScope, SourcePreference, scope_matches
from openexam.text_utils import make_snippet, substring_score


SearchMode = Literal["keyword", "fuzzy", "hybrid", "semantic"]
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _cjk_bigrams(text: str) -> list[str]:
    compact = "".join(CJK_RE.findall(text))
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


def _lexical_overlap_score(query_norm: str, text_norm: str) -> float:
    grams = _cjk_bigrams(query_norm)
    if not grams:
        terms = [term for term in query_norm.split() if term]
        if not terms:
            return 0.0
        return sum(1 for term in terms if term in text_norm) / len(terms)
    return sum(1 for gram in grams if gram in text_norm) / len(grams)


def _content_score(text_norm: str) -> float:
    return min(len(text_norm) / 350.0, 1.0)


def _semantic_relevance_score(query_norm: str, text_norm: str, semantic_score: float) -> float:
    if semantic_score <= 0:
        return 0.0
    lexical = _lexical_overlap_score(query_norm, text_norm)
    content = _content_score(text_norm)
    return min(1.0, 0.82 * semantic_score + 0.10 * content + 0.08 * lexical)


def _match_type(fts_value: float, substring_value: float, fuzzy_value: float, semantic_value: float, mode: SearchMode) -> str:
    parts: list[str] = []
    if mode in {"keyword", "hybrid"} and fts_value > 0:
        parts.append("fts")
    if mode in {"keyword", "hybrid"} and substring_value > 0:
        parts.append("substring")
    if mode in {"fuzzy", "hybrid"} and fuzzy_value > 0:
        parts.append("fuzzy")
    if mode in {"semantic", "hybrid"} and semantic_value > 0:
        parts.append("semantic")
    return "+".join(parts) if parts else mode


def search_index(
    query: str,
    top_k: int = 10,
    config: AppConfig = DEFAULT_CONFIG,
    mode: SearchMode = "hybrid",
    scope: SearchScope = "all",
    prefer: SourcePreference = "none",
    per_file_cap: int = 0,
    timing: dict[str, float] | None = None,
) -> list[SearchResult]:
    total_start = time.perf_counter()
    if mode not in {"keyword", "fuzzy", "hybrid", "semantic"}:
        raise ValueError("mode must be one of: keyword, fuzzy, hybrid, semantic")
    if scope not in {"all", "lecture", "textbook_ocr", "other"}:
        raise ValueError("scope must be one of: all, lecture, textbook_ocr, other")
    if prefer not in {"none", "lecture", "textbook_ocr"}:
        raise ValueError("prefer must be one of: none, lecture, textbook_ocr")
    if per_file_cap < 0:
        raise ValueError("per_file_cap must be non-negative")

    query_norm = normalized_query(query)
    if not query_norm:
        return []

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    try:
        retrieval_start = time.perf_counter()
        fts_rows = fts_search(conn, query, config.fts_candidate_limit) if mode in {"keyword", "hybrid"} else []
        fts_rows = [row for row in fts_rows if scope_matches(row["source_type"], scope)]
        fts_raw = {int(row["chunk_db_id"]): float(row["rank"]) for row in fts_rows}
        fts_norm = normalize_fts_scores(fts_raw)

        candidates: dict[int, sqlite3.Row] = {int(row["chunk_db_id"]): row for row in fts_rows}
        all_rows = [row for row in all_chunks_for_fuzzy(conn, config.fuzzy_scan_limit) if scope_matches(row["source_type"], scope)]
        all_by_id = {int(row["chunk_db_id"]): row for row in all_rows}

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

        semantic_score_map: dict[int, float] = {}
        if mode in {"semantic", "hybrid"}:
            semantic_start = time.perf_counter()
            try:
                semantic_score_map = semantic_scores(query, config=config, limit=config.semantic_candidate_limit)
            except EmbeddingError:
                if mode == "semantic":
                    raise
                semantic_score_map = {}
            if timing is not None:
                timing["semantic_time_ms"] = (time.perf_counter() - semantic_start) * 1000
            for chunk_db_id in semantic_score_map:
                row = all_by_id.get(chunk_db_id)
                if row is not None:
                    candidates.setdefault(chunk_db_id, row)

        if timing is not None:
            timing["retrieval_time_ms"] = (time.perf_counter() - retrieval_start) * 1000
        ranking_start = time.perf_counter()
        results: list[SearchResult] = []
        for chunk_db_id, row in candidates.items():
            fts_score = fts_norm.get(chunk_db_id, 0.0)
            sub_score = substring_scores.get(chunk_db_id, 0.0)
            text_score, filename_score = fuzzy_scores.get(chunk_db_id, (0.0, 0.0))
            sem_score = semantic_score_map.get(chunk_db_id, 0.0)
            sem_rank_score = _semantic_relevance_score(query_norm, row["text_norm"], sem_score)
            if mode == "keyword":
                score = combine_keyword_scores(fts_score, sub_score)
            elif mode == "fuzzy":
                score = combine_fuzzy_scores(text_score, filename_score)
            elif mode == "semantic":
                score = round(sem_rank_score * 100, 2)
            else:
                score = combine_hybrid_scores(fts_score, sub_score, text_score, filename_score, sem_rank_score)
            if score <= 0:
                continue
            if prefer != "none" and row["source_type"] == prefer:
                score = min(100.0, round(score * 1.08, 2))
            match_type = _match_type(fts_score, sub_score, text_score, sem_score, mode)
            results.append(
                SearchResult(
                    chunk_db_id=chunk_db_id,
                    document_id=int(row["document_id"]),
                    file_name=row["file_name"],
                    source_path=row["source_path"],
                    source_type=row["source_type"],
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
                    semantic_score=round(sem_score, 4),
                    match_type=match_type,
                    mode=mode,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        if timing is not None:
            timing["ranking_time_ms"] = (time.perf_counter() - ranking_start) * 1000
            timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
        if per_file_cap > 0:
            capped: list[SearchResult] = []
            counts: dict[str, int] = {}
            for result in results:
                count = counts.get(result.source_path, 0)
                if count >= per_file_cap:
                    continue
                counts[result.source_path] = count + 1
                capped.append(result)
                if len(capped) >= top_k:
                    break
            return capped
        return results[:top_k]
    finally:
        conn.close()
