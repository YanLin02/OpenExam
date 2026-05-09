from __future__ import annotations


def normalize_fts_scores(raw_scores: dict[int, float]) -> dict[int, float]:
    if not raw_scores:
        return {}
    ordered = sorted(raw_scores.items(), key=lambda item: item[1])
    total = max(1, len(ordered) - 1)
    return {chunk_id: 1.0 - (index / total if total else 0.0) for index, (chunk_id, _) in enumerate(ordered)}


def combine_scores(fts_score: float, fuzzy_text_score: float, fuzzy_filename_score: float) -> float:
    return round((0.65 * fts_score + 0.25 * fuzzy_text_score + 0.10 * fuzzy_filename_score) * 100, 2)


def combine_keyword_scores(fts_score: float, substring_score: float) -> float:
    return round((0.60 * fts_score + 0.40 * substring_score) * 100, 2)


def combine_fuzzy_scores(fuzzy_text_score: float, fuzzy_filename_score: float) -> float:
    return round((0.90 * fuzzy_text_score + 0.10 * fuzzy_filename_score) * 100, 2)


def combine_hybrid_scores(
    fts_score: float,
    substring_score: float,
    fuzzy_text_score: float,
    fuzzy_filename_score: float,
    semantic_score: float = 0.0,
) -> float:
    fuzzy_score = max(fuzzy_text_score, fuzzy_filename_score * 0.8)
    if semantic_score > 0:
        return round(
            (
                0.35 * fts_score
                + 0.15 * substring_score
                + 0.15 * fuzzy_score
                + 0.35 * semantic_score
            )
            * 100,
            2,
        )
    return round((0.35 * fts_score + 0.35 * substring_score + 0.20 * fuzzy_text_score + 0.10 * fuzzy_filename_score) * 100, 2)
