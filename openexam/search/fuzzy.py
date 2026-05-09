from __future__ import annotations

from difflib import SequenceMatcher


try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - dependency fallback for minimal environments
    fuzz = None


def fuzzy_score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    if fuzz is not None:
        return fuzz.partial_ratio(query, text) / 100.0
    return SequenceMatcher(None, query, text).ratio()
