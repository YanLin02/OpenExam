from __future__ import annotations

import re
import hashlib
import unicodedata


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s\u4e00-\u9fff]+", re.UNICODE)
HIGHLIGHT_RE_TEMPLATE = r"({})"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_query_terms(query: str) -> list[str]:
    normalized = normalize_text(query)
    return [term for term in normalized.split(" ") if term]


def query_needles(query: str) -> list[str]:
    normalized = normalize_text(query)
    terms = split_query_terms(query)
    needles: list[str] = []
    if normalized:
        needles.append(normalized)
        compact = normalized.replace(" ", "")
        if compact and compact != normalized:
            needles.append(compact)
    needles.extend(terms)
    seen: set[str] = set()
    ordered: list[str] = []
    for needle in sorted(needles, key=len, reverse=True):
        if needle and needle not in seen:
            ordered.append(needle)
            seen.add(needle)
    return ordered


def substring_score(query: str, text_norm: str) -> float:
    needles = query_needles(query)
    if not needles or not text_norm:
        return 0.0
    phrase = needles[0]
    compact_text = text_norm.replace(" ", "")
    if phrase in text_norm or phrase.replace(" ", "") in compact_text:
        return 1.0

    terms = split_query_terms(query)
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if term in text_norm or term in compact_text)
    if matched == 0:
        return 0.0
    score = matched / len(terms)
    if matched == len(terms):
        score = max(score, 0.85)
    return score


def make_snippet(text: str, query: str, max_chars: int = 260, highlight: bool = True) -> str:
    compact = SPACE_RE.sub(" ", text).strip()
    if len(compact) <= max_chars:
        snippet = compact
    else:
        normalized_compact = normalize_text(compact)
        lowered = compact.lower()
        needles = query_needles(query)
        positions: list[int] = []
        for needle in needles:
            if not needle:
                continue
            raw_pos = lowered.find(needle.lower())
            if raw_pos >= 0:
                positions.append(raw_pos)
                continue
            norm_pos = normalized_compact.find(needle)
            if norm_pos >= 0:
                positions.append(min(norm_pos, len(compact) - 1))
        if positions:
            center = min(positions)
            start = max(0, center - max_chars // 2)
        else:
            start = 0
        end = min(len(compact), start + max_chars)
        start = max(0, end - max_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(compact) else ""
        snippet = f"{prefix}{compact[start:end]}{suffix}"

    if not highlight:
        return snippet
    for needle in query_needles(query):
        if not needle or " " in needle:
            continue
        pattern = re.compile(HIGHLIGHT_RE_TEMPLATE.format(re.escape(needle)), re.IGNORECASE)
        snippet = pattern.sub(r"**\1**", snippet)
    return snippet
