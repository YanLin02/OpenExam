from __future__ import annotations

import sqlite3

from openexam.text_utils import normalize_text, split_query_terms


def make_fts_query(query: str) -> str:
    terms = split_query_terms(query)
    if not terms:
        return ""
    escaped = [term.replace('"', '""') for term in terms]
    return " AND ".join(f'text_norm : "{term}"' for term in escaped)


def fts_search(conn: sqlite3.Connection, query: str, limit: int) -> list[sqlite3.Row]:
    match_query = make_fts_query(query)
    if not match_query:
        return []
    try:
        return list(
            conn.execute(
                """
                SELECT
                  c.id AS chunk_db_id,
                  c.document_id,
                  c.chunk_id,
                  c.source_path,
                  c.file_name,
                  c.location_type,
                  c.location_label,
                  c.page_number,
                  c.paragraph_index,
                  c.slide_number,
                  c.text,
                  c.text_norm,
                  bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, limit),
            )
        )
    except sqlite3.OperationalError:
        return []


def all_chunks_for_fuzzy(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
              c.id AS chunk_db_id,
              c.document_id,
              c.chunk_id,
              c.source_path,
              c.file_name,
              c.location_type,
              c.location_label,
              c.page_number,
              c.paragraph_index,
              c.slide_number,
              c.text,
              c.text_norm
            FROM chunks c
            ORDER BY c.id
            LIMIT ?
            """,
            (limit,),
        )
    )


def normalized_query(query: str) -> str:
    return normalize_text(query)
