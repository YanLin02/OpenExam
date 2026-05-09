from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openexam.config import DEFAULT_CONFIG
from openexam.db import connect, failed_documents, index_stats
from openexam.embeddings import EmbeddingError, build_embeddings, embedding_status
from openexam.ingest import ingest_directory
from openexam.search import search_index


def cmd_ingest(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.expanduser().exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    stats = ingest_directory(root, config=DEFAULT_CONFIG, rebuild=args.rebuild)
    print(f"Scanned files: {stats.scanned_files}")
    print(f"Indexed files: {stats.indexed_files}")
    print(f"Skipped unchanged files: {stats.skipped_files}")
    print(f"Failed files: {stats.failed_files}")
    print(f"Chunks indexed: {stats.chunks_indexed}")
    if stats.errors:
        print("\nWarnings / failed files:")
        for path, error in stats.errors:
            print(f"- {path}: {error}")
    print(f"\nIndex database: {DEFAULT_CONFIG.db_path}")
    return 0 if stats.failed_files == 0 else 1


def _format_location(result) -> str:
    if result.page_number is not None:
        return f"page {result.page_number}"
    if result.slide_number is not None:
        return f"slide {result.slide_number}"
    if result.paragraph_index is not None:
        return f"paragraph {result.paragraph_index}"
    return result.location_label


def cmd_search(args: argparse.Namespace) -> int:
    if not DEFAULT_CONFIG.db_path.exists():
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.", file=sys.stderr)
        return 2
    if not args.query.strip():
        print("Empty query. Please provide search text.", file=sys.stderr)
        return 2
    try:
        results = search_index(args.query, top_k=args.top_k, config=DEFAULT_CONFIG, mode=args.mode)
    except EmbeddingError as exc:
        print(f"Semantic search unavailable: {exc}", file=sys.stderr)
        print("Run `python3 -m openexam embed` after starting Ollama and installing the embedding model.", file=sys.stderr)
        return 2
    if not results:
        print(f"No results found. mode={args.mode}")
        if args.mode == "semantic":
            status = embedding_status(DEFAULT_CONFIG)
            print(f"Semantic index status: {status.message}")
        return 1
    for index, result in enumerate(results, start=1):
        print(
            f"\n[{index}] mode {result.mode} | score {result.score:.2f} | "
            f"{result.file_name} | {_format_location(result)} | {result.match_type}"
        )
        print(result.snippet)
        print(result.source_path)
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    if not DEFAULT_CONFIG.db_path.exists():
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.", file=sys.stderr)
        return 2
    try:
        stats = build_embeddings(DEFAULT_CONFIG)
    except EmbeddingError as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        print("If Ollama is not running, start it with: ollama serve", file=sys.stderr)
        print("If the model is missing, pull it while online: ollama pull bge-m3", file=sys.stderr)
        return 2
    print(f"Embedding provider: {DEFAULT_CONFIG.embedding_provider}")
    print(f"Embedding model: {stats.model}")
    print(f"Embedded chunks: {stats.chunk_count}")
    print(f"Vector dimension: {stats.vector_dim}")
    print(f"Vectors: {stats.npy_path}")
    print(f"Metadata: {stats.json_path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not DEFAULT_CONFIG.db_path.exists():
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.")
        return 1
    conn = connect(DEFAULT_CONFIG.db_path)
    try:
        stats = index_stats(conn)
        print(f"Indexed documents: {stats['documents']}")
        print(f"Failed documents: {stats['failed_documents']}")
        print(f"Chunks: {stats['chunks']}")
        print(f"Latest indexed at: {stats['latest_indexed_at']}")
        semantic = embedding_status(DEFAULT_CONFIG)
        print(f"Semantic index: {'ready' if semantic.valid else 'not ready'}")
        print(f"Semantic model: {semantic.model}")
        print(f"Semantic chunks: {semantic.vector_count}")
        print(f"Semantic message: {semantic.message}")
        failures = failed_documents(conn)
        if failures:
            print("\nRecent failures:")
            for row in failures:
                print(f"- {row['path']}: {row['error']}")
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m openexam", description="Offline local document search.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index a local file or directory.")
    ingest_parser.add_argument("path", help="Local file or directory to index.")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Clear the existing index before ingesting.")
    ingest_parser.set_defaults(func=cmd_ingest)

    embed_parser = subparsers.add_parser("embed", help="Build local Ollama embeddings for indexed chunks.")
    embed_parser.set_defaults(func=cmd_embed)

    search_parser = subparsers.add_parser("search", help="Search the local index.")
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.default_top_k, help="Number of results to show.")
    search_parser.add_argument(
        "--mode",
        choices=("keyword", "fuzzy", "hybrid", "semantic"),
        default="hybrid",
        help="Search mode: keyword uses FTS5 plus substring fallback; fuzzy uses rapidfuzz; semantic uses local embeddings; hybrid combines available signals.",
    )
    search_parser.set_defaults(func=cmd_search)

    status_parser = subparsers.add_parser("status", help="Show index statistics and recent failures.")
    status_parser.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
