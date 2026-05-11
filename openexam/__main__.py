from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openexam.ask import LLMError, ask_question, render_ask_response
from openexam.config import DEFAULT_CONFIG
from openexam.db import connect, failed_documents, index_stats
from openexam.embeddings import EmbeddingError, build_embeddings, embedding_status
from openexam.file_utils import file_uri, open_local_file, open_pdf_page_in_chrome
from openexam.ingest import ingest_directory
from openexam.ollama_utils import ensure_ollama_running
from openexam.problem_types import ProblemType, classify_problem
from openexam.priority_sources import find_indexed_answer_bank_sources
from openexam.search import search_index
from openexam.solve import render_solve_response, solve_question


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
        if args.mode in {"semantic", "hybrid"} and embedding_status(DEFAULT_CONFIG).valid:
            ollama_status = ensure_ollama_running(
                DEFAULT_CONFIG.ollama_base_url,
                auto_start=args.auto_start_ollama,
                log_path=DEFAULT_CONFIG.index_dir / "ollama.log",
            )
            if args.mode == "semantic" and not ollama_status.reachable:
                print(ollama_status.message, file=sys.stderr)
                return 2
        timing: dict[str, float] = {}
        results = search_index(
            args.query,
            top_k=args.top_k,
            config=DEFAULT_CONFIG,
            mode=args.mode,
            scope=args.scope,
            prefer=args.prefer,
            per_file_cap=args.per_file_cap,
            timing=timing,
        )
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
    print(
        "timing: "
        f"total_time_ms={timing.get('total_time_ms', 0.0):.1f}, "
        f"retrieval_time_ms={timing.get('retrieval_time_ms', 0.0):.1f}, "
        f"semantic_time_ms={timing.get('semantic_time_ms', 0.0):.1f}, "
        f"ranking_time_ms={timing.get('ranking_time_ms', 0.0):.1f}"
    )
    for index, result in enumerate(results, start=1):
        print(
            f"\n[{index}] mode {result.mode} | score {result.score:.2f} | "
            f"{result.file_name} | {_format_location(result)} | {result.source_type} | {result.match_type}"
        )
        print(result.snippet)
        print(result.source_path)
        if result.page_number is not None:
            print(file_uri(result.source_path, result.page_number))
    if args.open_first and results:
        if args.open_first_method == "chrome" and results[0].page_number is not None:
            ok, message = open_pdf_page_in_chrome(results[0].source_path, results[0].page_number)
        else:
            ok, message = open_local_file(results[0].source_path)
        if not ok:
            print(message, file=sys.stderr)
            return 2
        print(message)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not DEFAULT_CONFIG.db_path.exists():
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.", file=sys.stderr)
        return 2
    if not args.question.strip():
        print("Empty question. Please provide a question.", file=sys.stderr)
        return 2
    try:
        response = ask_question(
            args.question,
            config=DEFAULT_CONFIG,
            mode=args.mode,
            scope=args.scope,
            prefer=args.prefer,
            per_file_cap=args.per_file_cap,
            top_k=args.top_k,
            llm_model=args.llm_model,
            evidence_policy=args.evidence_policy,
            detail=args.detail,
            auto_start_ollama=args.auto_start_ollama,
            priority_answer_bank=args.priority_answer_bank,
        )
    except EmbeddingError as exc:
        print(f"Retrieval unavailable: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"检索配置：mode={response.search_mode}, scope={response.scope}, prefer={response.prefer}, "
        f"per_file_cap={response.per_file_cap}, top_k={response.top_k}, llm_model={response.llm_model}, "
        f"evidence_policy={response.evidence_policy}, evidence_status={response.evidence_status}, "
        f"detail={response.detail}, priority_answer_bank={response.priority_answer_bank}\n"
    )
    print(
        "timing: "
        f"retrieval_time_ms={response.timing.get('retrieval_time_ms', 0.0):.1f}, "
        f"prompt_build_time_ms={response.timing.get('prompt_build_time_ms', 0.0):.1f}, "
        f"llm_time_ms={response.timing.get('llm_time_ms', 0.0):.1f}, "
        f"total_time_ms={response.timing.get('total_time_ms', 0.0):.1f}\n"
    )
    print(render_ask_response(response))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    if not args.question.strip():
        print("Empty question. Please provide a question.", file=sys.stderr)
        return 2
    effective_problem_type = classify_problem(args.question).value if args.problem_type == "auto" else args.problem_type
    if not DEFAULT_CONFIG.db_path.exists() and effective_problem_type != ProblemType.CALCULATION.value:
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.", file=sys.stderr)
        return 2
    try:
        response = solve_question(
            args.question,
            mode=args.problem_type,
            config=DEFAULT_CONFIG,
            search_mode=args.mode,
            scope=args.scope,
            prefer=args.prefer,
            per_file_cap=args.per_file_cap,
            top_k=args.top_k,
            llm_model=args.llm_model,
            evidence_policy=args.evidence_policy,
            detail=args.detail,
            auto_start_ollama=args.auto_start_ollama,
            priority_answer_bank=args.priority_answer_bank,
        )
    except EmbeddingError as exc:
        print(f"Retrieval unavailable: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    ask_response = response.ask_response
    print(
        f"解题配置：problem_type={response.problem_type.value}, mode={ask_response.search_mode}, "
        f"scope={ask_response.scope}, prefer={ask_response.prefer}, per_file_cap={ask_response.per_file_cap}, "
        f"top_k={ask_response.top_k}, llm_model={ask_response.llm_model}, "
        f"evidence_policy={ask_response.evidence_policy}, evidence_status={ask_response.evidence_status}, "
        f"detail={ask_response.detail}, priority_answer_bank={ask_response.priority_answer_bank}, "
        f"priority_answer_bank_arg={args.priority_answer_bank if args.priority_answer_bank is not None else 'auto'}\n"
    )
    print(
        "timing: "
        f"retrieval_time_ms={ask_response.timing.get('retrieval_time_ms', 0.0):.1f}, "
        f"prompt_build_time_ms={ask_response.timing.get('prompt_build_time_ms', 0.0):.1f}, "
        f"llm_time_ms={ask_response.timing.get('llm_time_ms', 0.0):.1f}, "
        f"total_time_ms={ask_response.timing.get('total_time_ms', 0.0):.1f}\n"
    )
    print(render_solve_response(response))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    if not DEFAULT_CONFIG.db_path.exists():
        print(f"Index not found: {DEFAULT_CONFIG.db_path}. Run ingest first.", file=sys.stderr)
        return 2
    try:
        ollama_status = ensure_ollama_running(
            DEFAULT_CONFIG.ollama_base_url,
            auto_start=args.auto_start_ollama,
            log_path=DEFAULT_CONFIG.index_dir / "ollama.log",
        )
        if not ollama_status.reachable:
            print(f"Embedding failed: {ollama_status.message}", file=sys.stderr)
            return 2
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
        print(f"Lecture documents: {stats['lecture_documents']}")
        print(f"Textbook OCR documents: {stats['textbook_ocr_documents']}")
        print(f"Other documents: {stats['other_documents']}")
        print(f"Latest indexed at: {stats['latest_indexed_at']}")
        semantic = embedding_status(DEFAULT_CONFIG)
        print(f"Semantic index: {'ready' if semantic.valid else 'not ready'}")
        print(f"Semantic model: {semantic.model}")
        print(f"Semantic chunks: {semantic.vector_count}")
        print(f"Semantic message: {semantic.message}")
        answer_bank_sources = find_indexed_answer_bank_sources(DEFAULT_CONFIG)
        print(f"Exam answer bank: indexed {len(answer_bank_sources)}/3")
        if answer_bank_sources:
            for source in answer_bank_sources:
                print(f"- {source}")
        else:
            print("Exam answer bank message: 未检测到考试答案库文件。请将三份文件放入资料目录并重新建立索引。")
        failures = failed_documents(conn)
        if failures:
            print("\nRecent failures:")
            for row in failures:
                print(f"- {row['path']}: {row['error']}")
    finally:
        conn.close()
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from openexam.launcher import launch_streamlit

    return launch_streamlit(address=args.address, port=args.port, headless=args.headless)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m openexam", description="Offline local document search.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index a local file or directory.")
    ingest_parser.add_argument("path", help="Local file or directory to index.")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Clear the existing index before ingesting.")
    ingest_parser.set_defaults(func=cmd_ingest)

    embed_parser = subparsers.add_parser("embed", help="Build local Ollama embeddings for indexed chunks.")
    embed_parser.add_argument("--auto-start-ollama", dest="auto_start_ollama", action="store_true", default=True, help="Try to start `ollama serve` if Ollama is not reachable. Default: enabled.")
    embed_parser.add_argument("--no-auto-start-ollama", dest="auto_start_ollama", action="store_false", help="Do not try to start Ollama automatically.")
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
    search_parser.add_argument(
        "--scope",
        choices=("all", "lecture", "textbook_ocr", "other"),
        default="all",
        help="Restrict results to a source type. Default: all.",
    )
    search_parser.add_argument(
        "--prefer",
        choices=("none", "lecture", "textbook_ocr"),
        default="none",
        help="Lightly boost a source type without filtering. Default: none.",
    )
    search_parser.add_argument(
        "--per-file-cap",
        type=int,
        default=0,
        help="Maximum results per file. 0 disables the cap.",
    )
    search_parser.add_argument("--open-first", action="store_true", help="Open the top result file with macOS `open`.")
    search_parser.add_argument(
        "--open-first-method",
        choices=("default", "chrome"),
        default="default",
        help="How --open-first opens PDFs. chrome tries Google Chrome with file URI #page=N; default uses macOS open.",
    )
    search_parser.add_argument("--auto-start-ollama", dest="auto_start_ollama", action="store_true", default=True, help="Try to start `ollama serve` for semantic search if needed. Default: enabled.")
    search_parser.add_argument("--no-auto-start-ollama", dest="auto_start_ollama", action="store_false", help="Do not try to start Ollama automatically.")
    search_parser.set_defaults(func=cmd_search)

    ask_parser = subparsers.add_parser("ask", help="Answer a question using local retrieval plus local Ollama LLM citations.")
    ask_parser.add_argument("question", help="Question to answer from local indexed chunks.")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.llm_context_top_k, help="Number of retrieved chunks to pass to the local LLM.")
    ask_parser.add_argument(
        "--mode",
        choices=("keyword", "fuzzy", "hybrid", "semantic"),
        default="hybrid",
        help="Retrieval mode used before asking the local LLM.",
    )
    ask_parser.add_argument(
        "--scope",
        choices=("all", "lecture", "textbook_ocr", "other"),
        default="all",
        help="Restrict retrieved context to a source type. Default: all.",
    )
    ask_parser.add_argument(
        "--prefer",
        choices=("none", "lecture", "textbook_ocr"),
        default="lecture",
        help="Lightly boost a source type during retrieval. Default: lecture.",
    )
    ask_parser.add_argument(
        "--per-file-cap",
        type=int,
        default=2,
        help="Maximum retrieved chunks per file. 0 disables the cap. Default: 2.",
    )
    ask_parser.add_argument(
        "--llm-model",
        default=DEFAULT_CONFIG.llm_model,
        help="Local Ollama LLM model to use. Default: qwen3:8b.",
    )
    ask_parser.add_argument(
        "--evidence-policy",
        choices=("strict", "warn", "open"),
        default="warn",
        help="How ask handles insufficient local evidence. strict refuses, warn answers with warnings, open answers even with no local evidence. Default: warn.",
    )
    ask_parser.add_argument(
        "--detail",
        choices=("concise", "standard", "detailed"),
        default="standard",
        help="Answer detail level. concise is short, standard is default, detailed gives a longer explanation.",
    )
    ask_parser.add_argument("--auto-start-ollama", dest="auto_start_ollama", action="store_true", default=True, help="Try to start `ollama serve` if Ollama is not reachable. Default: enabled.")
    ask_parser.add_argument("--no-auto-start-ollama", dest="auto_start_ollama", action="store_false", help="Do not try to start Ollama automatically.")
    ask_parser.add_argument(
        "--priority-answer-bank",
        dest="priority_answer_bank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prioritize indexed exam answer bank sources during retrieval. Default: disabled for ask.",
    )
    ask_parser.set_defaults(func=cmd_ask)

    solve_parser = subparsers.add_parser("solve", help="Solve an exam-style problem using classification plus local Ask fallback.")
    solve_parser.add_argument("question", help="Exam problem to solve from local indexed chunks.")
    solve_parser.add_argument(
        "--problem-type",
        choices=("auto", "concept", "calculation", "derivation", "design", "compare", "short_answer", "unknown"),
        default="auto",
        help="Problem type. auto classifies with lightweight local rules. Default: auto.",
    )
    solve_parser.add_argument(
        "--mode",
        choices=("keyword", "fuzzy", "hybrid", "semantic"),
        default="hybrid",
        help="Retrieval mode used before asking the local LLM.",
    )
    solve_parser.add_argument(
        "--scope",
        choices=("all", "lecture", "textbook_ocr", "other"),
        default="all",
        help="Restrict retrieved context to a source type. Default: all.",
    )
    solve_parser.add_argument(
        "--prefer",
        choices=("none", "lecture", "textbook_ocr"),
        default="lecture",
        help="Lightly boost a source type during retrieval. Default: lecture.",
    )
    solve_parser.add_argument(
        "--per-file-cap",
        type=int,
        default=2,
        help="Maximum retrieved chunks per file. 0 disables the cap. Default: 2.",
    )
    solve_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CONFIG.llm_context_top_k,
        help="Number of retrieved chunks to pass to the local LLM.",
    )
    solve_parser.add_argument(
        "--evidence-policy",
        choices=("strict", "warn", "open"),
        default="warn",
        help="How solve handles insufficient local evidence. strict refuses, warn answers with warnings, open answers even with no local evidence. Default: warn.",
    )
    solve_parser.add_argument(
        "--detail",
        choices=("concise", "standard", "detailed"),
        default="standard",
        help="Answer detail level. concise is short, standard is default, detailed gives a longer explanation.",
    )
    solve_parser.add_argument(
        "--llm-model",
        default=DEFAULT_CONFIG.llm_model,
        help="Local Ollama LLM model to use. Default: qwen3:8b.",
    )
    solve_parser.add_argument("--auto-start-ollama", dest="auto_start_ollama", action="store_true", default=True, help="Try to start `ollama serve` if Ollama is not reachable. Default: enabled.")
    solve_parser.add_argument("--no-auto-start-ollama", dest="auto_start_ollama", action="store_false", help="Do not try to start Ollama automatically.")
    solve_parser.add_argument(
        "--priority-answer-bank",
        dest="priority_answer_bank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prioritize indexed exam answer bank sources. Default: auto for solve concept/short_answer.",
    )
    solve_parser.set_defaults(func=cmd_solve)

    status_parser = subparsers.add_parser("status", help="Show index statistics and recent failures.")
    status_parser.set_defaults(func=cmd_status)

    ui_parser = subparsers.add_parser("ui", help="Start the local Streamlit UI.")
    ui_parser.add_argument("--address", default="127.0.0.1", help="Address for the Streamlit server. Default: 127.0.0.1.")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit server. Default: 8501.")
    ui_parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode.")
    ui_parser.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
