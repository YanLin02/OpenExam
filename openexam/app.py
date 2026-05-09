from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from openexam.ask import (
    LLMError,
    NO_EVIDENCE,
    NO_LOCAL_EVIDENCE_WARNING,
    PARTIAL_EVIDENCE_WARNING,
    ask_question,
    format_evidence,
)
from openexam.config import DEFAULT_CONFIG
from openexam.db import connect, failed_documents, index_stats
from openexam.embeddings import EmbeddingError, embedding_status
from openexam.file_utils import open_local_file, reveal_local_file
from openexam.ingest import ingest_directory
from openexam.jobs import (
    JobRecord,
    SearchJobResult,
    create_ask_executor,
    create_search_executor,
    job_elapsed_seconds,
    submit_ask_job,
    submit_search_job,
    update_job_from_future,
)
from openexam.models import SearchResult
from openexam.ollama_utils import choose_default_llm_model, ensure_ollama_running, list_ollama_models
from openexam.pdf_preview import PdfPreviewError, render_pdf_page
from openexam.search import search_index
from openexam.ui_state import (
    build_ask_signature,
    build_search_signature,
    compact_index_status,
    compact_ollama_status,
    compact_source_status,
    preview_state_key,
    preview_toggle_label,
)


def format_location(result: SearchResult) -> str:
    if result.page_number is not None:
        return f"page {result.page_number}"
    if result.slide_number is not None:
        return f"slide {result.slide_number}"
    if result.paragraph_index is not None:
        return f"paragraph {result.paragraph_index}"
    return result.location_label


def show_parameter_help() -> None:
    with st.sidebar.expander("搜索参数怎么选", expanded=False):
        st.markdown(
            """
- `mode`: `keyword` 适合查精确术语；`fuzzy` 适合拼写不确定或中文短词；`semantic` 适合自然语言问题；`hybrid` 是默认推荐。
- `scope`: `all` 搜索全部资料；`lecture` 只搜索课件；`textbook_ocr` 只搜索 OCR 教材；`other` 只搜索其他文件。
- `prefer`: `none` 不偏向任何来源；`lecture` 轻微优先课件；`textbook_ocr` 轻微优先教材。
- `per-file-cap`: 限制同一文件最多出现几条结果，避免单个 PDF 霸榜。
- `evidence-policy`: `strict` 证据不足就拒答；`warn` 证据不足也回答但显式标注，考试推荐；`open` 无本地依据也回答但标注无本地来源。
- `top-k`: 返回或提供给 LLM 的片段数量，越大越全面但越慢。
- `detail`: `concise` 快速定位；`standard` 考试推荐；`detailed` 适合复习理解。
"""
        )


def render_index_status() -> None:
    semantic = embedding_status(DEFAULT_CONFIG)
    st.sidebar.caption(f"Index path: {DEFAULT_CONFIG.db_path}")
    st.sidebar.caption(f"Semantic path: {DEFAULT_CONFIG.embeddings_npy_path}")
    if not DEFAULT_CONFIG.db_path.exists():
        st.sidebar.warning("索引不存在。")
        st.sidebar.caption(compact_index_status(False, 0, 0, False, 0))
        return

    conn = connect(DEFAULT_CONFIG.db_path)
    try:
        stats = index_stats(conn)
        st.sidebar.caption(
            compact_index_status(
                True,
                stats["documents"],
                stats["chunks"],
                semantic.valid,
                semantic.vector_count,
            )
        )
        st.sidebar.caption(
            compact_source_status(
                stats["lecture_documents"],
                stats["textbook_ocr_documents"],
                stats["other_documents"],
            )
        )
        st.sidebar.caption(f"Embedding model: {semantic.model}. {semantic.message}")
        failures = failed_documents(conn, limit=10)
        if failures:
            with st.sidebar.expander("Recent failed files", expanded=False):
                for row in failures:
                    st.caption(row["path"])
                    st.caption(row["error"] or "")
    finally:
        conn.close()


def render_ollama_status() -> None:
    st.sidebar.subheader("Ollama")
    col1, col2 = st.sidebar.columns(2)
    refresh = col1.button("Refresh", use_container_width=True)
    start = col2.button("Start", use_container_width=True)
    if start:
        status = ensure_ollama_running(DEFAULT_CONFIG.ollama_base_url, auto_start=True, log_path=DEFAULT_CONFIG.index_dir / "ollama.log")
    else:
        status = ensure_ollama_running(DEFAULT_CONFIG.ollama_base_url, auto_start=False, log_path=DEFAULT_CONFIG.index_dir / "ollama.log")
    model_preview = status.models[:3]
    st.sidebar.caption(compact_ollama_status(status.reachable, model_preview))
    st.sidebar.caption(status.message)
    if refresh or start or status.models:
        with st.sidebar.expander("本地模型列表", expanded=False):
            if status.models:
                st.code("\n".join(status.models), language="text")
            else:
                st.caption("未找到本地模型列表。")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("管理")
        data_dir = st.text_input("资料目录", value="", key="data_dir")
        col1, col2 = st.columns(2)
        ingest_clicked = col1.button("建立/更新索引", type="primary", use_container_width=True)
        rebuild_clicked = col2.button("清空并重建", use_container_width=True)

        if ingest_clicked or rebuild_clicked:
            if not data_dir.strip():
                st.error("请输入资料目录。")
            else:
                root = Path(data_dir).expanduser()
                if not root.exists():
                    st.error(f"路径不存在：{root}")
                else:
                    with st.spinner("Indexing local files..."):
                        stats = ingest_directory(root, config=DEFAULT_CONFIG, rebuild=rebuild_clicked)
                    st.success(
                        f"Scanned {stats.scanned_files}, indexed {stats.indexed_files}, "
                        f"skipped {stats.skipped_files}, failed {stats.failed_files}, chunks {stats.chunks_indexed}."
                    )
                    if stats.scanned_files == 0:
                        st.warning("目录中没有找到支持的文件。")
                    if stats.errors:
                        with st.expander("Indexing warnings / failures", expanded=False):
                            for path, error in stats.errors:
                                st.caption(path)
                                st.caption(error)

        st.divider()
        st.subheader("索引状态")
        render_index_status()
        st.divider()
        render_ollama_status()
        st.divider()
        show_parameter_help()


def llm_model_options() -> tuple[list[str], str | None]:
    models = list_ollama_models(DEFAULT_CONFIG.ollama_base_url)
    selected = choose_default_llm_model(models, preferred=DEFAULT_CONFIG.llm_model, embedding_model=DEFAULT_CONFIG.embedding_model)
    return models, selected


@st.cache_data(show_spinner=False)
def cached_pdf_page(path: str, mtime: float, page_number: int, zoom: float) -> bytes:
    return render_pdf_page(path, page_number, zoom=zoom)


@st.cache_resource(show_spinner=False)
def get_search_executor(max_workers: int = 4):
    return create_search_executor(max_workers=max_workers)


@st.cache_resource(show_spinner=False)
def get_ask_executor(max_workers: int = 1):
    return create_ask_executor(max_workers=max_workers)


def clear_preview_state(prefix: str) -> None:
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and (
            key.startswith(f"{prefix}:preview:") or (key.startswith(f"{prefix}-") and ":preview:" in key)
        ):
            st.session_state.pop(key, None)


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_file_actions(path: str, page_number: int | None, key_prefix: str, chunk_db_id: int) -> None:
    target = Path(path)
    is_pdf_page = target.suffix.lower() == ".pdf" and page_number is not None
    action_cols = st.columns(3 if is_pdf_page else 2)
    if is_pdf_page:
        preview_key = preview_state_key(key_prefix, chunk_db_id, path, page_number)
        preview_visible = bool(st.session_state.get(preview_key))
        if action_cols[0].button(preview_toggle_label(preview_visible), key=f"{key_prefix}-preview-toggle", use_container_width=True):
            if preview_visible:
                st.session_state.pop(preview_key, None)
            else:
                st.session_state[preview_key] = True
            rerun_app()
        open_col = action_cols[1]
        reveal_col = action_cols[2]
    else:
        open_col = action_cols[0]
        reveal_col = action_cols[1]

    if open_col.button("打开文件", key=f"{key_prefix}-open", use_container_width=True):
        ok, message = open_local_file(path)
        if ok:
            st.success(message)
        else:
            st.error(message)
    if reveal_col.button("Finder", key=f"{key_prefix}-reveal", use_container_width=True):
        ok, message = reveal_local_file(path)
        if ok:
            st.success(message)
        else:
            st.error(message)

    if is_pdf_page and st.session_state.get(preview_state_key(key_prefix, chunk_db_id, path, page_number)):
        try:
            image = cached_pdf_page(str(target), target.stat().st_mtime, int(page_number), 1.5)
        except (OSError, PdfPreviewError) as exc:
            st.error(str(exc))
        else:
            st.caption(f"page {page_number}")
            st.image(image)


def render_path_expander(path: str, key_prefix: str) -> None:
    with st.expander("完整路径", expanded=False):
        st.caption(path)


def render_search_result_card(result: SearchResult, index: int, key_prefix: str = "search") -> None:
    card_key = f"{key_prefix}-{result.chunk_db_id}"
    with st.container(border=True):
        st.markdown(f"**{index}. {result.file_name}**")
        st.caption(
            f"{format_location(result)} | {result.source_type} | score {result.score:.2f} | "
            f"{result.mode} | {result.match_type}"
        )
        st.write(result.snippet)
        render_path_expander(result.source_path, f"{card_key}-path")
        render_file_actions(result.source_path, result.page_number, card_key, result.chunk_db_id)


def ask_answer_text(response) -> str:
    answer = response.answer.strip() or NO_EVIDENCE
    if answer == NO_EVIDENCE:
        return NO_EVIDENCE
    if response.evidence_status == "partial" and not answer.startswith(PARTIAL_EVIDENCE_WARNING):
        return f"{PARTIAL_EVIDENCE_WARNING}\n\n{answer}"
    if response.evidence_status == "none" and not answer.startswith(NO_LOCAL_EVIDENCE_WARNING):
        return f"{NO_LOCAL_EVIDENCE_WARNING}\n\n{answer}"
    return answer


def render_source_card(result: SearchResult, index: int, key_prefix: str) -> None:
    with st.container(border=True):
        st.markdown(f"**[{index}] {result.file_name}**")
        st.caption(f"{format_location(result)} | {result.source_type} | score {result.score:.2f}")
        st.write(format_evidence(result, index))
        render_path_expander(result.source_path, f"{key_prefix}-path-{index}-{result.chunk_db_id}")
        render_file_actions(result.source_path, result.page_number, f"{key_prefix}-{index}-{result.chunk_db_id}", result.chunk_db_id)


def render_ask_summary(response, stale: bool = False) -> None:
    if stale:
        st.warning("参数已改变，请点击搜索更新结果。下面显示的是旧结果。")
    config_text = (
        f"mode={response.search_mode}, scope={response.scope}, prefer={response.prefer}, "
        f"per_file_cap={response.per_file_cap}, top_k={response.top_k}"
    )
    st.caption(
        f"evidence={response.evidence_status} | total {response.timing.get('total_time_ms', 0.0):.1f} ms | "
        f"retrieval {response.timing.get('retrieval_time_ms', 0.0):.1f} ms | "
        f"llm {response.timing.get('llm_time_ms', 0.0):.1f} ms"
    )
    st.caption(f"{config_text} | llm_model={response.llm_model} | policy={response.evidence_policy} | detail={response.detail}")
    st.markdown(ask_answer_text(response).replace("\n", "  \n"))


def render_ask_details(response, key_prefix: str = "ask") -> None:
    with st.expander("依据片段", expanded=False):
        if response.results:
            for index, result in enumerate(response.results, start=1):
                st.write(format_evidence(result, index))
        else:
            st.write("无本地依据")
    with st.expander("来源列表", expanded=True):
        if response.results:
            for index, result in enumerate(response.results, start=1):
                render_source_card(result, index, key_prefix)
        else:
            st.write("无本地来源")


def render_controls() -> tuple[str, str, str, int, str, int, str, str, str, str, bool, bool, bool, int]:
    top_cols = st.columns([2, 2, 2, 1])
    action = top_cols[0].radio("Action", options=["Search", "Ask local AI", "Parallel Search", "Parallel Ask"], horizontal=True)
    is_ask = action in {"Ask local AI", "Parallel Ask"}
    is_parallel = action in {"Parallel Search", "Parallel Ask"}
    mode = top_cols[1].selectbox("Mode", options=["hybrid", "keyword", "fuzzy", "semantic"], index=0)
    scope = top_cols[2].selectbox("Scope", options=["all", "lecture", "textbook_ocr", "other"], index=0)
    top_default = DEFAULT_CONFIG.llm_context_top_k if is_ask else 10
    top_k = top_cols[3].number_input("Top-k", min_value=1, max_value=50, value=top_default, step=1)

    refresh_clicked = False
    if is_parallel:
        submit_label = "提交搜索任务" if action == "Parallel Search" else "提交提问任务"
        clear_label = "清除搜索任务" if action == "Parallel Search" else "清除提问任务"
        query_cols = st.columns([7, 1.4, 1.4, 1.4])
        query = query_cols[0].text_area(
            "批量输入",
            value="",
            label_visibility="collapsed",
            placeholder="每行一个关键词、术语或问题",
            height=120,
        )
        search_clicked = query_cols[1].button(submit_label, type="primary", use_container_width=True)
        refresh_clicked = query_cols[2].button("刷新任务状态", use_container_width=True)
        clear_clicked = query_cols[3].button(clear_label, use_container_width=True)
    else:
        query_cols = st.columns([8, 1, 1])
        query = query_cols[0].text_input("搜索 / 问题", value="", label_visibility="collapsed", placeholder="输入关键词、术语或问题")
        search_clicked = query_cols[1].button("搜索", type="primary", use_container_width=True)
        clear_clicked = query_cols[2].button("清除", use_container_width=True)

    ask_workers = 1
    if is_ask:
        param_cols = st.columns([2, 1, 2, 2, 3, 1] if action == "Parallel Ask" else [2, 1, 2, 2, 3])
    else:
        param_cols = st.columns([2, 1])
    prefer = param_cols[0].selectbox("Prefer", options=["none", "lecture", "textbook_ocr"], index=1 if is_ask else 0)
    cap_default = 2 if is_ask else 0
    per_file_cap = param_cols[1].number_input("Per-file cap", min_value=0, max_value=20, value=cap_default, step=1)

    evidence_policy = "warn"
    detail = "standard"
    llm_model = DEFAULT_CONFIG.llm_model
    if is_ask:
        evidence_policy = param_cols[2].selectbox("Evidence", options=["warn", "strict", "open"], index=0)
        detail_label = param_cols[3].selectbox("Detail", options=["简洁", "标准", "详细"], index=1)
        detail = {"简洁": "concise", "标准": "standard", "详细": "detailed"}[detail_label]
        models, selected_llm = llm_model_options()
        if selected_llm is None:
            param_cols[4].warning("未找到本地 LLM 模型。")
            llm_model = param_cols[4].text_input("LLM model", value=DEFAULT_CONFIG.llm_model)
        else:
            model_index = models.index(selected_llm) if selected_llm in models else 0
            llm_model = param_cols[4].selectbox("LLM model", options=models, index=model_index)
        if action == "Parallel Ask":
            ask_workers = int(param_cols[5].selectbox("Ask workers", options=[1, 2], index=0))
    return (
        action,
        query,
        mode,
        int(top_k),
        scope,
        int(per_file_cap),
        prefer,
        evidence_policy,
        detail,
        llm_model,
        search_clicked,
        refresh_clicked,
        clear_clicked,
        ask_workers,
    )


def main() -> None:
    st.set_page_config(page_title="OpenExam", layout="wide")
    render_sidebar()

    st.markdown("# OpenExam")
    (
        action,
        query,
        mode,
        top_k,
        scope,
        per_file_cap,
        prefer,
        evidence_policy,
        detail,
        llm_model,
        search_clicked,
        refresh_clicked,
        clear_clicked,
        ask_workers,
    ) = render_controls()

    if action == "Ask local AI":
        handle_ask(query, mode, scope, prefer, per_file_cap, top_k, llm_model, evidence_policy, detail, search_clicked, clear_clicked)
    elif action == "Search":
        handle_search(query, mode, scope, prefer, per_file_cap, top_k, search_clicked, clear_clicked)
    elif action == "Parallel Ask":
        handle_parallel_ask(
            query,
            mode,
            scope,
            prefer,
            per_file_cap,
            top_k,
            llm_model,
            evidence_policy,
            detail,
            ask_workers,
            search_clicked,
            refresh_clicked,
            clear_clicked,
        )
    else:
        handle_parallel_search(query, mode, scope, prefer, per_file_cap, top_k, search_clicked, refresh_clicked, clear_clicked)


def batch_inputs(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def session_jobs(key: str) -> list[JobRecord]:
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def update_jobs(jobs: list[JobRecord]) -> None:
    for job in jobs:
        update_job_from_future(job)


def render_job_header(job: JobRecord, index: int) -> None:
    st.markdown(f"**{index}. {job.input_text}**")
    st.caption(f"status={job.status} | elapsed {job_elapsed_seconds(job):.1f}s")


def render_parallel_search_job(job: JobRecord, index: int) -> None:
    with st.container(border=True):
        render_job_header(job, index)
        if job.status == "error":
            st.error(job.error or "Search job failed.")
            return
        if job.status != "done":
            return
        if not isinstance(job.result, SearchJobResult):
            st.error("Search job returned an unexpected result.")
            return

        result = job.result
        st.caption(
            f"total {result.timing.get('total_time_ms', 0.0):.1f} ms | "
            f"retrieval {result.timing.get('retrieval_time_ms', 0.0):.1f} ms | "
            f"semantic {result.timing.get('semantic_time_ms', 0.0):.1f} ms | "
            f"ranking {result.timing.get('ranking_time_ms', 0.0):.1f} ms"
        )
        if not result.results:
            st.info("No results found.")
            return
        for result_index, search_result in enumerate(result.results, start=1):
            render_search_result_card(search_result, result_index, key_prefix=f"parallel-search-{job.job_id}")


def render_parallel_ask_job(job: JobRecord, index: int) -> None:
    with st.container(border=True):
        render_job_header(job, index)
        if job.status == "error":
            st.error(job.error or "Ask job failed.")
            return
        if job.status != "done":
            return
        if job.result is None:
            st.error("Ask job returned an empty result.")
            return

        render_ask_summary(job.result)
        render_ask_details(job.result, key_prefix=f"parallel-ask-{job.job_id}")


def handle_parallel_search(
    query_text: str,
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
    submit_clicked: bool,
    refresh_clicked: bool,
    clear_clicked: bool,
) -> None:
    jobs = session_jobs("search_jobs")
    update_jobs(jobs)

    if clear_clicked:
        st.session_state["search_jobs"] = []
        clear_preview_state("parallel-search")
        st.info("已清除搜索任务。")
        return

    if submit_clicked:
        queries = batch_inputs(query_text)
        if not queries:
            st.warning("请输入至少一行搜索内容。")
            return
        if not DEFAULT_CONFIG.db_path.exists():
            st.warning("还没有索引，请先建立索引。")
            return

        executor = get_search_executor(max_workers=4)
        for query in queries:
            signature = build_search_signature(
                query=query,
                mode=mode,
                scope=scope,
                prefer=prefer,
                per_file_cap=per_file_cap,
                top_k=top_k,
            )
            jobs.append(
                submit_search_job(
                    executor,
                    query,
                    signature=signature,
                    config=DEFAULT_CONFIG,
                    mode=mode,
                    scope=scope,
                    prefer=prefer,
                    per_file_cap=per_file_cap,
                    top_k=top_k,
                )
            )
        st.success(f"已提交 {len(queries)} 个搜索任务。")

    if refresh_clicked:
        st.info("任务状态已刷新。")
    if not jobs:
        st.info("提交批量搜索任务后，结果会按任务独立显示。")
        return
    for index, job in enumerate(jobs, start=1):
        render_parallel_search_job(job, index)


def handle_parallel_ask(
    query_text: str,
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
    llm_model: str,
    evidence_policy: str,
    detail: str,
    ask_workers: int,
    submit_clicked: bool,
    refresh_clicked: bool,
    clear_clicked: bool,
) -> None:
    jobs = session_jobs("ask_jobs")
    update_jobs(jobs)

    if clear_clicked:
        st.session_state["ask_jobs"] = []
        clear_preview_state("parallel-ask")
        st.info("已清除提问任务。")
        return

    if submit_clicked:
        questions = batch_inputs(query_text)
        if not questions:
            st.warning("请输入至少一行问题。")
            return
        if not DEFAULT_CONFIG.db_path.exists():
            st.warning("还没有索引，请先建立索引。")
            return

        executor = get_ask_executor(max_workers=ask_workers)
        for question in questions:
            signature = build_ask_signature(
                query=question,
                mode=mode,
                scope=scope,
                prefer=prefer,
                per_file_cap=per_file_cap,
                top_k=top_k,
                llm_model=llm_model,
                evidence_policy=evidence_policy,
                detail=detail,
            )
            jobs.append(
                submit_ask_job(
                    executor,
                    question,
                    signature=signature,
                    config=DEFAULT_CONFIG,
                    mode=mode,
                    scope=scope,
                    prefer=prefer,
                    per_file_cap=per_file_cap,
                    top_k=top_k,
                    llm_model=llm_model,
                    evidence_policy=evidence_policy,
                    detail=detail,
                )
            )
        st.success(f"已提交 {len(questions)} 个提问任务。")

    if refresh_clicked:
        st.info("任务状态已刷新。")
    if not jobs:
        st.info("提交批量提问任务后，默认按单 worker 排队执行。")
        return
    st.caption(f"Ask workers for newly submitted tasks: {ask_workers}")
    for index, job in enumerate(jobs, start=1):
        render_parallel_ask_job(job, index)


def handle_ask(
    query: str,
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
    llm_model: str,
    evidence_policy: str,
    detail: str,
    search_clicked: bool,
    clear_clicked: bool,
) -> None:
    ask_signature = build_ask_signature(
        query=query,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=top_k,
        llm_model=llm_model,
        evidence_policy=evidence_policy,
        detail=detail,
    )
    if clear_clicked:
        st.session_state.pop("last_ask_signature", None)
        st.session_state.pop("last_ask_response", None)
        clear_preview_state("ask")
        st.info("已清除 Ask 结果。")
        return
    if search_clicked:
        if not query.strip():
            st.warning("请输入问题。")
            return
        if not DEFAULT_CONFIG.db_path.exists():
            st.warning("还没有索引，请先建立索引。")
            return
        try:
            response = ask_question(
                query,
                config=DEFAULT_CONFIG,
                mode=mode,
                scope=scope,
                prefer=prefer,
                per_file_cap=per_file_cap,
                top_k=top_k,
                llm_model=llm_model,
                evidence_policy=evidence_policy,
                detail=detail,
            )
        except (EmbeddingError, LLMError) as exc:
            st.error(str(exc))
            st.info("请确认 Ollama 已启动：ollama serve；如果模型不存在，请联网时提前运行：ollama pull qwen3:8b。")
            return
        st.session_state["last_ask_signature"] = ask_signature
        st.session_state["last_ask_response"] = response

    response = st.session_state.get("last_ask_response")
    if response is None:
        st.info("点击“搜索”后才会调用本地 LLM。")
        return
    stale = st.session_state.get("last_ask_signature") != ask_signature
    render_ask_summary(response, stale=stale)
    render_ask_details(response)


def handle_search(
    query: str,
    mode: str,
    scope: str,
    prefer: str,
    per_file_cap: int,
    top_k: int,
    search_clicked: bool,
    clear_clicked: bool,
) -> None:
    search_signature = build_search_signature(
        query=query,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=top_k,
    )
    if clear_clicked:
        st.session_state.pop("last_search_signature", None)
        st.session_state.pop("last_search_response", None)
        st.session_state.pop("last_search_timing", None)
        clear_preview_state("search")
        st.info("已清除搜索结果。")
        return
    if search_clicked:
        if not query.strip():
            st.warning("请输入搜索内容。")
            return
        if not DEFAULT_CONFIG.db_path.exists():
            st.warning("还没有索引，请先建立索引。")
            return
        try:
            timing: dict[str, float] = {}
            results = search_index(
                query,
                top_k=top_k,
                config=DEFAULT_CONFIG,
                mode=mode,
                scope=scope,
                prefer=prefer,
                per_file_cap=per_file_cap,
                timing=timing,
            )
        except EmbeddingError as exc:
            st.error(f"Semantic search unavailable: {exc}")
            st.info("请先启动 Ollama：ollama serve；如果模型不存在，请联网时提前运行：ollama pull bge-m3。")
            return
        st.session_state["last_search_signature"] = search_signature
        st.session_state["last_search_response"] = results
        st.session_state["last_search_timing"] = timing

    results = st.session_state.get("last_search_response")
    timing = st.session_state.get("last_search_timing", {})
    if results is None:
        st.info("点击“搜索”后才会执行检索。")
        return
    if st.session_state.get("last_search_signature") != search_signature:
        st.warning("参数已改变，请点击搜索更新结果。")
    st.caption(
        f"mode={mode}, scope={scope}, prefer={prefer}, per_file_cap={per_file_cap}, top_k={top_k} | "
        f"total {timing.get('total_time_ms', 0.0):.1f} ms | retrieval {timing.get('retrieval_time_ms', 0.0):.1f} ms | "
        f"semantic {timing.get('semantic_time_ms', 0.0):.1f} ms | ranking {timing.get('ranking_time_ms', 0.0):.1f} ms"
    )
    if not results:
        st.info(f"No results found. mode={mode}")
    for index, result in enumerate(results, start=1):
        render_search_result_card(result, index)


if __name__ == "__main__":
    main()
