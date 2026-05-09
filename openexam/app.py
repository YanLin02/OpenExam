from __future__ import annotations

from pathlib import Path

import streamlit as st

from openexam.ask import LLMError, ask_question, format_evidence, format_source, render_ask_response
from openexam.config import DEFAULT_CONFIG
from openexam.db import connect, failed_documents, index_stats
from openexam.embeddings import EmbeddingError, embedding_status
from openexam.file_utils import open_local_file, open_pdf_page_in_chrome, reveal_local_file
from openexam.ingest import ingest_directory
from openexam.ollama_utils import choose_default_llm_model, ensure_ollama_running, list_ollama_models
from openexam.pdf_preview import PdfPreviewError, render_pdf_page
from openexam.search import search_index
from openexam.ui_state import build_ask_signature


def format_location(result) -> str:
    if result.page_number is not None:
        return f"page {result.page_number}"
    if result.slide_number is not None:
        return f"slide {result.slide_number}"
    if result.paragraph_index is not None:
        return f"paragraph {result.paragraph_index}"
    return result.location_label


def show_index_status() -> None:
    st.caption(f"Index path: {DEFAULT_CONFIG.db_path}")
    semantic = embedding_status(DEFAULT_CONFIG)
    st.caption(f"Semantic index path: {DEFAULT_CONFIG.embeddings_npy_path}")
    if not DEFAULT_CONFIG.db_path.exists():
        st.warning("索引不存在。请先输入资料目录并建立索引。")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", 0)
        col2.metric("Failed", 0)
        col3.metric("Chunks", 0)
        col4.metric("Semantic", "missing")
        return

    conn = connect(DEFAULT_CONFIG.db_path)
    try:
        stats = index_stats(conn)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Documents", stats["documents"])
        col2.metric("Failed", stats["failed_documents"])
        col3.metric("Chunks", stats["chunks"])
        col4.metric("Latest", stats["latest_indexed_at"] or "-")
        col5.metric("Semantic", "ready" if semantic.valid else "missing/stale")
        col6.metric("Embeddings", semantic.vector_count)
        st.caption(f"Embedding model: {semantic.model}. {semantic.message}")
        st.caption(
            f"Source types: lecture={stats['lecture_documents']}, "
            f"textbook_ocr={stats['textbook_ocr_documents']}, other={stats['other_documents']}"
        )
        failures = failed_documents(conn, limit=10)
        if failures:
            with st.expander("Recent failed files"):
                for row in failures:
                    st.caption(row["path"])
                    st.code(row["error"] or "", language="text")
    finally:
        conn.close()


def show_ollama_status() -> None:
    with st.expander("Ollama 状态", expanded=False):
        col1, col2 = st.columns(2)
        refresh = col1.button("Refresh Ollama status")
        start = col2.button("Start Ollama")
        if start:
            status = ensure_ollama_running(DEFAULT_CONFIG.ollama_base_url, auto_start=True, log_path=DEFAULT_CONFIG.index_dir / "ollama.log")
        else:
            status = ensure_ollama_running(DEFAULT_CONFIG.ollama_base_url, auto_start=False, log_path=DEFAULT_CONFIG.index_dir / "ollama.log")
        if refresh or start or True:
            st.write(f"Status: {'running' if status.reachable else 'not reachable'}")
            st.caption(status.message)
            st.write(f"Embedding model: {DEFAULT_CONFIG.embedding_model}")
            st.write(f"Default LLM model: {DEFAULT_CONFIG.llm_model}")
            if status.models:
                st.write("Available models:")
                st.code("\n".join(status.models), language="text")
            else:
                st.warning("未找到本地模型列表。请确认 Ollama 已启动。")


def show_parameter_help() -> None:
    with st.expander("搜索参数怎么选", expanded=False):
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


def llm_model_options() -> tuple[list[str], str | None]:
    models = list_ollama_models(DEFAULT_CONFIG.ollama_base_url)
    selected = choose_default_llm_model(models, preferred=DEFAULT_CONFIG.llm_model, embedding_model=DEFAULT_CONFIG.embedding_model)
    return models, selected


@st.cache_data(show_spinner=False)
def cached_pdf_page(path: str, mtime: float, page_number: int, zoom: float) -> bytes:
    return render_pdf_page(path, page_number, zoom=zoom)


def show_file_actions(path: str, page_number: int | None, key_prefix: str) -> None:
    target = Path(path)
    is_pdf_page = target.suffix.lower() == ".pdf" and page_number is not None
    if is_pdf_page:
        if st.button("预览该页", key=f"{key_prefix}-preview"):
            try:
                image = cached_pdf_page(str(target), target.stat().st_mtime, int(page_number), 1.5)
            except (OSError, PdfPreviewError) as exc:
                st.error(str(exc))
            else:
                st.caption(f"page {page_number}")
                st.image(image)
    col1, col2, col3 = st.columns(3)
    if col1.button("打开文件", key=f"{key_prefix}-open"):
        ok, message = open_local_file(path)
        if ok:
            st.success(message)
        else:
            st.error(message)
    if col2.button("在 Finder 中显示", key=f"{key_prefix}-reveal"):
        ok, message = reveal_local_file(path)
        if ok:
            st.success(message)
        else:
            st.error(message)
    if is_pdf_page and col3.button("用 Chrome 打开到该页", key=f"{key_prefix}-chrome"):
        ok, message = open_pdf_page_in_chrome(path, page_number)
        if ok:
            st.success(message)
        else:
            st.error(message)


def render_ask_response_block(response, stale: bool = False) -> None:
    if stale:
        st.warning("参数已改变，请重新点击生成回答。下面显示的是旧结果。")
    response_config_text = (
        f"mode={response.search_mode}, scope={response.scope}, prefer={response.prefer}, "
        f"per_file_cap={response.per_file_cap}, top_k={response.top_k}"
    )
    st.caption(
        f"检索配置: {response_config_text}, llm_model={response.llm_model}, "
        f"evidence_policy={response.evidence_policy}, evidence_status={response.evidence_status}, detail={response.detail}"
    )
    st.caption(
        f"耗时: 检索 {response.timing.get('retrieval_time_ms', 0.0):.1f} ms, "
        f"LLM {response.timing.get('llm_time_ms', 0.0):.1f} ms, "
        f"总计 {response.timing.get('total_time_ms', 0.0):.1f} ms"
    )
    st.subheader("LLM 回答")
    st.markdown(render_ask_response(response).replace("\n", "  \n"))
    st.subheader("依据片段")
    if response.results:
        for index, result in enumerate(response.results, start=1):
            st.write(format_evidence(result, index))
    else:
        st.write("无本地依据")
    st.subheader("来源列表")
    if response.results:
        for index, result in enumerate(response.results, start=1):
            st.code(format_source(result, index), language="text")
            show_file_actions(result.source_path, result.page_number, f"ask-{index}-{result.chunk_db_id}")
    else:
        st.code("无本地来源", language="text")


def main() -> None:
    st.set_page_config(page_title="OpenExam", layout="wide")
    st.title("OpenExam")
    st.caption("Offline local search over PDF, TXT, MD, DOCX, and PPTX files.")

    data_dir = st.text_input("资料目录", value="")
    col1, col2 = st.columns([1, 1])
    ingest_clicked = col1.button("建立/更新索引", type="primary")
    rebuild_clicked = col2.button("清空并重建索引")

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
                    with st.expander("Indexing warnings / failures"):
                        for path, error in stats.errors:
                            st.caption(path)
                            st.code(error, language="text")

    show_index_status()
    show_ollama_status()

    st.divider()
    action = st.radio("Action", options=["Search", "Ask local AI"], horizontal=True)
    show_parameter_help()
    query = st.text_input("搜索", value="")
    mode = st.selectbox("Search mode", options=["hybrid", "keyword", "fuzzy", "semantic"], index=0)
    scope = st.selectbox("Scope", options=["all", "lecture", "textbook_ocr", "other"], index=0)
    prefer_default = 1 if action == "Ask local AI" else 0
    prefer = st.selectbox("Prefer", options=["none", "lecture", "textbook_ocr"], index=prefer_default)
    cap_default = 2 if action == "Ask local AI" else 0
    per_file_cap = st.number_input("Per-file cap", min_value=0, max_value=20, value=cap_default, step=1)
    top_default = DEFAULT_CONFIG.llm_context_top_k if action == "Ask local AI" else 10
    top_k = st.number_input("Top-k", min_value=1, max_value=50, value=top_default, step=1)
    models, selected_llm = llm_model_options()
    if selected_llm is None:
        st.warning("未找到本地 LLM 模型。请联网时运行 ollama pull qwen3:8b。")
        llm_model = st.text_input("LLM model", value=DEFAULT_CONFIG.llm_model)
    else:
        model_index = models.index(selected_llm) if selected_llm in models else 0
        llm_model = st.selectbox("LLM model", options=models, index=model_index)
    evidence_policy = st.selectbox("Evidence policy", options=["warn", "strict", "open"], index=0)
    detail_label = st.selectbox("Detail", options=["简洁", "标准", "详细"], index=1)
    detail = {"简洁": "concise", "标准": "standard", "详细": "detailed"}[detail_label]
    if not query.strip():
        st.info("请输入搜索内容。")
        return
    if not DEFAULT_CONFIG.db_path.exists():
        st.warning("还没有索引，请先建立索引。")
        return

    config_text = f"mode={mode}, scope={scope}, prefer={prefer}, per_file_cap={int(per_file_cap)}, top_k={int(top_k)}"
    if action == "Ask local AI":
        ask_signature = build_ask_signature(
            query=query,
            mode=mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=int(per_file_cap),
            top_k=int(top_k),
            llm_model=llm_model,
            evidence_policy=evidence_policy,
            detail=detail,
        )
        col1, col2 = st.columns(2)
        generate_clicked = col1.button("生成回答", type="primary")
        clear_clicked = col2.button("清除回答")
        if clear_clicked:
            st.session_state.pop("last_ask_signature", None)
            st.session_state.pop("last_ask_response", None)
            st.info("已清除回答。")
            return
        if generate_clicked:
            try:
                response = ask_question(
                    query,
                    config=DEFAULT_CONFIG,
                    mode=mode,
                    scope=scope,
                    prefer=prefer,
                    per_file_cap=int(per_file_cap),
                    top_k=int(top_k),
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
            st.info("点击“生成回答”后才会调用本地 LLM。")
            return
        stale = st.session_state.get("last_ask_signature") != ask_signature
        render_ask_response_block(response, stale=stale)
        return

    try:
        timing: dict[str, float] = {}
        results = search_index(
            query,
            top_k=int(top_k),
            config=DEFAULT_CONFIG,
            mode=mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=int(per_file_cap),
            timing=timing,
        )
    except EmbeddingError as exc:
        st.error(f"Semantic search unavailable: {exc}")
        st.info("请先启动 Ollama：ollama serve；如果模型不存在，请联网时提前运行：ollama pull bge-m3。")
        return
    st.caption(f"Search config: {config_text}")
    st.caption(
        f"耗时: 检索 {timing.get('retrieval_time_ms', 0.0):.1f} ms, "
        f"语义 {timing.get('semantic_time_ms', 0.0):.1f} ms, "
        f"排序 {timing.get('ranking_time_ms', 0.0):.1f} ms, "
        f"总计 {timing.get('total_time_ms', 0.0):.1f} ms"
    )
    if not results:
        st.info(f"No results found. mode={mode}")
    for result in results:
        with st.container(border=True):
            st.subheader(result.file_name)
            st.caption(
                f"mode {result.mode} | {format_location(result)} | source {result.source_type} | "
                f"score {result.score:.2f} | {result.match_type}"
            )
            st.write(result.snippet)
            st.code(result.source_path, language="text")
            show_file_actions(result.source_path, result.page_number, f"search-{result.chunk_db_id}")


if __name__ == "__main__":
    main()
