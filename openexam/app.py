from __future__ import annotations

from pathlib import Path

import streamlit as st

from openexam.config import DEFAULT_CONFIG
from openexam.db import connect, failed_documents, index_stats
from openexam.embeddings import EmbeddingError, embedding_status
from openexam.ingest import ingest_directory
from openexam.search import search_index


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
        failures = failed_documents(conn, limit=10)
        if failures:
            with st.expander("Recent failed files"):
                for row in failures:
                    st.caption(row["path"])
                    st.code(row["error"] or "", language="text")
    finally:
        conn.close()


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

    st.divider()
    query = st.text_input("搜索", value="")
    mode = st.selectbox("Search mode", options=["hybrid", "keyword", "fuzzy", "semantic"], index=0)
    top_k = st.number_input("Top-k", min_value=1, max_value=50, value=10, step=1)
    if not query.strip():
        st.info("请输入搜索内容。")
        return
    if not DEFAULT_CONFIG.db_path.exists():
        st.warning("还没有索引，请先建立索引。")
        return

    try:
        results = search_index(query, top_k=int(top_k), config=DEFAULT_CONFIG, mode=mode)
    except EmbeddingError as exc:
        st.error(f"Semantic search unavailable: {exc}")
        st.info("请先启动 Ollama：ollama serve；如果模型不存在，请联网时提前运行：ollama pull bge-m3。")
        return
    if not results:
        st.info(f"No results found. mode={mode}")
    for result in results:
        with st.container(border=True):
            st.subheader(result.file_name)
            st.caption(f"mode {result.mode} | {format_location(result)} | score {result.score:.2f} | {result.match_type}")
            st.write(result.snippet)
            st.code(result.source_path, language="text")


if __name__ == "__main__":
    main()
