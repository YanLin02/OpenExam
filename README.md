# OpenExam

OpenExam is a local-first document retrieval application for private document collections. It indexes files on your machine and returns source-backed results with file names, locations, snippets, relevance scores, and optional local-model assistance.

OpenExam is designed for offline use against an existing local index. Search does not use cloud APIs.

## Features

- Local indexing for PDF, TXT, MD, DOCX, and PPTX files.
- Source-backed results with file path, file name, page/slide/paragraph location, snippet, and relevance score.
- Keyword search using SQLite FTS5.
- Fuzzy search using rapidfuzz.
- Retrieval modes: `keyword`, `fuzzy`, `semantic`, and `hybrid`.
- Optional local semantic retrieval through Ollama embeddings.
- Optional local cited Q&A through Ollama chat models.
- Streamlit UI with explicit search controls, PDF page preview, and local file actions.
- Offline runtime against a previously built local index.

## Supported File Types

- PDF: text extraction with PyMuPDF; page numbers are preserved.
- TXT and MD: paragraph-like positions are preserved.
- DOCX: paragraph positions are preserved.
- PPTX: slide numbers are preserved.

Scanned PDFs without embedded text are not OCRed. OCR is not implemented.

## Installation

Use Python 3.11 or newer.

```bash
git clone <your-repo-url> OpenExam
cd OpenExam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install dependencies before you need offline access. Do not commit `.openexam/`; it contains local indexes and generated artifacts.

## Quick Start

Build or rebuild the local index:

```bash
python -m openexam ingest "/path/to/documents" --rebuild
```

Check index status:

```bash
python -m openexam status
```

Run a hybrid search:

```bash
python -m openexam search "your query" --mode hybrid --top-k 5
```

Start the local UI:

```bash
streamlit run openexam/app.py --server.address 127.0.0.1
```

## CLI Usage

### Ingest

Index a directory or file:

```bash
python -m openexam ingest "./docs"
```

Rebuild the index from scratch:

```bash
python -m openexam ingest "./docs" --rebuild
```

The default index is stored under:

```text
.openexam/index.sqlite3
```

### Status

```bash
python -m openexam status
```

### Search

```bash
python -m openexam search "attention mechanism" --mode hybrid --top-k 5
```

Search modes:

- `keyword`: SQLite FTS5 plus substring fallback.
- `fuzzy`: rapidfuzz-based fuzzy matching.
- `semantic`: local vector search over Ollama embeddings.
- `hybrid`: combines available keyword, substring, fuzzy, and semantic signals.

Useful options:

- `--scope all|lecture|textbook_ocr|other`: restrict results by inferred source type.
- `--prefer none|lecture|textbook_ocr`: lightly boost a source type without filtering.
- `--per-file-cap N`: cap repeated results from the same file.
- `--open-first`: open the top result file with the operating system.
- `--auto-start-ollama` / `--no-auto-start-ollama`: control whether OpenExam may try to start a local Ollama server.

### Embed

Build local semantic embeddings after ingesting documents:

```bash
python -m openexam embed
```

### Ask

Ask a local cited question using retrieved chunks:

```bash
python -m openexam ask "summarize this concept" --mode hybrid --top-k 6 --evidence-policy warn
```

## Semantic Search

Semantic search is optional. It uses a local Ollama embedding model, with `bge-m3` as the default model.

OpenExam does not download or pull models automatically. Pull required models yourself while online:

```bash
ollama pull bge-m3
```

Start Ollama locally:

```bash
ollama serve
```

Build embeddings:

```bash
python -m openexam embed
```

Embedding artifacts are saved locally:

```text
.openexam/embeddings_bge-m3.npy
.openexam/embeddings_bge-m3.json
```

Run semantic search:

```bash
python -m openexam search "your query" --mode semantic --top-k 5
```

If the semantic index is missing or stale, `hybrid` mode falls back to keyword and fuzzy signals.

## Local Cited Q&A

OpenExam can use a local Ollama chat model to answer from retrieved chunks. The model does not read files directly; it only receives the retrieved snippets.

`qwen3:8b` is the default example model. Pull it yourself while online:

```bash
ollama pull qwen3:8b
```

Run a cited local answer:

```bash
python -m openexam ask "summarize this concept" --mode hybrid --top-k 6 --evidence-policy warn --detail standard
```

Evidence policy:

- `strict`: refuse when local evidence is missing or insufficient.
- `warn`: answer with an explicit warning when evidence is partial or missing.
- `open`: allow a general explanation when no local evidence is found, while clearly marking that there is no local source.

Detail level:

- `concise`: shorter answer.
- `standard`: balanced default.
- `detailed`: longer explanation with citations when available.

Outputs include the answer, evidence snippets, and source list with file name, location, source type, and path.

## Streamlit UI

Start the UI locally:

```bash
streamlit run openexam/app.py --server.address 127.0.0.1
```

The UI provides:

- Search and Ask modes with explicit `Search` and `Clear` actions.
- Compact retrieval controls for mode, scope, top-k, source preference, and per-file caps.
- Sidebar controls for indexing, index status, semantic status, Ollama status, and parameter help.
- PDF page preview inside the app for results with page numbers.
- Local file open and Finder reveal actions.
- Cached Search and Ask results so file actions do not rerun retrieval or local LLM generation.
- Timing information for retrieval and local generation.

Browser `file://` links are not used as the primary open mechanism because browsers may block local-file navigation from a localhost page. PDF page preview inside Streamlit is the most reliable way to inspect the referenced page.

### 并行搜索 / 后台提问

The Streamlit UI also includes `Parallel Search` and `Parallel Ask` actions:

- `Parallel Search` accepts one query per line and is useful for checking several terms at once.
- `Parallel Ask` accepts one question per line and runs questions in the background.
- Ask tasks use one worker by default so a local LLM such as `qwen3:8b` does not compete with itself for resources.
- If the machine has enough CPU/GPU memory, set `Ask workers` to `2` before submitting new ask tasks.
- Tasks are kept only in the current Streamlit session and are not persisted to disk or the index database.

## Privacy and Offline Use

- OpenExam stores indexes and generated local artifacts under `.openexam/`.
- `.openexam/` should not be committed to version control.
- OpenExam does not use cloud APIs for search.
- Ollama integration is local-only and uses the configured local Ollama server.
- Ollama models must be installed by the user beforehand; OpenExam does not pull models automatically.
- After dependencies, models, and indexes are prepared, search can run offline against the existing local index.

## Limitations

- OCR is not implemented. Scanned PDFs without embedded text are not searchable.
- Search quality depends on the quality of extracted text.
- Semantic search requires a local Ollama embedding model and a built semantic index.
- Local cited Q&A requires a local Ollama chat model.
- External PDF page jumping is not guaranteed across PDF readers or browsers.
- Source type classification is filename-based and may need adjustment for a specific collection.
- OpenExam does not include FAISS, Chroma, cloud APIs, or automatic model downloads.

## Development

Run tests:

```bash
python -m pytest
```
