# OpenExam

OpenExam is a reusable offline open-book document search and local Q&A tool. It indexes private files on your machine, returns source-backed search results, and can ask a local Ollama model to answer from retrieved snippets.

Search runs against a local SQLite index. Semantic retrieval and local answers use your local Ollama server; OpenExam does not call cloud APIs.

## Features

- Index PDF, TXT, MD, DOCX, and PPTX files.
- Search with `keyword`, `fuzzy`, `hybrid`, or `semantic` modes.
- Build local semantic embeddings with Ollama, using `bge-m3` by default.
- Ask local AI with cited evidence from retrieved chunks, using `qwen3:8b` by default.
- Use a priority answer bank for curated Markdown, TXT, PDF, DOCX, or PPTX sources.
- Run a Streamlit UI with Search and Ask local AI modes.
- Inspect PDF page previews inside the UI when page numbers are available.
- Open local files or reveal them in Finder on macOS.
- Check index, semantic, Ollama, and priority answer bank status.

## Supported Files

- PDF: extracted with PyMuPDF; page numbers are preserved.
- TXT and MD: paragraph positions are preserved.
- DOCX: paragraph positions are preserved.
- PPTX: slide numbers are preserved.

Scanned PDFs without embedded text are not OCRed. Malformed or encrypted PDFs may fail preview; use Open File or Finder as the fallback.

## Installation

Use Python 3.11 or newer.

```bash
git clone <your-repo-url> OpenExam
cd OpenExam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

OpenExam stores indexes and local artifacts under `.openexam/`. Do not commit that directory.

## Quick Start

Build or rebuild the local index:

```bash
python3 -m openexam ingest "/path/to/documents" --rebuild
```

Check status:

```bash
python3 -m openexam status
```

Run hybrid search:

```bash
python3 -m openexam search "your query" --mode hybrid --top-k 5
```

Start the UI:

```bash
python3 -m openexam ui
```

The UI binds to `127.0.0.1` by default:

```text
http://127.0.0.1:8501
```

## CLI

Available commands:

```text
ingest
search
ask
embed
status
ui
```

### Ingest

```bash
python3 -m openexam ingest "./docs"
python3 -m openexam ingest "./docs" --rebuild
```

The default database is:

```text
.openexam/index.sqlite3
```

### Search

```bash
python3 -m openexam search "attention mechanism" --mode hybrid --top-k 5
```

Useful options:

- `--mode keyword|fuzzy|hybrid|semantic`: choose the retrieval strategy.
- `--scope all|lecture|textbook_ocr|other`: restrict by inferred source type.
- `--prefer none|lecture|textbook_ocr`: lightly boost a source type without filtering.
- `--per-file-cap N`: limit repeated results from the same file.
- `--priority-answer-bank` / `--no-priority-answer-bank`: prioritize detected answer bank sources.
- `--open-first`: open the top result file with the operating system.
- `--auto-start-ollama` / `--no-auto-start-ollama`: control local Ollama startup for semantic search.

### Embed

Semantic search is optional and uses local Ollama embeddings.

OpenExam does not pull models automatically. Prepare the embedding model while online:

```bash
ollama pull bge-m3
```

Start Ollama:

```bash
ollama serve
```

The first Ollama startup can be slow. If OpenExam reports that Ollama did not become reachable after waiting, start it manually with `ollama serve` in a separate terminal and retry.

Build embeddings:

```bash
python3 -m openexam embed
```

Embedding artifacts are saved locally:

```text
.openexam/embeddings_bge-m3.npy
.openexam/embeddings_bge-m3.json
```

### Ask Local AI

Ask uses retrieval first, then sends only the retrieved snippets to a local Ollama chat model.

Prepare the default chat model while online:

```bash
ollama pull qwen3:8b
```

Run a cited local answer:

```bash
python3 -m openexam ask "summarize this concept" --mode hybrid --top-k 6 --evidence-policy warn
```

Useful options:

- `--evidence-policy strict|warn|open`: choose how missing or partial local evidence is handled.
- `--detail concise|standard|detailed`: control answer length.
- `--llm-model MODEL`: select a local Ollama chat model.
- `--priority-answer-bank` / `--no-priority-answer-bank`: prioritize detected answer bank sources during retrieval.

If local evidence is insufficient under strict mode, OpenExam uses:

```text
本地资料中未找到充分依据。
```

### Status

```bash
python3 -m openexam status
```

Status reports indexed documents, failed files, chunks, semantic index state, and detected priority answer bank files with labels.

## Priority Answer Bank

The priority answer bank is for curated sources that should be shown before regular search hits when `--priority-answer-bank` or the UI checkbox is enabled.

Directory names detected by default:

```text
answer_bank
exam_answer_bank
priority_sources
易考
重点
答案库
```

`DEFAULT_PRIORITY_PATTERNS` is intentionally empty. OpenExam does not ship built-in filename patterns. To add filename patterns or custom directories, create:

```text
.openexam/priority_sources.json
```

Example:

```json
{
  "answer_bank_dirs": ["my_answer_bank"],
  "priority_patterns": ["curated_notes"],
  "labels": {
    "my_answer_bank": "personal_bank",
    "curated_notes": "curated_bank"
  }
}
```

The repository includes reusable examples:

- [examples/answer_bank/concepts.md](examples/answer_bank/concepts.md)
- [examples/priority_sources.json](examples/priority_sources.json)

Recommended layout:

```text
data/
  answer_bank/
    concepts.md
  notes/
    chapter-01.pdf
```

For Markdown or TXT answer bank files, write each curated entry as a `###` section:

```markdown
### What is dropout?

Dropout randomly disables activations during training.

### What is attention?

Attention weights relationships between tokens.
```

Then rebuild the index:

```bash
python3 -m openexam ingest data --rebuild
```

For files detected as answer bank sources, OpenExam groups each `###` heading with its following body as one searchable section. Regular Markdown and TXT files keep the normal paragraph extraction behavior.

## Streamlit UI

Start the UI:

```bash
python3 -m openexam ui
```

The UI includes:

- `Search`
- `Ask local AI`

The sidebar includes:

- Build/update index and rebuild index actions.
- Index and semantic status, including a `重建语义索引` button.
- Detected priority answer bank files and labels.
- Ollama status, Start, Stop, Refresh, and local model list.

Search and Ask local AI both use explicit submit buttons. Result cards can be minimized, expanded, or closed. Closing a running Ask card hides it from the UI; it does not force-stop the local LLM request.

The Ollama sidebar `Start` button attempts to run `ollama serve` and records the process id under `.openexam/ollama.pid`. The `Stop` button only stops an Ollama server that OpenExam started.

## macOS Launcher

After installation, start the package CLI:

```bash
openexam ui
```

Or make the bundled command file executable:

```bash
chmod +x scripts/run_openexam.command
```

Then double-click:

```text
scripts/run_openexam.command
```

## Privacy

- Search uses a local SQLite index.
- Semantic embeddings and Ask local AI use the configured local Ollama server.
- OpenExam does not upload documents or snippets to cloud APIs.
- Ollama models must be installed by the user; OpenExam does not download models automatically.
- After dependencies, models, and indexes are prepared, search can run offline against the existing local index.

## Limitations

- OCR is not implemented.
- Search quality depends on extracted text quality.
- Semantic search requires a local Ollama embedding model and a built semantic index.
- Ask local AI requires a local Ollama chat model.
- PDF page preview can fail for malformed or encrypted PDFs.
- Source type classification is filename-based.
- OpenExam does not include FAISS, Chroma, cloud APIs, or automatic model downloads.

## Development

Run tests:

```bash
python3 -m pytest
```
