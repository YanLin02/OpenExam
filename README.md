# OpenExam

OpenExam is an offline local search tool for open-book exams. It indexes local course files and returns source-backed results: file name, path, page/slide/paragraph location, snippet, and relevance score.

It does not use cloud APIs, online model services, OCR, FAISS, Chroma, or LLM answer generation. Optional semantic search uses a local Ollama embedding model only.

## Supported Files

- PDF, using PyMuPDF. Page numbers are preserved.
- TXT and MD. Paragraph positions are preserved.
- DOCX, using python-docx. Paragraph positions are preserved.
- PPTX, using python-pptx. Slide numbers are preserved.

Scanned PDFs with no embedded text are recorded as failed files instead of crashing the indexer. OCR is intentionally not implemented in the MVP.

## Install

Use Python 3.11 or newer.

```bash
cd /Users/lin/Documents/Code/Homework/OpenExam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install dependencies before the exam while network access is available. Runtime search over an existing index works offline.

## Index Course Materials

Create or update the local SQLite index:

```bash
python -m openexam ingest "/Users/lin/Documents/Code/Homework/DeepLearning/data"
```

Clear the old index and rebuild from scratch:

```bash
python -m openexam ingest "/Users/lin/Documents/Code/Homework/DeepLearning/data" --rebuild
```

The index is stored locally at:

```text
.openexam/index.sqlite3
```

Show index status and recent failed files:

```bash
python -m openexam status
```

## Search From CLI

```bash
python -m openexam search "Transformer" --top-k 5 --mode hybrid
python -m openexam search "正则化 优化" --top-k 5 --mode hybrid
python -m openexam search "生成对抗网络" --top-k 5 --mode hybrid
```

Search modes:

- `keyword`: SQLite FTS5 plus local substring fallback.
- `fuzzy`: rapidfuzz partial-ratio fuzzy matching.
- `semantic`: local cosine-similarity search over Ollama embeddings.
- `hybrid`: default mode, combining FTS5, substring, fuzzy, and semantic scores when a semantic index exists.

Results include mode, score, file name, page/slide/paragraph, snippet, and full path. Chinese snippets are centered around local substring matches where possible.

Result control options:

- `--scope all|lecture|textbook_ocr|other`: hard-filter results by source type. Default: `all`.
- `--prefer none|lecture|textbook_ocr`: lightly boost a source type without filtering. Default: `none`.
- `--per-file-cap N`: limit repeated results from the same file. `0` disables the cap.

Source types are inferred from file names:

- `lecture`: files such as `Chapter...`, `Course Overview...`, `附录...`, or courseware-like PDFs.
- `textbook_ocr`: files whose names contain `OCR` or `layered`.
- `other`: files that do not match the above.

Recommended exam searches:

```bash
# Prefer course slides / lecture PDFs.
python -m openexam search "Transformer 中注意力机制的作用" --mode hybrid --scope lecture --top-k 5

# Prefer full OCR textbook explanations.
python -m openexam search "为什么正则化可以缓解过拟合" --mode semantic --scope textbook_ocr --top-k 5

# Balanced results, with course slides lightly preferred and no single file dominating.
python -m openexam search "生成对抗网络的训练目标" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 5
```

## Semantic Search

Semantic search is optional and local. It uses Ollama at `http://127.0.0.1:11434` with the default embedding model `bge-m3`.

OpenExam never pulls models automatically. Before the exam, while online, install Ollama and pull the model yourself:

```bash
ollama pull bge-m3
```

When using semantic search, start Ollama locally:

```bash
ollama serve
```

Build embeddings after ingest:

```bash
python -m openexam embed
```

Embedding files are saved locally:

```text
.openexam/embeddings_bge-m3.npy
.openexam/embeddings_bge-m3.json
```

Run semantic search:

```bash
python -m openexam search "Transformer 中注意力机制的作用" --top-k 5 --mode semantic
```

Notes:

- If Ollama is not running, `embed` tells you to run `ollama serve`.
- If `bge-m3` is missing, `embed` tells you to run `ollama pull bge-m3` while online.
- If chunks, text hashes, or embedding model change, rerun `python -m openexam embed`.
- `hybrid` falls back to keyword/fuzzy search if the semantic index is missing or stale.

## Streamlit UI

Start the local UI on localhost:

```bash
streamlit run openexam/app.py --server.address 127.0.0.1
```

Then enter the material directory, build or rebuild the index, and search with a top-k value.

## 考试前检查清单

Use this checklist before the exam, while network access is still available:

1. Confirm Python 3.11+ is available:

   ```bash
   python3 --version
   ```

2. Install dependencies in the project environment:

   ```bash
   cd /Users/lin/Documents/Code/Homework/OpenExam
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]"
   ```

3. Build or rebuild the index for all exam materials:

   ```bash
   python -m openexam ingest "/Users/lin/Documents/Code/Homework/DeepLearning/data" --rebuild
   ```

4. Confirm the index contains documents and chunks:

   ```bash
   python -m openexam status
   ```

5. Test representative searches:

   ```bash
   python -m openexam search "Transformer" --top-k 3 --mode hybrid
   python -m openexam search "正则化 优化" --top-k 3 --mode hybrid
   python -m openexam search "生成对抗网络" --top-k 3 --mode hybrid
   ```

6. If semantic search is needed, prepare the local embedding index:

   ```bash
   ollama pull bge-m3
   ollama serve
   python -m openexam embed
   python -m openexam search "Transformer 中注意力机制的作用" --top-k 3 --mode semantic
   ```

7. Start the local UI once and confirm it loads:

   ```bash
   streamlit run openexam/app.py --server.address 127.0.0.1
   ```

8. Disconnect from the network and repeat one CLI search against the existing index.

## Offline Usage

Before disconnecting:

1. Install Python dependencies.
2. Run indexing for all exam materials.
3. Run `python -m openexam status` and confirm documents/chunks are present.
4. Test several expected queries.

After disconnecting:

1. Do not reinstall dependencies.
2. Run CLI searches against the existing `.openexam/index.sqlite3`.
3. Or start Streamlit with `--server.address 127.0.0.1`.

OpenExam does not make network requests during indexing or searching.

## 真实 DeepLearning PDF 测试命令

These commands use the real local DeepLearning PDF directory:

```bash
cd /Users/lin/Documents/Code/Homework/OpenExam

python3 -m openexam ingest "/Users/lin/Documents/Code/Homework/DeepLearning/data" --rebuild
python3 -m openexam embed
python3 -m openexam search "Transformer" --top-k 5 --mode hybrid
python3 -m openexam search "正则化 优化" --top-k 5 --mode hybrid
python3 -m openexam search "生成对抗网络" --top-k 5 --mode hybrid
python3 -m openexam search "卷积神经网络" --top-k 5 --mode hybrid
python3 -m openexam search "Transformer 中注意力机制的作用" --top-k 5 --mode semantic
python3 -m openexam search "Transformer 中注意力机制的作用" --mode hybrid --scope lecture --top-k 5
python3 -m openexam search "为什么正则化可以缓解过拟合" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 5
python3 -m pytest
streamlit run openexam/app.py --server.address 127.0.0.1
```

Expected behavior:

- The ingest command should index the PDFs without crashing.
- Search results should show mode, score, file name, page number, snippet, and path.
- Scanned or empty PDFs, if any, should be reported as warnings or failed files instead of stopping the run.

## Known Limitations

- OCR is not implemented. Scanned PDFs without embedded text cannot be searched.
- Chinese search uses SQLite FTS5 plus substring and rapidfuzz fallback; it is not a full Chinese word-segmentation engine.
- Semantic search requires a local Ollama service and a pre-pulled `bge-m3` model.
- The MVP does not include FAISS, Chroma, local LLM Q&A, cloud APIs, or automatic model downloads.
- Search quality depends on the quality of text extracted from the source file.

## Development Tests

```bash
python -m pytest
```

The tests create temporary TXT, MD, DOCX, PPTX, and PDF fixtures locally where the related parser dependencies are installed.
