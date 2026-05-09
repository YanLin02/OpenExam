# OpenExam

OpenExam is an offline local search tool for open-book exams. It indexes local course files and returns source-backed results: file name, path, page/slide/paragraph location, snippet, and relevance score.

It does not use cloud APIs, online model services, OCR, FAISS, or Chroma. Optional semantic search and cited Q&A use local Ollama models only.

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

Results include mode, score, file name, page/slide/paragraph, snippet, and full path. Chinese snippets are centered around local substring matches where possible. CLI search also prints `total_time_ms`, `retrieval_time_ms`, `semantic_time_ms`, and `ranking_time_ms`.

Result control options:

- `--scope all|lecture|textbook_ocr|other`: hard-filter results by source type. Default: `all`.
- `--prefer none|lecture|textbook_ocr`: lightly boost a source type without filtering. Default: `none`.
- `--per-file-cap N`: limit repeated results from the same file. `0` disables the cap.
- `--open-first`: open the top result file with macOS `open`.
- `--auto-start-ollama` / `--no-auto-start-ollama`: control whether semantic search tries to start local Ollama with `ollama serve`. Auto-start is enabled by default.

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

When using semantic search, start Ollama locally, or let OpenExam try to start it with `ollama serve`:

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

- If Ollama is not running, `embed`, `semantic`, and `ask` can try to start it with `ollama serve`.
- If `bge-m3` is missing, `embed` tells you to run `ollama pull bge-m3` while online.
- If chunks, text hashes, or embedding model change, rerun `python -m openexam embed`.
- `hybrid` falls back to keyword/fuzzy search if the semantic index is missing or stale.

## 搜索参数怎么选

- `mode`
  - `keyword`: 关键词全文搜索，适合查精确术语。
  - `fuzzy`: 模糊搜索，适合拼写不确定或中文短词。
  - `semantic`: 语义搜索，适合用自然语言描述问题。
  - `hybrid`: 混合搜索，默认推荐。
- `scope`
  - `all`: 搜索全部资料。
  - `lecture`: 只搜索课件。
  - `textbook_ocr`: 只搜索 OCR 教材。
  - `other`: 只搜索其他文件。
- `prefer`
  - `none`: 不偏向任何来源。
  - `lecture`: 轻微优先课件，但不硬过滤。
  - `textbook_ocr`: 轻微优先教材，但不硬过滤。
- `per-file-cap`: 限制同一文件最多出现几条结果，避免单个 PDF 霸榜。
- `evidence-policy`
  - `strict`: 证据不足就拒答。
  - `warn`: 证据不足也回答，但显式标注，考试推荐。
  - `open`: 无本地依据也回答，但标注无本地来源。
- `top-k`: 返回或提供给 LLM 的片段数量，越大越全面但越慢。
- `detail`
  - `concise`: 快速定位，回答控制在 3-5 句话。
  - `standard`: 默认推荐，适合考试现场使用。
  - `detailed`: 更详细解释，适合复习理解。

## Local Cited Q&A

OpenExam can ask a local Ollama LLM to answer using only retrieved chunks. It does not let the LLM read files directly, and it does not call cloud APIs.

Default LLM settings:

- provider: `ollama`
- model: `qwen3:8b`
- base URL: `http://127.0.0.1:11434`

Prepare the model while online:

```bash
ollama pull qwen3:8b
```

Run cited Q&A:

```bash
python -m openexam ask "Transformer 中注意力机制的作用" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 6 --evidence-policy warn --detail standard
```

`ask` first runs the normal search pipeline, then passes only the returned chunks to the local LLM. The output contains:

1. `回答`
2. `依据`
3. `来源`

Each source includes file name, page/slide/paragraph, source type, and full path. CLI ask also prints `retrieval_time_ms`, `prompt_build_time_ms`, `llm_time_ms`, and `total_time_ms`.

Evidence policy controls what happens when retrieved chunks do not fully cover the question:

- `--evidence-policy strict`: most conservative. If local evidence is missing or partial, return `本地资料中未找到充分依据。`
- `--evidence-policy warn`: recommended for exams. Continue answering, but explicitly label partial or missing local evidence.
- `--evidence-policy open`: answer even with no local results, while clearly marking that there is no local evidence and without inventing sources.

When evidence is partial, output includes `回答`, `资料依据状态`, `依据`, `来源`, and `补充说明`. When no local results exist, `依据` is `无本地依据` and `来源` is `无本地来源`.

Answer detail controls output length:

- `--detail concise`: short answer, useful for fast exam lookup; uses a smaller local generation budget.
- `--detail standard`: default and recommended for exams.
- `--detail detailed`: longer explanation for review and understanding; it still must cite local sources or mark insufficient evidence.

Useful options:

```bash
python -m openexam ask "为什么正则化可以缓解过拟合" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 6 --evidence-policy warn
python -m openexam ask "卷积神经网络的局部连接和权值共享是什么意思" --mode hybrid --scope lecture --top-k 6 --evidence-policy strict
python -m openexam ask "一个本地资料中不存在的随机问题" --evidence-policy open
python -m openexam ask "生成对抗网络的训练目标是什么" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 6 --evidence-policy warn --detail detailed
```

Error handling:

- If `--evidence-policy strict` is used and local evidence is missing or partial, OpenExam does not call the LLM and prints `本地资料中未找到充分依据。`
- With the default `--evidence-policy warn`, OpenExam may call the LLM with a clear warning when local evidence is partial or missing.
- If Ollama is not running, it prints `Ollama is not reachable. Start it with: ollama serve`.
- If `qwen3:8b` is missing, pull it while online: `ollama pull qwen3:8b`.
- If semantic index is missing and `ask --mode semantic` is requested, `ask` falls back to `hybrid`.
- `--auto-start-ollama` is enabled by default for ask; it only runs local `ollama serve` and never pulls models.

## Streamlit UI

Start the local UI on localhost:

```bash
streamlit run openexam/app.py --server.address 127.0.0.1
```

Then enter the material directory, build or rebuild the index, and search with a top-k value.

The UI includes:

- Search mode, scope, source preference, per-file cap, and top-k controls with Chinese parameter explanations.
- Evidence policy and detail selectors for Ask local AI.
- Ollama status panel, model list, refresh button, and start button.
- Local LLM model dropdown. It prefers `qwen3:8b`; if missing, it chooses the first non-embedding local model.
- Timing display for retrieval, semantic search, ranking, LLM generation, and total time.
- Open file buttons and local file URI display. For PDFs, OpenExam tries `file:///path/to/file.pdf#page=N`; PDF reader support for `#page` varies.
- Clear message when the index is missing or no result is found.

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

7. If local cited Q&A is needed, prepare and test the LLM:

   ```bash
   ollama pull qwen3:8b
   python -m openexam ask "Transformer 中注意力机制的作用" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 6 --evidence-policy warn
   ```

8. Start the local UI once and confirm it loads:

   ```bash
   streamlit run openexam/app.py --server.address 127.0.0.1
   ```

9. Disconnect from the network and repeat one CLI search or ask command against the existing index.

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
python3 -m openexam ask "Transformer 中注意力机制的作用" --mode hybrid --prefer lecture --per-file-cap 2 --top-k 6 --evidence-policy warn
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
- Local cited Q&A requires a local Ollama service and a pre-pulled `qwen3:8b` model.
- The MVP does not include FAISS, Chroma, OCR, cloud APIs, or automatic model downloads.
- Search quality depends on the quality of text extracted from the source file.
- LLM answers are constrained by retrieved chunks; if retrieval misses the relevant page, Q&A quality will suffer.

## Development Tests

```bash
python -m pytest
```

The tests create temporary TXT, MD, DOCX, PPTX, and PDF fixtures locally where the related parser dependencies are installed.
