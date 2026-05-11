# OpenExam Solve Gap Report

This report tracks the current exam-solve coverage and the known gaps exposed by `tests/fixtures/exam_questions.json`.

## Supported Question Types

OpenExam `solve` currently routes these problem types:

- `calculation`: deterministic calculators first; no LLM is allowed to change numeric results.
- `design`: structured design templates, then local Ask/LLM for prose.
- `derivation`: structured derivation templates, then local Ask/LLM for prose.
- `compare`: structured comparison templates, then local Ask/LLM for prose.
- `concept` / `short_answer`: local Ask fallback with evidence policy, with automatic exam answer bank priority retrieval.

## Supported Calculators

Current deterministic calculator coverage:

- CNN and pooling output size.
- CNN output shape for explicit `CHW`, `NCHW`, and `NHWC` layouts.
- Conv2D parameter count.
- Linear layer parameter count.
- Structured multi-layer Conv/FC and MLP total parameter counts.
- Softmax and cross entropy from logits.
- MSE.
- Accuracy, precision, recall, and F1 from TP/FP/TN/FN.
- One-step gradient descent update.
- Simple RNN parameter count.
- LSTM parameter count.
- Transformer QKV parameter count.
- Multi-head attention parameter count.

## Parser Coverage

The parser intentionally covers common structured exam formats, not arbitrary prose. Stable examples include:

- `输入 32x32，卷积核 5x5，stride=1，padding=0`.
- `输入尺寸为 3x32x32，卷积核 3x3，输出通道 64`.
- `NCHW=8x3x32x32` and `NHWC=8x32x32x3`.
- `batch=16, channels=3, height=224, width=224`.
- `输入为 28×28，卷积核 5×5，步长 1，无填充`.
- `卷积层输入通道 3，输出通道 64，卷积核 3x3`.
- `全连接层输入 784，输出 10`.
- `Conv1: ...; Conv2: ...; FC: ... 求总参数量`.
- `MLP 结构为 784-128-64-10`.
- `logits=[2,1,0]，真实类别为 0`.
- `y_true=[...], y_pred=[...]`.
- `TP=80, FP=10, TN=90, FN=20`.
- `w=2, grad=0.5, lr=0.1`.
- `input_size=10, hidden_size=20, output_size=5`.
- `d_model=512`.

## Unstable Or Unsupported Inputs

These formats are not yet reliably parsed:

- Multi-layer network parameter totals described only in loose prose.
- Non-square tensors written without clear labels.
- Optimizer updates beyond basic scalar gradient descent.
- Confusion matrices written as a 2x2 matrix without TP/FP/TN/FN labels.
- Multi-class macro/micro precision, recall, and F1.
- Attention parameter questions with separate `d_k`, `d_v`, or per-head dimensions.
- Loss questions that require symbolic gradients rather than numeric MSE or cross entropy.

When parsing is unreliable, `solve` should return a manual-input fallback instead of guessing.

## Next Calculators And Parsers

Recommended next additions:

- Conv + pool shape tracing across multiple spatial layers.
- Multi-layer CNN/MLP parameter count parsing from less structured prose.
- Confusion-matrix parser for 2x2 table formats.
- Macro/micro/weighted classification metrics.
- Adam and momentum SGD one-step update calculators.
- Attention parameter parser with `num_heads`, `d_k`, `d_v`, and output projection variants.
- Numeric binary cross entropy and negative log likelihood helpers.

## Template Limitations

Design, derivation, and compare templates improve structure but still rely on local Ask/LLM for prose. They do not verify factual completeness beyond retrieved local context and evidence-policy behavior.

- Design templates may need manual adaptation for unusual constraints, such as latency, memory, or deployment requirements.
- Derivation templates provide visible exam steps but do not perform symbolic algebra checking.
- Compare templates enforce dimensions but cannot guarantee all course-specific emphasis is covered unless the local corpus contains it.

## Exam Answer Bank Priority

Concept and short-answer solve requests can prioritize indexed answer-bank directories. The default rules are path/directory based: `answer_bank`, `exam_answer_bank`, `priority_sources`, `易考`, `重点`, and `答案库`.

The strategy does not require a database schema change. It only reorders retrieved chunks before prompt construction, preserving source citations and original scores. It requires the files to be present in a recognized answer-bank directory and ingested. It is not used to compute or override `calculation` numeric answers.

Markdown exam answer banks are section-aware during extraction: `###` question headings are grouped with following answer paragraphs before normal chunking. This requires re-ingesting the files. If priority results still contain only question headings, run `python3 -m openexam ingest /path/to/materials --rebuild` first.

Built-in legacy filename patterns have been removed. Files such as old course answer sheets or glossary PDFs are not prioritized from the data root by filename alone. Move them under `answer_bank/` or configure `priority_patterns` in `.openexam/priority_sources.json`; a broken config falls back to directory defaults and surfaces a warning in status/UI.

## Recommended Exam Usage

- Use directly for parseable `calculation` questions. Numeric results come from deterministic calculators.
- For complex `calculation`, rewrite the problem with explicit structured parameters or pass `--problem-type calculation`.
- For `concept` and `short_answer`, use the default solve auto priority or pass `--priority-answer-bank`.
- For `design`, `derivation`, and `compare`, use `solve` with `--evidence-policy warn` so weak local evidence remains visible.
- Treat generated prose as an exam-answer draft and adjust it to the exact wording and marking rubric.

## Regression Tests

The regression fixture is:

```text
tests/fixtures/exam_questions.json
```

Run all tests:

```bash
python3 -m pytest
```

Run only the exam regression suite:

```bash
python3 -m pytest tests/test_exam_regression.py
```

To add a new exam question, append one fixture entry with stable `expected_contains` keywords. If the test fails, decide whether the failure is a classification rule gap, parser/calculator gap, template subtype gap, or expected fixture wording issue.
