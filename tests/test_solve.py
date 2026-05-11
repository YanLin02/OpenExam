from __future__ import annotations

from openexam.__main__ import build_parser, cmd_solve
from openexam.ask import AskResponse
from openexam.config import AppConfig
from openexam.models import SearchResult
from openexam.problem_types import ProblemType
from openexam.solve import SolveResponse, SolveSearchOptions, render_solve_response, solve_question


def make_result(index: int = 1) -> SearchResult:
    return SearchResult(
        chunk_db_id=index,
        document_id=10,
        file_name="cnn.md",
        source_path="/tmp/cnn.md",
        source_type="lecture",
        location_type="paragraph",
        location_label="paragraph 1",
        page_number=None,
        chunk_id=f"chunk-{index}",
        text="卷积神经网络可以用于手写数字识别。",
        snippet="卷积神经网络可以用于**手写数字识别**。",
        score=90.0,
        mode="hybrid",
    )


def make_ask_response(question: str) -> AskResponse:
    return AskResponse(
        question=question,
        answer="可以使用卷积层、池化层和全连接层构建 CNN。[1]",
        results=[make_result()],
        search_mode="hybrid",
        scope="all",
        prefer="lecture",
        per_file_cap=2,
        top_k=6,
        llm_model="qwen3:8b",
        llm_called=True,
        evidence_status="sufficient",
        evidence_policy="warn",
        missing_phrases=[],
        timing={"retrieval_time_ms": 1.0, "prompt_build_time_ms": 1.0, "llm_time_ms": 1.0, "total_time_ms": 3.0},
        detail="standard",
    )


def test_solve_question_classifies_and_calls_ask_fallback(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_ask_question(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return make_ask_response(question)

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)
    response = solve_question(
        "设计一个 CNN 完成手写数字识别任务",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        search_mode="keyword",
        scope="lecture",
        prefer="none",
        per_file_cap=1,
        top_k=3,
        llm_model="qwen3:8b",
        evidence_policy="warn",
        detail="concise",
    )

    assert response.problem_type == ProblemType.DESIGN
    assert "设计" in response.strategy or "任务目标" in response.strategy
    assert seen["question"] == "设计一个 CNN 完成手写数字识别任务"
    assert seen["mode"] == "keyword"
    assert seen["scope"] == "lecture"
    assert seen["prefer"] == "none"
    assert seen["per_file_cap"] == 1
    assert seen["top_k"] == 3
    assert seen["detail"] == "concise"


def test_solve_question_accepts_explicit_problem_type_and_search_options(monkeypatch, tmp_path) -> None:
    def fake_ask_question(question, **kwargs):
        assert kwargs["mode"] == "fuzzy"
        assert kwargs["scope"] == "textbook_ocr"
        assert kwargs["prefer"] == "textbook_ocr"
        assert kwargs["per_file_cap"] == 4
        assert kwargs["top_k"] == 5
        return make_ask_response(question)

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)
    response = solve_question(
        "写出公式",
        mode="derivation",
        search_options=SolveSearchOptions(
            search_mode="fuzzy",
            scope="textbook_ocr",
            prefer="textbook_ocr",
            per_file_cap=4,
            top_k=5,
        ),
        config=AppConfig(index_dir=tmp_path / ".openexam"),
    )

    assert response.problem_type == ProblemType.DERIVATION


def test_solve_question_uses_calculator_before_ask(monkeypatch, tmp_path) -> None:
    called = False

    def fake_ask_question(*args, **kwargs):
        nonlocal called
        called = True
        return make_ask_response(args[0])

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)

    response = solve_question(
        "给定输入 32x32，卷积核 5x5，stride=1，padding=0，输出尺寸是多少？",
        mode="calculation",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
    )
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.CALCULATION
    assert response.calculation_answer is not None
    assert response.calculation_answer.calculation_type == "conv2d_output_size"
    assert response.ask_response.llm_called is False
    assert not called
    assert "计算类型：\nconv2d_output_size" in rendered
    assert "输出尺寸为 28 x 28" in rendered
    assert "deterministic calculator" in rendered


def test_render_solve_response_contains_required_sections() -> None:
    response = SolveResponse(
        question="设计一个 CNN 完成手写数字识别任务",
        problem_type=ProblemType.DESIGN,
        strategy="先拆分任务目标。",
        ask_response=make_ask_response("设计一个 CNN 完成手写数字识别任务"),
        requested_mode="auto",
    )

    rendered = render_solve_response(response)

    assert "题型：\ndesign" in rendered
    assert "解题策略：" in rendered
    assert "答案：" in rendered
    assert "依据：" in rendered
    assert "来源：" in rendered
    assert "cnn.md" in rendered


def test_cli_solve_parser_accepts_requested_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "solve",
            "设计一个 CNN 完成手写数字识别任务",
            "--problem-type",
            "design",
            "--mode",
            "keyword",
            "--scope",
            "lecture",
            "--prefer",
            "none",
            "--per-file-cap",
            "1",
            "--top-k",
            "3",
            "--evidence-policy",
            "strict",
            "--detail",
            "detailed",
            "--llm-model",
            "qwen3:8b",
        ]
    )

    assert args.func is cmd_solve
    assert args.problem_type == "design"
    assert args.mode == "keyword"
    assert args.scope == "lecture"
    assert args.prefer == "none"
    assert args.per_file_cap == 1
    assert args.top_k == 3
    assert args.evidence_policy == "strict"
    assert args.detail == "detailed"
