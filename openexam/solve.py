from __future__ import annotations

import time
from dataclasses import dataclass

from openexam.ask import (
    AnswerDetail,
    AskResponse,
    EvidencePolicy,
    NO_EVIDENCE,
    NO_LOCAL_EVIDENCE_WARNING,
    PARTIAL_EVIDENCE_WARNING,
    ask_question,
    format_evidence,
    format_source,
)
from openexam.calculators.parser import CalculationAnswer, solve_calculation_question
from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.design_templates import DesignTaskType, build_design_answer_plan, build_design_prompt, classify_design_task
from openexam.problem_types import ProblemType, classify_problem
from openexam.search import SearchMode, search_index
from openexam.sources import SearchScope, SourcePreference


ProblemTypeMode = str | ProblemType


@dataclass(frozen=True)
class SolveSearchOptions:
    search_mode: SearchMode = "hybrid"
    scope: SearchScope = "all"
    prefer: SourcePreference = "lecture"
    per_file_cap: int = 2
    top_k: int | None = None
    llm_model: str | None = None
    auto_start_ollama: bool = True


@dataclass(frozen=True)
class SolveResponse:
    question: str
    problem_type: ProblemType
    strategy: str
    ask_response: AskResponse
    requested_mode: str
    calculation_answer: CalculationAnswer | None = None
    design_task_type: DesignTaskType | None = None
    design_sections: list[str] | None = None
    fallback_note: str | None = None


_SOLVE_STRATEGIES: dict[ProblemType, str] = {
    ProblemType.CONCEPT: "先定位本地资料中的定义、作用和关键特征，再用短答方式组织结论并保留引用。",
    ProblemType.CALCULATION: "先识别题目给出的已知量、目标量和适用公式；本轮先用本地资料与 Ask fallback 给出步骤化答案，后续计算器会接入这里。",
    ProblemType.DERIVATION: "先确认要推导的公式或过程，再按前提、关键变形和结论组织回答；本轮不展开符号计算器。",
    ProblemType.DESIGN: "先拆分任务目标、输入输出、模型/流程选择和评估方式，再给出可执行方案；本轮先使用本地资料生成基础设计答案。",
    ProblemType.COMPARE: "先列出比较维度，再分别说明相同点、差异点和适用场景，并用本地依据约束结论。",
    ProblemType.SHORT_ANSWER: "按考试短答题处理，优先给出直接结论，再补充必要解释和本地引用。",
    ProblemType.UNKNOWN: "题型不明确，先按短答题处理；如果本地依据不足，会按证据策略提示风险。",
}


def _coerce_problem_type(mode: ProblemTypeMode, question: str) -> ProblemType:
    if isinstance(mode, ProblemType):
        return mode
    if mode == "auto":
        return classify_problem(question)
    try:
        return ProblemType(mode)
    except ValueError as exc:
        choices = ", ".join(["auto", *(problem_type.value for problem_type in ProblemType)])
        raise ValueError(f"Unknown problem type mode: {mode}. Expected one of: {choices}") from exc


def solve_question(
    question: str,
    mode: ProblemTypeMode = "auto",
    search_options: SolveSearchOptions | None = None,
    *,
    config: AppConfig = DEFAULT_CONFIG,
    search_mode: SearchMode = "hybrid",
    scope: SearchScope = "all",
    prefer: SourcePreference = "lecture",
    per_file_cap: int = 2,
    top_k: int | None = None,
    llm_model: str | None = None,
    evidence_policy: EvidencePolicy = "warn",
    detail: AnswerDetail = "standard",
    auto_start_ollama: bool = True,
) -> SolveResponse:
    problem_type = _coerce_problem_type(mode, question)
    options = search_options or SolveSearchOptions(
        search_mode=search_mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=top_k,
        llm_model=llm_model,
        auto_start_ollama=auto_start_ollama,
    )
    if problem_type == ProblemType.CALCULATION:
        calculation_answer = solve_calculation_question(question)
        if not calculation_answer.need_manual_input:
            evidence_response = _calculation_evidence_response(
                question,
                calculation_answer,
                config=config,
                options=options,
                evidence_policy=evidence_policy,
                detail=detail,
            )
            return SolveResponse(
                question=question,
                problem_type=problem_type,
                strategy=_SOLVE_STRATEGIES[problem_type],
                ask_response=evidence_response,
                requested_mode=mode.value if isinstance(mode, ProblemType) else mode,
                calculation_answer=calculation_answer,
            )

        ask_response = ask_question(
            question,
            config=config,
            mode=options.search_mode,
            scope=options.scope,
            prefer=options.prefer,
            per_file_cap=options.per_file_cap,
            top_k=options.top_k,
            llm_model=options.llm_model,
            evidence_policy=evidence_policy,
            detail=detail,
            auto_start_ollama=options.auto_start_ollama,
        )
        return SolveResponse(
            question=question,
            problem_type=problem_type,
            strategy=_SOLVE_STRATEGIES[problem_type],
            ask_response=ask_response,
            requested_mode=mode.value if isinstance(mode, ProblemType) else mode,
            calculation_answer=calculation_answer,
            fallback_note="未能可靠解析题目参数，以下为基于本地资料和模型的解题说明。",
        )

    if problem_type == ProblemType.DESIGN:
        design_task_type = classify_design_task(question)
        design_plan = build_design_answer_plan(question, design_task_type)
        enhanced_question = build_design_prompt(
            question,
            design_plan,
            retrieved_context="OpenExam Ask 会在下一步检索本地片段并附带来源。",
            evidence_status="unknown",
            detail=detail,
        )
        ask_response = ask_question(
            enhanced_question,
            config=config,
            mode=options.search_mode,
            scope=options.scope,
            prefer=options.prefer,
            per_file_cap=options.per_file_cap,
            top_k=options.top_k,
            llm_model=options.llm_model,
            evidence_policy=evidence_policy,
            detail=detail,
            auto_start_ollama=options.auto_start_ollama,
        )
        return SolveResponse(
            question=question,
            problem_type=problem_type,
            strategy=design_plan.guidance,
            ask_response=ask_response,
            requested_mode=mode.value if isinstance(mode, ProblemType) else mode,
            design_task_type=design_task_type,
            design_sections=design_plan.sections,
        )

    ask_response = ask_question(
        question,
        config=config,
        mode=options.search_mode,
        scope=options.scope,
        prefer=options.prefer,
        per_file_cap=options.per_file_cap,
        top_k=options.top_k,
        llm_model=options.llm_model,
        evidence_policy=evidence_policy,
        detail=detail,
        auto_start_ollama=options.auto_start_ollama,
    )
    return SolveResponse(
        question=question,
        problem_type=problem_type,
        strategy=_SOLVE_STRATEGIES[problem_type],
        ask_response=ask_response,
        requested_mode=mode.value if isinstance(mode, ProblemType) else mode,
    )


def _calculation_evidence_response(
    question: str,
    calculation_answer: CalculationAnswer,
    *,
    config: AppConfig,
    options: SolveSearchOptions,
    evidence_policy: EvidencePolicy,
    detail: AnswerDetail,
) -> AskResponse:
    effective_top_k = options.top_k if options.top_k is not None else config.llm_context_top_k
    results = []
    search_mode = options.search_mode
    retrieval_time_ms = 0.0
    if config.db_path.exists():
        try:
            retrieval_start = time.perf_counter()
            results = search_index(
                question,
                top_k=effective_top_k,
                config=config,
                mode=options.search_mode,
                scope=options.scope,
                prefer=options.prefer,
                per_file_cap=options.per_file_cap,
            )
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
        except Exception:
            results = []
    evidence_status = "sufficient" if results else "none"
    return AskResponse(
        question=question,
        answer=calculation_answer.result_text,
        results=results,
        search_mode=search_mode,
        scope=options.scope,
        prefer=options.prefer,
        per_file_cap=options.per_file_cap,
        top_k=effective_top_k,
        llm_model=options.llm_model or config.llm_model,
        llm_called=False,
        evidence_status=evidence_status,
        evidence_policy=evidence_policy,
        missing_phrases=[],
        timing={
            "retrieval_time_ms": retrieval_time_ms,
            "prompt_build_time_ms": 0.0,
            "llm_time_ms": 0.0,
            "total_time_ms": retrieval_time_ms,
        },
        detail=detail,
    )


def _solve_answer_text(response: AskResponse) -> str:
    answer = response.answer.strip() or NO_EVIDENCE
    if answer == NO_EVIDENCE:
        return answer
    if response.evidence_status == "partial" and not answer.startswith(PARTIAL_EVIDENCE_WARNING):
        return f"{PARTIAL_EVIDENCE_WARNING}\n{answer}"
    if response.evidence_status == "none" and not answer.startswith(NO_LOCAL_EVIDENCE_WARNING):
        return f"{NO_LOCAL_EVIDENCE_WARNING}\n{answer}"
    return answer


def render_solve_response(response: SolveResponse) -> str:
    ask_response = response.ask_response
    evidence = "\n".join(format_evidence(result, index) for index, result in enumerate(ask_response.results, start=1))
    sources = "\n".join(format_source(result, index) for index, result in enumerate(ask_response.results, start=1))
    if not evidence:
        evidence = "无本地依据"
    if not sources:
        sources = "无本地来源"
    if response.calculation_answer is not None and not response.calculation_answer.need_manual_input:
        calculation = response.calculation_answer
        return (
            f"题型：\n{response.problem_type.value}\n\n"
            f"计算类型：\n{calculation.calculation_type}\n\n"
            f"解题策略：\n{response.strategy}\n\n"
            f"公式：\n{calculation.formula}\n\n"
            f"代入：\n{calculation.substitution}\n\n"
            f"结果：\n{calculation.result_text}\n\n"
            f"本地依据：\n{evidence}\n\n"
            f"来源：\n{sources}\n\n"
            "计算说明：\n数值结果来自 deterministic calculator；LLM 不参与也不能覆盖上述数值结果。"
        )
    if response.problem_type == ProblemType.DESIGN and response.design_task_type is not None:
        sections = "\n".join(f"{index}. {section}" for index, section in enumerate(response.design_sections or [], start=1))
        return (
            f"题型：\n{response.problem_type.value}\n\n"
            f"设计任务类型：\n{response.design_task_type.value}\n\n"
            f"解题策略：\n{response.strategy}\n\n"
            f"答题结构：\n{sections}\n\n"
            f"答案：\n{_solve_answer_text(ask_response)}\n\n"
            f"依据：\n{evidence}\n\n"
            f"来源：\n{sources}"
        )
    note = f"{response.fallback_note}\n\n" if response.fallback_note else ""
    return (
        f"题型：\n{response.problem_type.value}\n\n"
        f"解题策略：\n{response.strategy}\n\n"
        f"答案：\n{note}{_solve_answer_text(ask_response)}\n\n"
        f"依据：\n{evidence}\n\n"
        f"来源：\n{sources}"
    )
