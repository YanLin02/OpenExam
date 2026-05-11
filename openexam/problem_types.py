from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    CONCEPT = "concept"
    CALCULATION = "calculation"
    DERIVATION = "derivation"
    DESIGN = "design"
    COMPARE = "compare"
    SHORT_ANSWER = "short_answer"
    UNKNOWN = "unknown"


_KEYWORDS: dict[ProblemType, tuple[tuple[str, int], ...]] = {
    ProblemType.CALCULATION: (
        ("输出尺寸", 3),
        ("参数量", 3),
        ("交叉熵", 2),
        ("softmax", 2),
        ("梯度下降", 2),
        ("accuracy", 2),
        ("precision", 2),
        ("recall", 2),
        ("f1", 2),
        ("维度", 2),
        ("多少", 2),
        ("计算", 2),
        ("求", 1),
    ),
    ProblemType.DERIVATION: (
        ("反向传播过程", 5),
        ("写出公式", 5),
        ("推导", 5),
        ("证明", 5),
        ("反向传播", 3),
        ("公式", 1),
    ),
    ProblemType.DESIGN: (
        ("给出网络结构", 3),
        ("网络结构", 2),
        ("设计", 2),
        ("构建", 2),
        ("方案", 2),
        ("实验", 2),
        ("流程", 2),
        ("模型", 1),
    ),
    ProblemType.COMPARE: (
        ("优缺点", 5),
        ("比较", 5),
        ("区别", 5),
        ("异同", 5),
        ("不同", 1),
    ),
}

_CONCEPT_KEYWORDS = (
    "是什么",
    "什么是",
    "名词解释",
    "定义",
    "概念",
    "作用",
    "原理",
    "解释",
    "说明",
    "简述",
)

_TIE_PRIORITY = (
    ProblemType.DERIVATION,
    ProblemType.COMPARE,
    ProblemType.DESIGN,
    ProblemType.CALCULATION,
)


def classify_problem(text: str) -> ProblemType:
    normalized = text.strip().lower()
    if not normalized:
        return ProblemType.UNKNOWN

    scores: dict[ProblemType, int] = {}
    for problem_type, weighted_keywords in _KEYWORDS.items():
        score = sum(weight for keyword, weight in weighted_keywords if keyword.lower() in normalized)
        scores[problem_type] = score

    best_score = max(scores.values(), default=0)
    if best_score > 0:
        winners = {problem_type for problem_type, score in scores.items() if score == best_score}
        for problem_type in _TIE_PRIORITY:
            if problem_type in winners:
                return problem_type

    if any(keyword in normalized for keyword in _CONCEPT_KEYWORDS):
        return ProblemType.CONCEPT
    return ProblemType.SHORT_ANSWER
