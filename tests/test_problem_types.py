from __future__ import annotations

from openexam.problem_types import ProblemType, classify_problem


def test_classifies_calculation_questions() -> None:
    assert classify_problem("给定输入 32x32，卷积核 5x5，stride=1，padding=0，输出尺寸是多少？") == ProblemType.CALCULATION
    assert classify_problem("计算 accuracy、precision、recall 和 F1") == ProblemType.CALCULATION


def test_classifies_design_questions() -> None:
    assert classify_problem("设计一个 CNN 完成手写数字识别任务") == ProblemType.DESIGN
    assert classify_problem("给出网络结构和训练流程") == ProblemType.DESIGN


def test_classifies_derivation_questions() -> None:
    assert classify_problem("推导 softmax 交叉熵的梯度") == ProblemType.DERIVATION
    assert classify_problem("证明反向传播过程中的链式法则") == ProblemType.DERIVATION


def test_classifies_compare_questions() -> None:
    assert classify_problem("比较 CNN 和 Transformer 的优缺点") == ProblemType.COMPARE
    assert classify_problem("说明 accuracy 和 recall 的区别") == ProblemType.COMPARE


def test_classifies_concept_short_answer_and_unknown() -> None:
    assert classify_problem("什么是 dropout 的作用？") == ProblemType.CONCEPT
    assert classify_problem("请回答深度学习考试中的常见注意事项") == ProblemType.SHORT_ANSWER
    assert classify_problem("   ") == ProblemType.UNKNOWN
