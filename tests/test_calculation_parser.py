from __future__ import annotations

import pytest

from openexam.calculators.parser import detect_calculation_type, parse_calculation_question, solve_calculation_question


def test_parse_conv_output_english_tokens() -> None:
    question = "给定输入 32x32，卷积核 5x5，stride=1，padding=0，输出尺寸是多少？"

    parsed = parse_calculation_question(question)
    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "conv2d_output_size"
    assert not parsed.need_manual_input
    assert parsed.parameters["input_h"] == 32
    assert parsed.parameters["kernel_h"] == 5
    assert answer.values["output_h"] == 28
    assert answer.values["output_w"] == 28


def test_parse_conv_output_chinese_tokens() -> None:
    question = "输入为 28×28，卷积核 5×5，步长 1，无填充，输出尺寸是多少？"

    answer = solve_calculation_question(question)

    assert answer.calculation_type == "conv2d_output_size"
    assert answer.values["output_h"] == 24
    assert answer.values["output_w"] == 24


def test_parse_conv_param_count() -> None:
    question = "卷积层输入通道 3，输出通道 64，卷积核 3x3，参数量是多少？"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "conv2d_param_count"
    assert answer.values["params"] == 1792


def test_parse_linear_param_count() -> None:
    question = "全连接层输入 784，输出 10，参数量是多少？"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "linear_param_count"
    assert answer.values["params"] == 7850


def test_parse_logits_softmax_cross_entropy() -> None:
    question = "logits=[2,1,0]，真实类别为 0，求 softmax 和交叉熵"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "softmax_cross_entropy"
    assert sum(answer.values["probs"]) == pytest.approx(1.0)
    assert answer.values["cross_entropy"] == pytest.approx(0.4076059644)


def test_parse_classification_metrics() -> None:
    question = "TP=80, FP=10, TN=90, FN=20，求 accuracy precision recall F1"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "classification_metrics"
    assert answer.values["accuracy"] == pytest.approx(0.85)
    assert answer.values["precision"] == pytest.approx(80 / 90)
    assert answer.values["recall"] == pytest.approx(0.8)


def test_parse_gradient_descent_step() -> None:
    question = "w=2, grad=0.5, lr=0.1，梯度下降一步后的 w 是多少？"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "gradient_descent_step"
    assert answer.values["w_new"] == pytest.approx(1.95)


def test_unparseable_question_requires_manual_input() -> None:
    parsed = parse_calculation_question("计算这个模型的复杂度")

    assert parsed.need_manual_input
    assert parsed.calculation_type == "need_manual_input"
