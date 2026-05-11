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


def test_parse_chw_conv_shape() -> None:
    question = "输入尺寸为 3x32x32，卷积核 3x3，输出通道 64，stride=1，padding=1，输出尺寸是多少？"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "conv2d_output_shape"
    assert answer.values["output_shape"] == (64, 32, 32)
    assert "64 x 32 x 32" in answer.result_text


def test_parse_explicit_cxhxw_conv_shape() -> None:
    question = "输入为 CxHxW = 3x224x224，经过 7x7 卷积，stride=2，padding=3，输出通道 64，输出 shape 是多少？"

    answer = solve_calculation_question(question)

    assert answer.values["layout"] == "CHW"
    assert answer.values["output_shape"] == (64, 112, 112)


def test_parse_nchw_conv_shape() -> None:
    question = "输入 shape 为 NCHW=8x3x32x32，卷积核 3x3，输出通道 64，stride=1，padding=1，输出 shape 是多少？"

    answer = solve_calculation_question(question)

    assert answer.values["layout"] == "NCHW"
    assert answer.values["output_shape"] == (8, 64, 32, 32)


def test_parse_nhwc_conv_shape() -> None:
    question = "输入 shape 为 NHWC=8x32x32x3，卷积核 3x3，输出通道 64，stride=1，padding=1，输出 shape 是多少？"

    answer = solve_calculation_question(question)

    assert answer.values["layout"] == "NHWC"
    assert answer.values["output_shape"] == (8, 32, 32, 64)


def test_ambiguous_4d_conv_shape_requires_layout() -> None:
    question = "输入 shape 为 8x32x32x3，卷积核 3x3，输出通道 64，stride=1，padding=1，输出 shape 是多少？"

    parsed = parse_calculation_question(question)
    answer = solve_calculation_question(question)

    assert parsed.need_manual_input
    assert answer.need_manual_input
    assert "layout" in answer.reason


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


def test_parse_multi_layer_conv_fc_param_count() -> None:
    question = "网络包含 Conv1: 输入通道 3，输出通道 16，卷积核 3x3；Conv2: 输入通道 16，输出通道 32，卷积核 3x3；FC: 输入 800，输出 10，求总参数量"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "multi_layer_param_count"
    assert answer.values["total_params"] == 13098
    assert [layer["params"] for layer in answer.values["layers"]] == [448, 4640, 8010]


def test_parse_mlp_param_count() -> None:
    question = "MLP 结构为 784-128-64-10，求全连接参数量"

    answer = solve_calculation_question(question)

    assert detect_calculation_type(question) == "multi_layer_param_count"
    assert answer.values["total_params"] == 109386


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
