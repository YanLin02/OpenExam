from __future__ import annotations

import math

import pytest

from openexam.calculators import (
    attention_qkv_param_count,
    classification_metrics,
    conv2d_output_shape,
    conv2d_output_size,
    conv2d_param_count,
    cross_entropy_from_logits,
    gradient_descent_step,
    linear_param_count,
    lstm_param_count,
    make_conv2d_layer_param_spec,
    make_linear_layer_param_spec,
    mse,
    multi_layer_param_count,
    multihead_attention_param_count,
    pool2d_output_size,
    simple_rnn_param_count,
    softmax,
)


def test_conv2d_output_size_examples() -> None:
    assert conv2d_output_size(32, 32, 5, 5, stride_h=1, stride_w=1, padding_h=0, padding_w=0)["output_h"] == 28
    assert conv2d_output_size(32, 32, 5, 5, stride_h=1, stride_w=1, padding_h=0, padding_w=0)["output_w"] == 28
    assert conv2d_output_size(28, 28, 5, 5)["output_h"] == 24
    assert conv2d_output_size(28, 28, 5, 5)["output_w"] == 24
    padded = conv2d_output_size(32, 32, 3, 3, padding_h=1, padding_w=1)
    assert padded["output_h"] == 32
    assert padded["output_w"] == 32


def test_pool2d_output_size() -> None:
    result = pool2d_output_size(32, 32, 2, 2, stride_h=2, stride_w=2)

    assert result["output_h"] == 16
    assert result["output_w"] == 16
    assert "池化" in str(result["result_text"])


def test_conv2d_output_shape_layouts() -> None:
    chw = conv2d_output_shape((3, 32, 32), "CHW", 64, 3, 3, padding_h=1, padding_w=1)
    nchw = conv2d_output_shape((8, 3, 32, 32), "NCHW", 64, 3, 3, padding_h=1, padding_w=1)
    nhwc = conv2d_output_shape((8, 32, 32, 3), "NHWC", 64, 3, 3, padding_h=1, padding_w=1)

    assert chw["output_shape"] == (64, 32, 32)
    assert nchw["output_shape"] == (8, 64, 32, 32)
    assert nhwc["output_shape"] == (8, 32, 32, 64)


def test_conv2d_param_count() -> None:
    result = conv2d_param_count(in_channels=3, out_channels=64, kernel_h=3, kernel_w=3, bias=True)

    assert result["params"] == 1792
    assert result["weight_params"] == 1728
    assert result["bias_params"] == 64


def test_linear_param_count() -> None:
    result = linear_param_count(in_features=784, out_features=10, bias=True)

    assert result["params"] == 7850


def test_multi_layer_param_count_conv_and_fc() -> None:
    answer = multi_layer_param_count(
        [
            make_conv2d_layer_param_spec("Conv1", 3, 16, 3, 3),
            make_conv2d_layer_param_spec("Conv2", 16, 32, 3, 3),
            make_linear_layer_param_spec("FC", 800, 10),
        ]
    )

    assert [layer.params for layer in answer.layers] == [448, 4640, 8010]
    assert answer.total_params == 13098


def test_multi_layer_param_count_mlp() -> None:
    answer = multi_layer_param_count(
        [
            make_linear_layer_param_spec("FC1", 784, 128),
            make_linear_layer_param_spec("FC2", 128, 64),
            make_linear_layer_param_spec("FC3", 64, 10),
        ]
    )

    assert answer.total_params == 109386


def test_softmax_probability_sum() -> None:
    probs = softmax([2.0, 1.0, 0.0])

    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_cross_entropy_from_logits() -> None:
    loss = cross_entropy_from_logits([2.0, 1.0, 0.0], target_index=0)

    assert loss == pytest.approx(0.4076059644)


def test_mse() -> None:
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4.0 / 3.0)


def test_classification_metrics() -> None:
    metrics = classification_metrics(tp=80, fp=10, tn=90, fn=20)

    assert metrics["accuracy"] == pytest.approx(0.85)
    assert metrics["precision"] == pytest.approx(80 / 90)
    assert metrics["recall"] == pytest.approx(0.8)
    assert metrics["f1"] == pytest.approx(2 * (80 / 90) * 0.8 / ((80 / 90) + 0.8))
    assert classification_metrics(tp=0, fp=0, tn=0, fn=0)["precision"] == 0.0


def test_gradient_descent_step() -> None:
    assert gradient_descent_step(w=2.0, grad=0.5, lr=0.1) == pytest.approx(1.95)


def test_rnn_lstm_attention_param_counts() -> None:
    assert simple_rnn_param_count(input_size=10, hidden_size=20, output_size=5, bias=True)["params"] == 725
    assert lstm_param_count(input_size=10, hidden_size=20, bias=True)["params"] == 2480
    assert attention_qkv_param_count(d_model=64, bias=True)["params"] == 12480
    assert multihead_attention_param_count(d_model=64, include_output_projection=True, bias=True)["params"] == 16640


def test_softmax_is_stable_for_large_logits() -> None:
    probs = softmax([1000.0, 1001.0, 1002.0])

    assert all(math.isfinite(prob) for prob in probs)
    assert sum(probs) == pytest.approx(1.0)
