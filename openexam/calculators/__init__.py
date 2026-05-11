from __future__ import annotations

from openexam.calculators.cnn import (
    LayerParamSpec,
    MultiLayerParamAnswer,
    conv2d_output_shape,
    conv2d_output_size,
    conv2d_param_count,
    linear_param_count,
    make_conv2d_layer_param_spec,
    make_linear_layer_param_spec,
    multi_layer_param_count,
    pool2d_output_size,
)
from openexam.calculators.losses import cross_entropy_from_logits, cross_entropy_from_probs, mse, softmax
from openexam.calculators.metrics import classification_metrics
from openexam.calculators.optimization import gradient_descent_step
from openexam.calculators.parser import (
    CalculationAnswer,
    CalculationParseResult,
    detect_calculation_type,
    parse_calculation_question,
    solve_calculation_question,
)
from openexam.calculators.sequence import lstm_param_count, simple_rnn_param_count
from openexam.calculators.transformer import attention_qkv_param_count, multihead_attention_param_count

__all__ = [
    "CalculationAnswer",
    "CalculationParseResult",
    "LayerParamSpec",
    "MultiLayerParamAnswer",
    "attention_qkv_param_count",
    "classification_metrics",
    "conv2d_output_shape",
    "conv2d_output_size",
    "conv2d_param_count",
    "cross_entropy_from_logits",
    "cross_entropy_from_probs",
    "detect_calculation_type",
    "gradient_descent_step",
    "linear_param_count",
    "lstm_param_count",
    "make_conv2d_layer_param_spec",
    "make_linear_layer_param_spec",
    "mse",
    "multi_layer_param_count",
    "multihead_attention_param_count",
    "parse_calculation_question",
    "pool2d_output_size",
    "simple_rnn_param_count",
    "softmax",
    "solve_calculation_question",
]
