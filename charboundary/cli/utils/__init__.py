"""
CLI utility modules for CharBoundary.
"""

from charboundary.cli.utils.onnx_utils import (
    # Conversion functions
    convert_segmenter_model,
    convert_custom_model,
    convert_all_models,
    # Benchmark functions
    benchmark_model_optimization_levels,
    benchmark_built_in_models,
    # Testing functions
    test_model,
    test_all_models,
    # Constants
    BUILTIN_MODELS,
)

__all__ = [
    "convert_segmenter_model",
    "convert_custom_model",
    "convert_all_models",
    "benchmark_model_optimization_levels",
    "benchmark_built_in_models",
    "test_model",
    "test_all_models",
    "BUILTIN_MODELS",
]
