#!/usr/bin/env python
"""
ONNX Utilities for CharBoundary

This script provides a comprehensive set of utilities for working with ONNX models:

1. Converting models to ONNX format with optimization levels
2. Benchmarking ONNX models against scikit-learn
3. Testing ONNX model compatibility and accuracy

Usage:
    # Convert all built-in models to ONNX with optimal optimization levels
    python onnx_utils.py convert --all-models
    
    # Convert a specific built-in model to ONNX with a custom optimization level
    python onnx_utils.py convert --model-name small --optimization-level 2
    
    # Convert a custom model file to ONNX
    python onnx_utils.py convert --input-file path/to/model.skops.xz --output-file path/to/model.onnx
    
    # Benchmark all built-in models with optimal optimization levels
    python onnx_utils.py benchmark --all-models
    
    # Benchmark a specific built-in model with all optimization levels
    python onnx_utils.py benchmark --model-name medium
    
    # Test ONNX model functionality and accuracy
    python onnx_utils.py test --all-models
    
    # Test a specific model with a specific optimization level
    python onnx_utils.py test --model-name large --optimization-level 3
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import charboundary
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Check if ONNX is available
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

if not ONNX_AVAILABLE:
    print("Error: ONNX support is not available.")
    print("Please install the ONNX dependencies with: pip install charboundary[onnx]")
    sys.exit(1)

# Import the CLI utilities
try:
    from charboundary.cli.utils.onnx_utils import (
        convert_segmenter_model,
        convert_custom_model,
        convert_all_models,
        benchmark_model_optimization_levels,
        benchmark_built_in_models,
        test_model,
        test_all_models,
        BUILTIN_MODELS
    )
    from charboundary import (
        get_small_segmenter, 
        get_default_segmenter, 
        get_large_segmenter
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure charboundary is installed correctly.")
    sys.exit(1)


def main():
    """Parse arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description="ONNX Utilities for CharBoundary")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert models to ONNX format")
    convert_group = convert_parser.add_mutually_exclusive_group(required=True)
    convert_group.add_argument("--all-models", action="store_true", 
                              help="Convert all built-in models with optimal optimization levels")
    convert_group.add_argument("--model-name", choices=["small", "medium", "large"],
                              help="Convert a specific built-in model")
    convert_group.add_argument("--input-file", type=str,
                              help="Path to a custom model file to convert")
    
    convert_parser.add_argument("--output-file", type=str,
                              help="Path to save the ONNX model (required with --input-file)")
    convert_parser.add_argument("--optimization-level", "-o", type=int, choices=[0, 1, 2, 3], default=2,
                              help="ONNX optimization level (default: 2)")
    convert_parser.add_argument("--feature-count", "-f", type=int,
                              help="Number of features in the model (if not provided, will try to infer)")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark ONNX models")
    benchmark_group = benchmark_parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument("--all-models", action="store_true", 
                                help="Benchmark all built-in models with optimal optimization levels")
    benchmark_group.add_argument("--model-name", choices=["small", "medium", "large"],
                                help="Benchmark a specific built-in model with all optimization levels")
    
    benchmark_parser.add_argument("--runs", "-r", type=int, default=20,
                                help="Number of benchmark runs (default: 20)")
    benchmark_parser.add_argument("--batch-size", "-b", type=int, default=500,
                                help="Number of samples per batch (default: 500)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test ONNX model functionality")
    test_group = test_parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--all-models", action="store_true", 
                           help="Test all built-in models")
    test_group.add_argument("--model-name", choices=["small", "medium", "large"],
                           help="Test a specific built-in model")
    
    test_parser.add_argument("--optimization-level", "-o", type=int, choices=[0, 1, 2, 3],
                           help="ONNX optimization level to test (if not provided, use optimal)")
    
    args = parser.parse_args()
    
    # Handle no command
    if args.command is None:
        parser.print_help()
        return
    
    # Convert command
    if args.command == "convert":
        if args.all_models:
            convert_all_models()
        elif args.model_name:
            model_info = BUILTIN_MODELS[args.model_name]
            convert_segmenter_model(
                model_info["getter"], 
                args.model_name, 
                optimization_level=args.optimization_level
            )
        elif args.input_file:
            if args.output_file is None:
                print("Error: --output-file is required with --input-file")
                return
            convert_custom_model(
                args.input_file, 
                args.output_file, 
                args.feature_count, 
                args.optimization_level
            )
    
    # Benchmark command
    elif args.command == "benchmark":
        if args.all_models:
            benchmark_built_in_models(args.runs, args.batch_size)
        elif args.model_name:
            # Get the appropriate model
            model_info = BUILTIN_MODELS[args.model_name]
            segmenter = model_info["getter"]()
            
            # Benchmark the model
            benchmark_model_optimization_levels(
                segmenter.model, 
                model_info["feature_count"], 
                args.model_name, 
                args.runs, 
                args.batch_size
            )
    
    # Test command
    elif args.command == "test":
        if args.all_models:
            test_all_models()
        elif args.model_name:
            test_model(args.model_name, args.optimization_level)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)