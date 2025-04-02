#!/usr/bin/env python
"""
Comprehensive ONNX benchmark script for charboundary models.

This script provides multiple benchmark modes:
1. Benchmark a specific model with different ONNX optimization levels
2. Benchmark all built-in models with their optimal optimization levels
3. Benchmark a custom model file with ONNX vs scikit-learn

Usage:
    # Benchmark all built-in models with optimal optimization levels
    python benchmark_onnx_models.py --built-in-models
    
    # Benchmark a specific built-in model with all optimization levels
    python benchmark_onnx_models.py --model-name small
    python benchmark_onnx_models.py --model-name medium
    python benchmark_onnx_models.py --model-name large
    
    # Benchmark a specific model file with optimization levels
    python benchmark_onnx_models.py --model-file path/to/model.skops.xz
    
    # Additional parameters
    python benchmark_onnx_models.py --runs 50 --batch-size 1000
"""

import os
import sys
import time
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
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

from charboundary import (
    get_small_segmenter, get_default_segmenter, get_large_segmenter
)
from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel
from charboundary import TextSegmenter


def generate_benchmark_data(feature_count: int, batch_size: int) -> List[List[int]]:
    """Generate random benchmark data for testing."""
    # Create random binary features (typical for charboundary)
    return [[random.randint(0, 1) for _ in range(feature_count)] for _ in range(batch_size)]


def run_sklearn_benchmark(model, data: List[List[int]], runs: int) -> Tuple[float, List[int]]:
    """Run sklearn inference benchmark."""
    # Ensure the model has a use_onnx attribute
    if not hasattr(model, 'use_onnx'):
        setattr(model, 'use_onnx', False)
    
    # Store original state and disable ONNX
    original_onnx_state = model.use_onnx
    model.use_onnx = False
    
    # Warmup run
    _ = model.predict(data)
    
    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = model.predict(data)
    end_time = time.time()
    
    # Restore original ONNX state
    model.use_onnx = original_onnx_state
    
    total_time = end_time - start_time
    return total_time, predictions


def run_onnx_benchmark(model, data: List[List[int]], runs: int, optimization_level: int) -> Tuple[float, List[int]]:
    """Run ONNX inference benchmark with specified optimization level."""
    # Ensure model has all required attributes
    if not hasattr(model, 'use_onnx'):
        setattr(model, 'use_onnx', False)
    
    if not hasattr(model, 'onnx_model'):
        setattr(model, 'onnx_model', None)
    
    if not hasattr(model, 'onnx_session'):
        setattr(model, 'onnx_session', None)
        
    if not hasattr(model, 'onnx_optimization_level'):
        setattr(model, 'onnx_optimization_level', optimization_level)
    
    # Make sure the model has a feature count
    if not hasattr(model, 'feature_count') or model.feature_count is None:
        # Try to infer feature count
        if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices:
            model.feature_count = len(model.selected_feature_indices)
        elif hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            model.feature_count = model.model.n_features_in_
        else:
            model.feature_count = len(data[0])  # Use input data as a guide
    
    # Convert to ONNX if needed
    if model.onnx_model is None:
        try:
            model.to_onnx()
        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            raise
    
    # Make sure ONNX is enabled with the specified optimization level
    try:
        model.enable_onnx(True, optimization_level=optimization_level)
    except Exception as e:
        print(f"Error enabling ONNX with optimization level {optimization_level}: {e}")
        raise
    
    # Warmup run
    _ = model.predict(data)
    
    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = model.predict(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    return total_time, predictions


def benchmark_model_optimization_levels(model, feature_count: int, model_name: str, 
                                      runs: int = 20, batch_size: int = 500) -> Dict[str, Any]:
    """
    Benchmark a model with different ONNX optimization levels.
    
    Args:
        model: The model to benchmark
        feature_count: Number of features in the model
        model_name: Name of the model (for reporting)
        runs: Number of benchmark runs
        batch_size: Number of samples per batch
        
    Returns:
        Dict: Benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_name.upper()} model")
    print(f"{'='*70}")
    
    print(f"Model: {model_name}")
    print(f"Feature count: {feature_count}")
    
    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {batch_size})...")
    data = generate_benchmark_data(feature_count, batch_size)
    
    # Run sklearn benchmark
    print(f"Running sklearn benchmark ({runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, runs)
    sklearn_ops = (runs * batch_size) / sklearn_time
    
    # Dictionary to store results
    results = {
        "model_name": model_name,
        "sklearn_time": sklearn_time,
        "sklearn_ops_per_sec": sklearn_ops,
        "feature_count": feature_count,
        "batch_size": batch_size,
        "runs": runs,
        "onnx_results": {}
    }
    
    # Test all ONNX optimization levels
    print("\nBenchmarking ONNX optimization levels:")
    print(f"{'Level':10} {'Description':25} {'Time (s)':10} {'Ops/sec':15} {'Rel. Speedup':12}")
    print("-" * 75)
    
    optimization_levels = [
        (0, "No optimization"),
        (1, "Basic optimizations"),
        (2, "Extended optimizations"),
        (3, "All optimizations")
    ]
    
    for level, description in optimization_levels:
        print(f"Level {level}: Testing {description}...", end="", flush=True)
        
        # Run benchmark with this optimization level
        onnx_time, onnx_predictions = run_onnx_benchmark(model, data, runs, level)
        onnx_ops = (runs * batch_size) / onnx_time
        speedup = onnx_ops / sklearn_ops
        
        # Verify results match
        predictions_match = (sklearn_predictions == onnx_predictions)
        if isinstance(predictions_match, list):
            predictions_match = all(predictions_match)
        elif isinstance(predictions_match, np.ndarray):
            predictions_match = predictions_match.all()
            
        # Store results
        results["onnx_results"][level] = {
            "time": onnx_time,
            "ops_per_sec": onnx_ops,
            "speedup": speedup,
            "predictions_match": predictions_match
        }
        
        # Print results
        accuracy_indicator = "✓" if predictions_match else "✗"
        print(f"\r{level:10} {description:25} {onnx_time:10.4f} {onnx_ops:15.2f} {speedup:11.2f}x {accuracy_indicator}")
    
    # Find the best optimization level
    best_level = max(results["onnx_results"].items(), key=lambda x: x[1]["ops_per_sec"])
    best_level_num, best_level_results = best_level
    
    print(f"\nBest performance with ONNX optimization level {best_level_num}.")
    print(f"ONNX with level {best_level_num} is {best_level_results['speedup']:.2f}x faster than scikit-learn.")
    
    return results


def benchmark_built_in_models(runs: int = 20, batch_size: int = 500) -> Dict[str, Any]:
    """
    Benchmark all built-in models with their recommended optimization levels.
    
    Args:
        runs: Number of benchmark runs
        batch_size: Number of samples per batch
        
    Returns:
        Dict: Benchmark results for all models
    """
    # Define the models and their optimal optimization levels
    models = [
        {"name": "small", "getter": get_small_segmenter, "feature_count": 19, "opt_level": 2},
        {"name": "medium", "getter": get_default_segmenter, "feature_count": 21, "opt_level": 2},
        {"name": "large", "getter": get_large_segmenter, "feature_count": 27, "opt_level": 3}
    ]
    
    results = {}
    
    for model_info in models:
        # Get the segmenter with the model
        segmenter = model_info["getter"]()
        model = segmenter.model
        
        # Ensure feature count is set
        model.feature_count = model_info["feature_count"]
        
        # Benchmark sklearn vs ONNX with optimal level
        print(f"\n{'='*70}")
        print(f"Benchmarking {model_info['name'].upper()} model with optimization level {model_info['opt_level']}")
        print(f"{'='*70}")
        
        # Generate benchmark data
        data = generate_benchmark_data(model_info["feature_count"], batch_size)
        
        # Benchmark sklearn
        sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, runs)
        sklearn_ops = (runs * batch_size) / sklearn_time
        print(f"scikit-learn: {sklearn_ops:.2f} ops/sec, {sklearn_time:.4f} seconds")
        
        # Benchmark ONNX with optimal level
        onnx_time, onnx_predictions = run_onnx_benchmark(model, data, runs, model_info["opt_level"])
        onnx_ops = (runs * batch_size) / onnx_time
        speedup = onnx_ops / sklearn_ops
        
        # Check if predictions match
        predictions_match = (sklearn_predictions == onnx_predictions)
        if isinstance(predictions_match, list):
            predictions_match = all(predictions_match)
        elif isinstance(predictions_match, np.ndarray):
            predictions_match = predictions_match.all()
        
        accuracy_indicator = "✓" if predictions_match else "✗"
        print(f"ONNX (level {model_info['opt_level']}): {onnx_ops:.2f} ops/sec, {onnx_time:.4f} seconds")
        print(f"Speedup: {speedup:.2f}x {accuracy_indicator}")
        
        # Save results
        results[model_info["name"]] = {
            "feature_count": model_info["feature_count"],
            "sklearn_ops": sklearn_ops,
            "onnx_ops": onnx_ops,
            "speedup": speedup,
            "opt_level": model_info["opt_level"],
            "predictions_match": predictions_match
        }
    
    # Print a summary table
    print("\n" + "="*80)
    print("ONNX PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Model':10} {'Opt. Level':15} {'sklearn (ops/s)':20} {'ONNX (ops/s)':20} {'Speedup':10}")
    print("-" * 80)
    
    for model_name, model_results in results.items():
        level_str = f"Level {model_results['opt_level']}"
        print(f"{model_name:10} {level_str:15} "
              f"{model_results['sklearn_ops']:20.2f} {model_results['onnx_ops']:20.2f} "
              f"{model_results['speedup']:10.2f}x")
    
    return results


def benchmark_model_file(model_path: str, runs: int = 20, batch_size: int = 500) -> Dict[str, Any]:
    """
    Benchmark a specific model file with all optimization levels.
    
    Args:
        model_path: Path to the model file
        runs: Number of benchmark runs
        batch_size: Number of samples per batch
        
    Returns:
        Dict: Benchmark results
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Try to load as a segmenter
        segmenter = TextSegmenter.load(model_path, trust_model=True)
        model = segmenter.model
        print("Successfully loaded model from TextSegmenter.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Determine feature count
    feature_count = None
    
    if isinstance(model, FeatureSelectedRandomForestModel) and model.selected_feature_indices:
        feature_count = len(model.selected_feature_indices)
    elif hasattr(model, "feature_count") and model.feature_count:
        feature_count = model.feature_count
    elif hasattr(model, "model") and hasattr(model.model, "n_features_in_"):
        feature_count = model.model.n_features_in_
    else:
        # Default to a reasonable size
        feature_count = 20
        print(f"Could not determine feature count, using default: {feature_count}")
    
    # Set the feature count on the model
    model.feature_count = feature_count
    
    # Get the model name from the file path
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Run the benchmark
    return benchmark_model_optimization_levels(model, feature_count, model_name, runs, batch_size)


def main():
    """Parse arguments and run the appropriate benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark ONNX models with different optimization levels.")
    
    # Model selection options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--built-in-models", action="store_true", 
                             help="Benchmark all built-in models with optimal optimization levels")
    model_group.add_argument("--model-name", choices=["small", "medium", "large"],
                             help="Benchmark a specific built-in model with all optimization levels")
    model_group.add_argument("--model-file", type=str, 
                             help="Path to a model file to benchmark")
    
    # Benchmark parameters
    parser.add_argument("--runs", "-r", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    parser.add_argument("--batch-size", "-b", type=int, default=500,
                        help="Number of samples per batch (default: 500)")
    
    args = parser.parse_args()
    
    # Run the appropriate benchmark
    if args.built_in_models:
        benchmark_built_in_models(args.runs, args.batch_size)
    elif args.model_name:
        # Get the appropriate model
        if args.model_name == "small":
            segmenter = get_small_segmenter()
            feature_count = 19
        elif args.model_name == "medium":
            segmenter = get_default_segmenter()
            feature_count = 21
        elif args.model_name == "large":
            segmenter = get_large_segmenter()
            feature_count = 27
        
        # Benchmark the model
        benchmark_model_optimization_levels(segmenter.model, feature_count, args.model_name, 
                                          args.runs, args.batch_size)
    elif args.model_file:
        # Validate path
        if not os.path.exists(args.model_file):
            print(f"Error: Model file {args.model_file} does not exist.")
            sys.exit(1)
        
        # Benchmark the model file
        benchmark_model_file(args.model_file, args.runs, args.batch_size)


if __name__ == "__main__":
    main()