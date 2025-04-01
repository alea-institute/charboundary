#!/usr/bin/env python
"""
Benchmark script to compare ONNX optimization levels for the small model.

This script loads the small model, and benchmarks it with different
ONNX optimization levels (0-3).

Usage:
    python benchmark_onnx_opt_small.py [--runs 20] [--batch-size 500]
"""

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Any, Tuple
import numpy as np

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

from charboundary import get_small_segmenter
from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel


def generate_benchmark_data(feature_count: int, batch_size: int) -> List[List[int]]:
    """Generate random benchmark data for testing."""
    # Create random binary features (typical for charboundary)
    return [[random.randint(0, 1) for _ in range(feature_count)] for _ in range(batch_size)]


def run_sklearn_benchmark(model, data: List[List[int]], runs: int) -> Tuple[float, List[int]]:
    """Run sklearn inference benchmark."""
    # Disable ONNX if enabled
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
    # Make sure ONNX is enabled with the specified optimization level
    if not model.onnx_model:
        model.to_onnx()
    model.enable_onnx(True, optimization_level=optimization_level)
    
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


def main():
    """Parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark ONNX optimization levels for small model.")
    parser.add_argument("--runs", "-r", type=int, default=20, help="Number of benchmark runs (default: 20)")
    parser.add_argument("--batch-size", "-b", type=int, default=500, help="Number of samples per batch (default: 500)")
    
    args = parser.parse_args()
    
    print("\nBenchmarking SMALL model with ONNX optimization levels")
    print("=" * 70)
    
    # Load the small model
    segmenter = get_small_segmenter()
    model = segmenter.model
    
    # Determine feature count
    if isinstance(model, FeatureSelectedRandomForestModel) and model.selected_feature_indices:
        feature_count = len(model.selected_feature_indices)
    elif hasattr(model, "feature_count") and model.feature_count:
        feature_count = model.feature_count
    else:
        feature_count = model.model.n_features_in_
    
    # Ensure the model has a feature count set
    model.feature_count = feature_count
    print(f"Model feature count: {feature_count}")
    
    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {args.batch_size})...")
    data = generate_benchmark_data(feature_count, args.batch_size)
    
    # Run sklearn benchmark first
    print(f"Running sklearn benchmark ({args.runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, args.runs)
    sklearn_ops = (args.runs * args.batch_size) / sklearn_time
    
    # Dictionary to store results
    results = {
        "sklearn_time": sklearn_time,
        "sklearn_ops_per_sec": sklearn_ops,
        "feature_count": feature_count,
        "batch_size": args.batch_size,
        "runs": args.runs,
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
        
        # Convert to ONNX if needed (will be done in the benchmark function)
        if not model.onnx_model:
            model.to_onnx()
            
        # Run benchmark with this optimization level
        onnx_time, onnx_predictions = run_onnx_benchmark(model, data, args.runs, level)
        onnx_ops = (args.runs * args.batch_size) / onnx_time
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
    
    # Print summary
    print("\nSummary:")
    print(f"{'Framework':20} {'Ops/sec':15} {'Rel. Speedup':12}")
    print("-" * 50)
    print(f"{'scikit-learn':20} {sklearn_ops:15.2f} {'1.00x':12}")
    
    # Find the best optimization level
    best_level = max(results["onnx_results"].items(), key=lambda x: x[1]["ops_per_sec"])
    best_level_num, best_level_results = best_level
    
    for level, level_desc in optimization_levels:
        level_results = results["onnx_results"][level]
        framework_name = f"ONNX (level {level})"
        highlight = " (best)" if level == best_level_num else ""
        print(f"{framework_name:20} {level_results['ops_per_sec']:15.2f} {level_results['speedup']:11.2f}x{highlight}")
    
    # Print final summary message
    print(f"\nBest performance with ONNX optimization level {best_level_num}.")
    print(f"ONNX with level {best_level_num} is {best_level_results['speedup']:.2f}x faster than scikit-learn.")


if __name__ == "__main__":
    main()