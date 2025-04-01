#!/usr/bin/env python
"""
Benchmark built-in models with different ONNX optimization levels.

This script tests the small, medium, and large built-in models with
different ONNX optimization levels to find the optimal configuration
for each model.

Usage:
    python benchmark_built_in_models_optimization.py [--runs 100] [--batch-size 1000]
"""

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Any, Tuple
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
    get_small_segmenter, get_default_segmenter, get_large_segmenter,
    get_small_onnx_segmenter, get_medium_onnx_segmenter, get_large_onnx_segmenter
)
from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel


def generate_benchmark_data(feature_count: int, batch_size: int) -> List[List[int]]:
    """Generate random benchmark data for testing."""
    # Create random binary features (typical for charboundary)
    return [[random.randint(0, 1) for _ in range(feature_count)] for _ in range(batch_size)]


def run_sklearn_benchmark(segmenter, data: List[List[int]], runs: int) -> Tuple[float, List[int]]:
    """Run sklearn inference benchmark."""
    # Ensure ONNX is disabled
    segmenter.config.use_onnx = False
    if isinstance(segmenter.model, (BinaryRandomForestModel, FeatureSelectedRandomForestModel)):
        segmenter.model.use_onnx = False
    
    # Warmup run
    _ = segmenter.model.predict(data)
    
    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = segmenter.model.predict(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    return total_time, predictions


def run_onnx_benchmark(segmenter, data: List[List[int]], runs: int, optimization_level: int) -> Tuple[float, List[int]]:
    """Run ONNX inference benchmark with specified optimization level."""
    # Enable ONNX with specified optimization level
    segmenter.config.use_onnx = True
    if isinstance(segmenter.model, (BinaryRandomForestModel, FeatureSelectedRandomForestModel)):
        if not segmenter.model.onnx_model:
            segmenter.model.to_onnx()
        segmenter.model.enable_onnx(True, optimization_level=optimization_level)
    
    # Warmup run
    _ = segmenter.model.predict(data)
    
    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = segmenter.model.predict(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    return total_time, predictions


def benchmark_built_in_model(model_name: str, runs: int = 100, batch_size: int = 1000) -> Dict[str, Any]:
    """Benchmark a built-in model with different ONNX optimization levels."""
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_name.upper()} model")
    print(f"{'='*70}")
    
    # Load the appropriate model
    if model_name == "small":
        segmenter = get_small_segmenter()
    elif model_name == "medium":
        segmenter = get_default_segmenter()
    elif model_name == "large":
        segmenter = get_large_segmenter()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Determine feature count
    if isinstance(segmenter.model, FeatureSelectedRandomForestModel) and segmenter.model.selected_feature_indices:
        feature_count = len(segmenter.model.selected_feature_indices)
    elif hasattr(segmenter.model, "feature_count") and segmenter.model.feature_count:
        feature_count = segmenter.model.feature_count
    else:
        feature_count = segmenter.model.model.n_features_in_
    
    # Ensure the model has a feature count set (needed for ONNX conversion)
    segmenter.model.feature_count = feature_count
    
    # Pre-convert to ONNX (will be used later in benchmarks)
    if not hasattr(segmenter.model, "onnx_model") or segmenter.model.onnx_model is None:
        print("Pre-converting model to ONNX for benchmarks...")
        segmenter.model.onnx_model = None  # Ensure attribute exists
        segmenter.model.to_onnx()
        
    print(f"Model: {model_name}")
    print(f"Feature count: {feature_count}")
    
    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {batch_size})...")
    data = generate_benchmark_data(feature_count, batch_size)
    
    # Run sklearn benchmark
    print(f"Running sklearn benchmark ({runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(segmenter, data, runs)
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
        onnx_time, onnx_predictions = run_onnx_benchmark(segmenter, data, runs, level)
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
    
    # Return all results
    return results


def main():
    """Parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark built-in models with different ONNX optimization levels.")
    parser.add_argument("--runs", "-r", type=int, default=100, help="Number of benchmark runs (default: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000, help="Number of samples per batch (default: 1000)")
    
    args = parser.parse_args()
    
    # Run benchmarks for all built-in models
    model_names = ["small", "medium", "large"]
    all_results = {}
    
    for model_name in model_names:
        all_results[model_name] = benchmark_built_in_model(model_name, args.runs, args.batch_size)
    
    # Print final summary
    print("\n\n" + "="*80)
    print("OPTIMIZATION LEVEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Model':10} {'Best Level':15} {'Speedup vs sklearn':20} {'Ops/sec':15}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        # Find the best optimization level
        best_level = max(results["onnx_results"].items(), key=lambda x: x[1]["ops_per_sec"])
        best_level_num, best_level_results = best_level
        
        print(f"{model_name:10} {f'Level {best_level_num}':15} {best_level_results['speedup']:19.2f}x {best_level_results['ops_per_sec']:15.2f}")
    
    # Generate markdown table for README
    print("\n\nMarkdown table for documentation:")
    print("```markdown")
    print("| Model | Best Optimization Level | Speedup vs sklearn | Operations/second |")
    print("|-------|------------------------|---------------------|------------------|")
    
    for model_name, results in all_results.items():
        best_level = max(results["onnx_results"].items(), key=lambda x: x[1]["ops_per_sec"])
        best_level_num, best_level_results = best_level
        
        print(f"| {model_name} | Level {best_level_num} | {best_level_results['speedup']:.2f}x | {best_level_results['ops_per_sec']:.2f} |")
    
    print("```")


if __name__ == "__main__":
    main()