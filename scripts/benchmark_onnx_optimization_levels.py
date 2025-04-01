#!/usr/bin/env python
"""
Benchmark script to compare different ONNX optimization levels.

This script loads a model, generates benchmark data, and compares the
inference speed with various ONNX optimization levels (0-3).

Usage:
    python benchmark_onnx_optimization_levels.py --model path/to/model.skops.xz [--runs 100] [--batch-size 1000]
    
Example:
    python benchmark_onnx_optimization_levels.py --model charboundary/resources/small_model.skops.xz --runs 20 --batch-size 500
"""

import os
import sys
import time
import random
import argparse
from typing import List, Tuple, Dict, Any
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

from charboundary import TextSegmenter
from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel


def infer_feature_count(model) -> int:
    """Try to infer the feature count from the model."""
    if isinstance(model, FeatureSelectedRandomForestModel):
        if hasattr(model, "selected_feature_indices") and model.selected_feature_indices:
            # For feature-selected models, get the count of selected features
            return len(model.selected_feature_indices)
    
    # For standard RandomForest models, get the feature count from the first tree
    if hasattr(model, "model") and hasattr(model.model, "n_features_in_"):
        return model.model.n_features_in_
    
    # If we can't determine the feature count, use a default value
    return 200


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


def benchmark_model_optimization_levels(model_path: str, runs: int = 100, batch_size: int = 1000) -> Dict[str, Any]:
    """Benchmark sklearn vs ONNX inference with different optimization levels."""
    print(f"Loading model from {model_path}...")
    
    try:
        # Try to load as a segmenter with trust_model=True
        segmenter = TextSegmenter.load(model_path, trust_model=True)
        model = segmenter.model
        print("Successfully loaded model from TextSegmenter.")
    except Exception as e:
        print(f"Error loading as TextSegmenter: {e}")
        print("Could not load the model. Please try using the segmenter loaders from the charboundary library.")
        sys.exit(1)
    
    # Verify we have a proper model
    if not isinstance(model, (BinaryRandomForestModel, FeatureSelectedRandomForestModel)):
        print(f"Error: Expected a charboundary model, but got {type(model)}.")
        sys.exit(1)
    
    # Infer feature count
    feature_count = infer_feature_count(model)
    print(f"Feature count: {feature_count}")
    
    # Set the feature count on the model (needed for ONNX conversion)
    model.feature_count = feature_count
    
    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {batch_size})...")
    data = generate_benchmark_data(feature_count, batch_size)
    
    # Run sklearn benchmark
    print(f"Running sklearn benchmark ({runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, runs)
    sklearn_ops = (runs * batch_size) / sklearn_time
    
    # Dictionary to store results
    results = {
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
        
        # Convert to ONNX if needed (will be done in the benchmark function)
        if not model.onnx_model:
            model.to_onnx()
            
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
    
    # Return all results
    return results


def main():
    """Parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark sklearn vs ONNX optimization levels.")
    parser.add_argument("--model", "-m", required=True, help="Path to the model file (.skops, .skops.xz, .pickle, etc.)")
    parser.add_argument("--runs", "-r", type=int, default=100, help="Number of benchmark runs (default: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000, help="Number of samples per batch (default: 1000)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist.")
        sys.exit(1)
    
    # Run the benchmark
    results = benchmark_model_optimization_levels(args.model, args.runs, args.batch_size)
    
    # Find the best optimization level
    best_level = max(results["onnx_results"].items(), key=lambda x: x[1]["ops_per_sec"])
    best_level_num, best_level_results = best_level
    
    # Print final summary message
    print(f"\nBest performance with ONNX optimization level {best_level_num}.")
    print(f"ONNX with level {best_level_num} is {best_level_results['speedup']:.2f}x faster than scikit-learn.")


if __name__ == "__main__":
    main()