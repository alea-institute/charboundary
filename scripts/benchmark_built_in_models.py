#!/usr/bin/env python
"""
Benchmark script to compare sklearn and ONNX inference performance for built-in models.

This script loads the standard models and compares sklearn vs ONNX inference speed.

Usage:
    python benchmark_built_in_models.py [--model small|medium|large] [--runs 100] [--batch-size 1000]
"""

import os
import sys
import time
import random
import argparse
from typing import List, Tuple, Dict, Any
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
    print("Please install the ONNX dependencies with: pip install onnx skl2onnx onnxruntime")
    sys.exit(1)

from charboundary import get_small_segmenter, get_default_segmenter, get_large_segmenter


def generate_benchmark_data(feature_count: int, batch_size: int, actual_model=None) -> List[List[int]]:
    """Generate random benchmark data for testing."""
    # If we have a FeatureSelectedRandomForestModel, ensure data has correct length
    if actual_model is not None and hasattr(actual_model, 'selected_feature_indices'):
        try:
            indices = actual_model.selected_feature_indices
            if indices is not None:
                # Generate data with all original features
                full_features = max(indices) + 1 if indices else feature_count
                print(f"Generating data with {full_features} features (selected: {len(indices)})")
                # First generate full data
                return [[random.randint(0, 1) for _ in range(full_features)] for _ in range(batch_size)]
        except Exception as e:
            print(f"Warning: Error generating model-specific data: {e}")
    
    # Create random binary features (typical for charboundary)
    return [[random.randint(0, 1) for _ in range(feature_count)] for _ in range(batch_size)]


def run_sklearn_benchmark(segmenter, data: List[List[int]], runs: int) -> Tuple[float, List[int]]:
    """Run sklearn inference benchmark."""
    # Get the model
    model = segmenter.model
    
    # Ensure model has use_onnx attribute (for models created before our changes)
    if not hasattr(model, 'use_onnx'):
        setattr(model, 'use_onnx', False)
    
    # Disable ONNX if enabled
    original_onnx_state = model.use_onnx
    model.use_onnx = False
    
    # Ensure we use model.predict directly for more accurate benchmarking
    # Warmup run
    _ = model.predict(data)
    
    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = model.predict(data)
    end_time = time.time()
    
    # Restore original ONNX state
    if hasattr(model, 'use_onnx'):
        model.use_onnx = original_onnx_state
    
    total_time = end_time - start_time
    return total_time, predictions


def run_onnx_benchmark(segmenter, data: List[List[int]], runs: int, model_name: str = "small") -> Tuple[float, List[int]]:
    """Run ONNX inference benchmark."""
    # Get the model
    model = segmenter.model
    
    # Ensure model has required attributes (for models created before our changes)
    if not hasattr(model, 'use_onnx'):
        setattr(model, 'use_onnx', False)
    if not hasattr(model, 'onnx_model'):
        setattr(model, 'onnx_model', None)
    if not hasattr(model, 'onnx_session'):
        setattr(model, 'onnx_session', None)
    
    # Set feature count if needed
    if not hasattr(model, 'feature_count') or model.feature_count is None:
        # Infer from data if possible
        if data and len(data) > 0:
            model.feature_count = len(data[0])
        else:
            model.feature_count = 32  # Default fallback
    
    # Try to convert to ONNX if not already done
    if model.onnx_model is None:
        onnx_model = segmenter.to_onnx()
        if onnx_model is None:
            raise RuntimeError("Failed to convert model to ONNX format")
    
    # Enable ONNX mode
    try:
        segmenter.enable_onnx(True)
    except Exception as e:
        # If we can't use segmenter's method, try direct attribute setting
        model.use_onnx = True
        # Try to load the ONNX model directly
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "..", "charboundary", "resources", "onnx", 
                                 f"{model_name}_model.onnx")
        if os.path.exists(onnx_path):
            try:
                from charboundary.onnx_support import load_onnx_model, create_onnx_inference_session
                model.onnx_model = load_onnx_model(onnx_path)
                model.onnx_session = create_onnx_inference_session(model.onnx_model)
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
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


def load_segmenter(model_name):
    """Load a segmenter based on model name."""
    if model_name == "small":
        return get_small_segmenter()
    elif model_name == "medium":
        return get_default_segmenter()
    elif model_name == "large":
        return get_large_segmenter()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def infer_feature_count(model):
    """Infer feature count from the model."""
    # For models with feature_importances_
    if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
        return len(model.model.feature_importances_)
    
    # For sklearn RandomForest
    if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
        return model.model.n_features_in_
        
    # For feature-selected models
    if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices:
        return len(model.selected_feature_indices)
    
    # For models with a FeatureSelectedRandomForestModel attribute
    if hasattr(model, 'model') and hasattr(model.model, 'selected_feature_indices') and model.model.selected_feature_indices:
        return len(model.model.selected_feature_indices)
    
    # Default sizes based on known model specifications
    model_sizes = {
        "small": 19,   # Small model has 19 features (from our conversion)
        "medium": 21,  # Medium model has 21 features (from our conversion)
        "large": 27    # Large model has 27 features (from our conversion)
    }
    
    # Check model type from its string representation
    model_str = str(model)
    for model_type, count in model_sizes.items():
        if model_type in model_str.lower():
            return count
    
    # Default fallback
    return 32


def benchmark_model(model_name: str, runs: int = 100, batch_size: int = 1000):
    """Benchmark sklearn vs ONNX inference for a built-in model."""
    print(f"Benchmarking {model_name} model...")
    
    # Load the segmenter
    print(f"Loading {model_name} model...")
    segmenter = load_segmenter(model_name)
    model = segmenter.model
    
    # Infer feature count
    feature_count = infer_feature_count(model)
    if not hasattr(model, 'feature_count') or model.feature_count is None:
        model.feature_count = feature_count
    print(f"Feature count: {feature_count}")
    
    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {batch_size}, features: {feature_count})...")
    data = generate_benchmark_data(feature_count, batch_size, model)
    
    # Run sklearn benchmark
    print(f"Running sklearn benchmark ({runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(segmenter, data, runs)
    sklearn_ops = (runs * batch_size) / sklearn_time
    
    # Run ONNX benchmark
    print(f"Running ONNX benchmark ({runs} runs)...")
    onnx_time, onnx_predictions = run_onnx_benchmark(segmenter, data, runs, model_name)
    onnx_ops = (runs * batch_size) / onnx_time
    
    # Verify results match
    predictions_match = all(a == b for a, b in zip(sklearn_predictions, onnx_predictions))
    
    # Print results
    print("\nBenchmark Results:")
    print(f"{'Framework':12} {'Time (s)':10} {'Ops/sec':15} {'Rel. Speedup':12}")
    print("-" * 50)
    print(f"{'sklearn':12} {sklearn_time:10.4f} {sklearn_ops:15.2f} {'1.00x':12}")
    print(f"{'ONNX':12} {onnx_time:10.4f} {onnx_ops:15.2f} {onnx_ops/sklearn_ops:11.2f}x")
    print("\nAccuracy check:")
    print(f"Predictions match: {'Yes' if predictions_match else 'No'}")
    
    # Return results
    return {
        "model": model_name,
        "feature_count": feature_count,
        "sklearn_time": sklearn_time,
        "sklearn_ops_per_sec": sklearn_ops,
        "onnx_time": onnx_time,
        "onnx_ops_per_sec": onnx_ops,
        "speedup": onnx_ops / sklearn_ops,
        "predictions_match": predictions_match,
    }


def main():
    """Parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark sklearn vs ONNX inference for built-in models.")
    parser.add_argument("--model", "-m", choices=["small", "medium", "large"], default="small",
                        help="Model to benchmark (small, medium, or large)")
    parser.add_argument("--runs", "-r", type=int, default=10, 
                        help="Number of benchmark runs (default: 10)")
    parser.add_argument("--batch-size", "-b", type=int, default=5000,
                        help="Number of samples per batch (default: 5000)")
    
    args = parser.parse_args()
    
    # Check if ONNX models exist, if not try to convert
    onnx_dir = os.path.join(PROJECT_DIR, "charboundary", "resources", "onnx")
    onnx_path = os.path.join(onnx_dir, f"{args.model}_model.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        print("You can create it using the convert_models_to_onnx.py script")
    
    # Run the benchmark
    results = benchmark_model(args.model, args.runs, args.batch_size)
    
    # Print final summary message
    if results["speedup"] > 1.0:
        print(f"\nONNX is {results['speedup']:.2f}x faster than sklearn for the {args.model} model.")
    else:
        print(f"\nONNX is {1/results['speedup']:.2f}x slower than sklearn for the {args.model} model.")


if __name__ == "__main__":
    main()