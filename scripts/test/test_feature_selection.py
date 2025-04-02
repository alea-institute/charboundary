#!/usr/bin/env python3
"""
Script to test and demonstrate the feature selection functionality in CharBoundary.
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from charboundary import TextSegmenter


def train_with_feature_selection(
    data_path: str,
    output_path: str,
    max_samples: int = 1000,
    sample_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 16,
    feature_selection_threshold: float = 0.01,
    max_features: int = None,
    left_window: int = 5,
    right_window: int = 5,
):
    """
    Train a model with feature selection and evaluate its performance.
    
    Args:
        data_path (str): Path to the training data file
        output_path (str): Path to save the trained model
        max_samples (int, optional): Maximum number of samples to use. Defaults to 1000.
        sample_rate (float, optional): Sampling rate for non-boundary samples. Defaults to 0.1.
        n_estimators (int, optional): Number of trees in the random forest. Defaults to 100.
        max_depth (int, optional): Maximum depth of the trees. Defaults to 16.
        feature_selection_threshold (float, optional): Threshold for feature selection. Defaults to 0.01.
        max_features (int, optional): Maximum number of features to select. Defaults to None.
        left_window (int, optional): Size of the left context window. Defaults to 5.
        right_window (int, optional): Size of the right context window. Defaults to 5.
    """
    print(f"Loading data from {data_path}")
    
    # Load the training data
    training_data = []
    if data_path.endswith('.jsonl.gz'):
        # Handle gzipped jsonl files
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    json_obj = json.loads(line.strip())
                    if 'text' in json_obj:
                        training_data.append(json_obj['text'])
                        if len(training_data) >= max_samples:
                            break
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line at position {i}")
    else:
        # Handle regular text files
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                training_data.append(line.strip())
                if len(training_data) >= max_samples:
                    break
    
    print(f"Loaded {len(training_data)} training samples")
    
    # Train models with and without feature selection for comparison
    print("\n=== Training with Feature Selection ===")
    segmenter_with_fs = TextSegmenter()
    start_time = time.time()
    metrics_with_fs = segmenter_with_fs.train(
        data=training_data,
        sample_rate=sample_rate,
        model_params={"n_estimators": n_estimators, "max_depth": max_depth},
        left_window=left_window,
        right_window=right_window,
        use_feature_selection=True,
        feature_selection_threshold=feature_selection_threshold,
        max_features=max_features
    )
    fs_time = time.time() - start_time
    
    # Train without feature selection for comparison
    print("\n=== Training without Feature Selection ===")
    segmenter_without_fs = TextSegmenter()
    start_time = time.time()
    metrics_without_fs = segmenter_without_fs.train(
        data=training_data,
        sample_rate=sample_rate,
        model_params={"n_estimators": n_estimators, "max_depth": max_depth},
        left_window=left_window,
        right_window=right_window,
    )
    no_fs_time = time.time() - start_time
    
    # Compare performance metrics
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} | {'With FS':<10} | {'Without FS':<10}")
    print(f"{'-'*20} | {'-'*10} | {'-'*10}")
    for metric in ["accuracy", "precision", "recall", "f1_score", "boundary_accuracy"]:
        with_val = metrics_with_fs.get(metric, 0)
        without_val = metrics_without_fs.get(metric, 0)
        print(f"{metric:<20} | {with_val:.4f}    | {without_val:.4f}")
    
    print(f"\nTraining Time:")
    print(f"{'With Feature Selection':<25}: {fs_time:.2f} seconds")
    print(f"{'Without Feature Selection':<25}: {no_fs_time:.2f} seconds")
    
    # Save the model with feature selection if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"\nSaving model with feature selection to {output_path}")
        segmenter_with_fs.save(output_path)
    
    # Test inference speed
    print("\n=== Inference Speed Comparison ===")
    
    # Create some test examples
    test_texts = [
        "Hello world. This is a simple test with multiple sentences. Dr. Smith visited Washington D.C. last week.",
        "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial segregation in public schools was unconstitutional. This landmark decision changed the course of U.S. history."
    ]
    
    # Test with feature selection
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        for text in test_texts:
            _ = segmenter_with_fs.segment_to_sentences(text)
    with_fs_inference_time = (time.time() - start_time) / iterations
    
    # Test without feature selection
    start_time = time.time()
    for _ in range(iterations):
        for text in test_texts:
            _ = segmenter_without_fs.segment_to_sentences(text)
    without_fs_inference_time = (time.time() - start_time) / iterations
    
    print(f"Average inference time per batch:")
    print(f"{'With Feature Selection':<25}: {with_fs_inference_time*1000:.2f} ms")
    print(f"{'Without Feature Selection':<25}: {without_fs_inference_time*1000:.2f} ms")
    
    speedup = without_fs_inference_time / with_fs_inference_time if with_fs_inference_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test feature selection in the CharBoundary model")
    
    parser.add_argument("--data", type=str, default="data/train.jsonl.gz",
                        help="Path to the training data")
    parser.add_argument("--output", type=str, default="models/feature_selected_model.skops.xz", 
                        help="Path to save the model with feature selection")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Maximum number of samples to use")
    parser.add_argument("--sample-rate", type=float, default=0.1,
                        help="Sampling rate for non-boundary samples")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Feature selection threshold (importance threshold)")
    parser.add_argument("--max-features", type=int, default=None,
                        help="Maximum number of features to select")
    parser.add_argument("--estimators", type=int, default=100,
                        help="Number of trees in the random forest")
    parser.add_argument("--depth", type=int, default=16,
                        help="Maximum depth of the trees")
    parser.add_argument("--left-window", type=int, default=5,
                        help="Size of the left context window")
    parser.add_argument("--right-window", type=int, default=5,
                        help="Size of the right context window")
    
    args = parser.parse_args()
    
    train_with_feature_selection(
        data_path=args.data,
        output_path=args.output,
        max_samples=args.samples,
        sample_rate=args.sample_rate,
        n_estimators=args.estimators,
        max_depth=args.depth,
        feature_selection_threshold=args.threshold,
        max_features=args.max_features,
        left_window=args.left_window,
        right_window=args.right_window,
    )