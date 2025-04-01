#!/usr/bin/env python
"""
Example script to demonstrate using different ONNX optimization levels.

This script tests the small model with different ONNX optimization levels and
compares the performance to scikit-learn.
"""

import time
from typing import List

from charboundary import get_small_segmenter

# Sample text for performance testing (repeat to make it longer)
SAMPLE_TEXT = """
The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial 
segregation in public schools was unconstitutional. This landmark decision changed 
the course of U.S. history. The ruling was unanimous and delivered by Chief Justice 
Earl Warren. Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson at 2:30 p.m.
""" * 50


def benchmark_segmentation(optimization_level: int = None) -> float:
    """
    Benchmark segmentation with a specific ONNX optimization level.
    
    Args:
        optimization_level: ONNX optimization level (0-3)
                          If None, use scikit-learn inference
    
    Returns:
        float: Processing speed in characters per second
    """
    # Get the small segmenter
    segmenter = get_small_segmenter()
    
    # Configure ONNX if needed
    if optimization_level is not None:
        # Make sure ONNX is enabled with the specified optimization level
        segmenter.model.feature_count = 19  # Small model has 19 features
        segmenter.model.to_onnx()
        segmenter.model.enable_onnx(True, optimization_level=optimization_level)
    else:
        # Disable ONNX to use scikit-learn
        segmenter.config.use_onnx = False
        if hasattr(segmenter.model, "enable_onnx"):
            segmenter.model.enable_onnx(False)
    
    # Warm-up run
    _ = segmenter.segment_to_sentences(SAMPLE_TEXT[:1000])
    
    # Benchmark
    start_time = time.time()
    sentences = segmenter.segment_to_sentences(SAMPLE_TEXT)
    end_time = time.time()
    
    # Calculate performance
    processing_time = end_time - start_time
    characters_per_second = len(SAMPLE_TEXT) / processing_time
    
    return characters_per_second, len(sentences)


def main():
    """Run the benchmark and display results."""
    print("ONNX Optimization Levels Benchmark\n")
    
    # Test scikit-learn (no ONNX)
    sklearn_speed, sentence_count = benchmark_segmentation(None)
    print(f"scikit-learn: {sklearn_speed:.2f} chars/sec, {sentence_count} sentences")
    
    # Test different ONNX optimization levels
    optimization_levels = [
        (0, "No optimization"),
        (1, "Basic optimizations"),
        (2, "Extended optimizations"),
        (3, "All optimizations")
    ]
    
    results = []
    for level, description in optimization_levels:
        speed, _ = benchmark_segmentation(level)
        speedup = speed / sklearn_speed
        results.append((level, description, speed, speedup))
        print(f"ONNX Level {level} ({description}): {speed:.2f} chars/sec, {speedup:.2f}x speedup")
    
    # Find the best optimization level
    best_level = max(results, key=lambda x: x[2])
    print(f"\nBest optimization level: Level {best_level[0]} ({best_level[1]})")
    print(f"  Speed: {best_level[2]:.2f} chars/sec")
    print(f"  Speedup vs scikit-learn: {best_level[3]:.2f}x")


if __name__ == "__main__":
    main()