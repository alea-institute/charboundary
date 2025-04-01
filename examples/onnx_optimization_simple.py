#!/usr/bin/env python
"""
Simple example demonstrating ONNX optimization levels with CharBoundary.

This example shows how to use different ONNX optimization levels with
the CharBoundary library and measures their performance.
"""

import time
import sys
from charboundary import TextSegmenter
from charboundary.segmenters import SegmenterConfig
from charboundary import get_large_segmenter

# Sample text for benchmarking (repeated to create a larger corpus)
TEST_TEXT = """
The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that 
racial segregation in public schools was unconstitutional. This landmark 
decision changed the course of U.S. history.

Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson 
at 2:30 p.m. They discussed Mr. Brown's case regarding Section 1.2.3 of 
the tax code.

"This is a direct quote," said the author. "It contains multiple sentences. 
And it spans multiple lines." The audience nodded in agreement.
""" * 100


def benchmark_segmenter(segmenter, name: str) -> float:
    """Benchmark a segmenter and return processing speed."""
    # Warmup
    _ = segmenter.segment_to_sentences(TEST_TEXT[:1000])
    
    # Benchmark
    start_time = time.time()
    sentences = segmenter.segment_to_sentences(TEST_TEXT)
    end_time = time.time()
    
    # Calculate processing speed
    elapsed = end_time - start_time
    chars_per_sec = len(TEST_TEXT) / elapsed
    
    print(f"{name}: {chars_per_sec:.0f} chars/sec, {len(sentences)} sentences, {elapsed:.3f} seconds")
    
    return chars_per_sec
    

def main():
    """Run benchmarks with different ONNX optimization levels."""
    print("\nCharBoundary ONNX Optimization Levels Example\n")
    
    # Check if ONNX is available
    try:
        from charboundary.onnx_support import check_onnx_available
        onnx_available = check_onnx_available()
    except ImportError:
        onnx_available = False
    
    if not onnx_available:
        print("ONNX is not available. Please install ONNX dependencies:")
        print("pip install charboundary[onnx]")
        sys.exit(1)
    
    # Load the large model (best for showing optimization benefits)
    base_segmenter = get_large_segmenter()
    
    # 1. Benchmark scikit-learn (no ONNX)
    base_segmenter.config.use_onnx = False
    sklearn_speed = benchmark_segmenter(base_segmenter, "scikit-learn")
    
    # Initialize feature count (needed for ONNX conversion)
    base_segmenter.model.feature_count = 27  # Large model has 27 features
    
    # Ensure ONNX model is created
    base_segmenter.model.to_onnx()
    
    # 2. Create and benchmark segmenters with different optimization levels
    optimization_levels = [
        (0, "No optimization"),
        (1, "Basic optimizations"),
        (2, "Extended optimizations"),
        (3, "All optimizations")
    ]
    
    best_speed = sklearn_speed
    best_level = "scikit-learn"
    
    for level, description in optimization_levels:
        # Configure ONNX with the specified optimization level
        base_segmenter.model.enable_onnx(True, optimization_level=level)
        
        # Benchmark
        speed = benchmark_segmenter(base_segmenter, f"ONNX Level {level}: {description}")
        
        # Calculate speedup
        speedup = speed / sklearn_speed
        print(f"  Speedup vs scikit-learn: {speedup:.2f}x\n")
        
        # Track the best performer
        if speed > best_speed:
            best_speed = speed
            best_level = f"ONNX Level {level}"
    
    # Print summary
    print("\nSummary:")
    print(f"Best performance: {best_level}")
    print(f"Speedup vs scikit-learn: {best_speed/sklearn_speed:.2f}x")
    print("\nRecommendation:")
    print("- For small/medium models: Use optimization level 2 (extended optimizations)")
    print("- For large models: Use optimization level 3 (all optimizations)")
    print("- Memory-constrained environments: Use optimization level 1")


if __name__ == "__main__":
    main()