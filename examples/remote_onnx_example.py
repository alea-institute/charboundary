"""
Example of using ONNX models with automatic downloading.

This example demonstrates how to:
1. Use the built-in functions to get ONNX segmenters
2. Have models automatically downloaded if not available locally
3. Compare performance between scikit-learn and ONNX
"""

import time

from charboundary import (
    get_small_segmenter,
    get_default_segmenter,
    get_large_segmenter,
    get_small_onnx_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter
)


def benchmark_segmenter(segmenter, text, iterations=5):
    """Benchmark a segmenter on a given text."""
    # Warmup
    segmenter.segment_to_sentences(text)
    
    # Timed iterations
    start_time = time.time()
    for _ in range(iterations):
        sentences = segmenter.segment_to_sentences(text)
    end_time = time.time()
    
    elapsed = end_time - start_time
    chars_per_sec = len(text) * iterations / elapsed
    
    return {
        "elapsed": elapsed,
        "iterations": iterations,
        "chars_per_sec": chars_per_sec,
        "num_sentences": len(sentences)
    }


def main():
    print("Remote ONNX Model Example")
    print("-----------------------")
    
    # Sample text for benchmarking
    text = """
    The court held in Brown v. Board of Education, 347 U.S. 483 (1954), that racial segregation
    was unconstitutional. This landmark decision changed the course of history.
    
    Plaintiff referenced Johnson v. Smith, 123 F.Supp.2d 456 (2022), which stated that "courts
    must review all available evidence." The attorney noted that the expert, Dr. Jones Ph.D.,
    disagreed with this assessment. She argued, "The precedent is clear on this matter!"
    
    Section 2(a)(i) requires timely filing. Section 2(b) states exceptions apply in cases of
    unavoidable delay. The defendant argued his documents were filed in accordance with Sec. 2(a).
    
    The contract requires: (1) timely payment; (2) quality deliverables; and (3) written notice of
    termination.
    """
    
    print("\nLoading regular (scikit-learn) models...")
    print("This will download models if they're not available locally")
    
    # Load regular segmenters
    small_segmenter = get_small_segmenter()
    medium_segmenter = get_default_segmenter()
    large_segmenter = get_large_segmenter()
    
    print("\nLoading ONNX models...")
    print("This will download models if they're not available locally")
    
    # Load ONNX segmenters - these will be downloaded if not available locally
    small_onnx = get_small_onnx_segmenter()
    medium_onnx = get_medium_onnx_segmenter()
    large_onnx = get_large_onnx_segmenter()
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    iterations = 10
    
    # Small model benchmarks
    print("\nSmall model:")
    small_results = benchmark_segmenter(small_segmenter, text, iterations)
    small_onnx_results = benchmark_segmenter(small_onnx, text, iterations)
    print(f"  scikit-learn: {small_results['chars_per_sec']:.0f} chars/sec")
    print(f"  ONNX:         {small_onnx_results['chars_per_sec']:.0f} chars/sec")
    print(f"  Speedup:      {small_onnx_results['chars_per_sec']/small_results['chars_per_sec']:.2f}x")
    
    # Medium model benchmarks
    print("\nMedium model:")
    medium_results = benchmark_segmenter(medium_segmenter, text, iterations)
    medium_onnx_results = benchmark_segmenter(medium_onnx, text, iterations)
    print(f"  scikit-learn: {medium_results['chars_per_sec']:.0f} chars/sec")
    print(f"  ONNX:         {medium_onnx_results['chars_per_sec']:.0f} chars/sec")
    print(f"  Speedup:      {medium_onnx_results['chars_per_sec']/medium_results['chars_per_sec']:.2f}x")
    
    # Large model benchmarks
    print("\nLarge model:")
    large_results = benchmark_segmenter(large_segmenter, text, iterations)
    large_onnx_results = benchmark_segmenter(large_onnx, text, iterations)
    print(f"  scikit-learn: {large_results['chars_per_sec']:.0f} chars/sec")
    print(f"  ONNX:         {large_onnx_results['chars_per_sec']:.0f} chars/sec")
    print(f"  Speedup:      {large_onnx_results['chars_per_sec']/large_results['chars_per_sec']:.2f}x")


if __name__ == "__main__":
    main()