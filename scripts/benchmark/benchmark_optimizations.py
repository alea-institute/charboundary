#!/usr/bin/env python3
"""
Script to benchmark the performance improvements of the optimized text segmentation.
"""

import time
import os
import json
from charboundary.segmenters import TextSegmenter

def benchmark_optimizations():
    """Benchmark the performance improvements of the optimized text segmentation."""
    print("Creating and training a segmenter...")
    
    # Simple training data with a mix of sentence and paragraph boundaries
    train_data = [
        "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
        "Here is a paragraph.<|sentence|> It contains multiple sentences.<|sentence|> And another one.<|sentence|><|paragraph|>",
        "Dr. Smith visited Washington D.C. last week.<|sentence|> He met with Prof. Johnson.<|sentence|><|paragraph|>",
        "The cat sat on the mat.<|sentence|> The dog barked loudly.<|sentence|> Both animals were happy.<|sentence|><|paragraph|>"
    ] * 10
    
    # Create and train a segmenter with optimized code
    segmenter = TextSegmenter()
    metrics = segmenter.train(
        data=train_data,
        model_params={"n_estimators": 32, "max_depth": 8},
        sample_rate=1.0,
        left_window=5,
        right_window=5
    )
    
    # Create test texts of different sizes
    sizes = [100, 500, 1000, 5000, 10000]
    test_texts = {}
    
    base_text = """
    Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson at 2:30 p.m.
    The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial segregation 
    in public schools was unconstitutional. This landmark decision changed U.S. history.
    
    The cat sat on the mat. The dog barked loudly. Both animals were happy at the farm.
    Mr. Johnson worked at a company called ABC, Inc. which was founded in 1985. 
    It specializes in (1) hardware manufacturing, (2) software development, and (3) cloud services.
    """
    
    # Generate texts of different sizes
    for size in sizes:
        repeats = max(1, int(size / len(base_text)) + 1)
        test_texts[size] = base_text * repeats
        # Trim to exactly the desired size
        test_texts[size] = test_texts[size][:size]
        print(f"Created test text of size {len(test_texts[size])} characters")
    
    print("\nBenchmarking inference methods...")
    results = {}
    
    methods = [
        ("segment_text", lambda text: segmenter.segment_text(text)),
        ("segment_to_sentences", lambda text: segmenter.segment_to_sentences(text)),
        ("segment_to_paragraphs", lambda text: segmenter.segment_to_paragraphs(text))
    ]
    
    for size, text in test_texts.items():
        results[size] = {}
        print(f"\nTesting with text size {size} characters:")
        
        for method_name, method_func in methods:
            # Warm up
            method_func(text)
            
            # Measure performance
            iterations = max(1, 10000 // size)  # Adjust iterations based on text size
            iterations = min(iterations, 5)     # Cap at 5 iterations for large texts
            
            start_time = time.time()
            for _ in range(iterations):
                method_func(text)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            chars_per_second = size / avg_time
            
            results[size][method_name] = {
                "avg_time_ms": avg_time * 1000,
                "chars_per_second": chars_per_second
            }
            
            print(f"  {method_name}: {avg_time*1000:.2f}ms ({chars_per_second:.2f} chars/sec)")
    
    # Save results
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to benchmarks/optimization_results.json")
    
    # Print a summary
    print("\nPerformance summary:")
    for method_name, _ in methods:
        print(f"\n{method_name}:")
        for size in sizes:
            data = results[size][method_name]
            print(f"  {size} chars: {data['avg_time_ms']:.2f}ms ({data['chars_per_second']:.2f} chars/sec)")

if __name__ == "__main__":
    benchmark_optimizations()