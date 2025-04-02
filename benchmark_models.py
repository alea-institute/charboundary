#!/usr/bin/env python
"""
Benchmark script to compare performance of scikit-learn (skops) and ONNX models.
"""

import time
import json
import gzip
import pandas as pd
from tqdm import tqdm

# Import charboundary modules
from charboundary import (
    get_small_segmenter, get_default_segmenter, get_large_segmenter,
    get_small_onnx_segmenter, get_medium_onnx_segmenter, get_large_onnx_segmenter,
    load_jsonl
)

# Check if ONNX is available
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

def load_data(file_path, num_samples=1000):
    """Load sample data from a JSONL file."""
    samples = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            samples.append(data['text'])
    return samples

def benchmark_model(segmenter, texts, name, warmup=3):
    """Benchmark a model's performance on segmenting text."""
    # Warmup runs to load models and initialize
    for _ in range(warmup):
        segmenter.segment_text(texts[0])
    
    # Start timing
    start_time = time.time()
    for text in texts:
        segmenter.segment_text(text)
    total_time = time.time() - start_time
    
    return {
        'model': name,
        'total_time': total_time,
        'avg_time_per_text': total_time / len(texts),
        'texts_per_second': len(texts) / total_time
    }

def run_benchmarks(num_samples=1000):
    """Run all benchmarks and collect results."""
    print(f"Loading {num_samples} text samples for benchmarking...")
    texts = load_data('data/train.jsonl.gz', num_samples)
    print(f"Loaded {len(texts)} samples")
    
    results = []
    
    # Benchmark scikit-learn models
    print("\nBenchmarking scikit-learn (skops) models:")
    
    print("Loading and benchmarking small model...")
    small_segmenter = get_small_segmenter()
    results.append(benchmark_model(small_segmenter, texts, "Small (sklearn)"))
    
    print("Loading and benchmarking medium model...")
    medium_segmenter = get_default_segmenter()
    results.append(benchmark_model(medium_segmenter, texts, "Medium (sklearn)"))
    
    print("Loading and benchmarking large model...")
    large_segmenter = get_large_segmenter()
    results.append(benchmark_model(large_segmenter, texts, "Large (sklearn)"))
    
    # Benchmark ONNX models if available
    if ONNX_AVAILABLE:
        print("\nBenchmarking ONNX models:")
        
        print("Loading and benchmarking small ONNX model...")
        small_onnx_segmenter = get_small_onnx_segmenter()
        results.append(benchmark_model(small_onnx_segmenter, texts, "Small (ONNX)"))
        
        print("Loading and benchmarking medium ONNX model...")
        medium_onnx_segmenter = get_medium_onnx_segmenter()
        results.append(benchmark_model(medium_onnx_segmenter, texts, "Medium (ONNX)"))
        
        print("Loading and benchmarking large ONNX model...")
        large_onnx_segmenter = get_large_onnx_segmenter()
        results.append(benchmark_model(large_onnx_segmenter, texts, "Large (ONNX)"))
    else:
        print("\nONNX support is not available. Install it with: pip install charboundary[onnx]")
    
    return results

def main():
    """Main function to run benchmarks and display results."""
    print("CharBoundary Model Benchmark")
    print("===========================")
    
    # Run the benchmarks
    results = run_benchmarks(num_samples=1000)
    
    # Convert to DataFrame for easier display
    df = pd.DataFrame(results)
    
    # Sort by model size and type
    model_order = {
        "Small (sklearn)": 0, 
        "Small (ONNX)": 1,
        "Medium (sklearn)": 2, 
        "Medium (ONNX)": 3,
        "Large (sklearn)": 4, 
        "Large (ONNX)": 5
    }
    df['order'] = df['model'].map(model_order)
    df = df.sort_values('order').drop('order', axis=1)
    
    # Format for display
    df['avg_time_ms'] = df['avg_time_per_text'] * 1000
    df['texts_per_second'] = df['texts_per_second'].round(2)
    
    # Calculate speedup for ONNX models
    if ONNX_AVAILABLE:
        sklearn_times = {
            'Small': df[df['model'] == 'Small (sklearn)']['avg_time_per_text'].values[0],
            'Medium': df[df['model'] == 'Medium (sklearn)']['avg_time_per_text'].values[0],
            'Large': df[df['model'] == 'Large (sklearn)']['avg_time_per_text'].values[0]
        }
        
        onnx_times = {
            'Small': df[df['model'] == 'Small (ONNX)']['avg_time_per_text'].values[0],
            'Medium': df[df['model'] == 'Medium (ONNX)']['avg_time_per_text'].values[0],
            'Large': df[df['model'] == 'Large (ONNX)']['avg_time_per_text'].values[0]
        }
        
        for size in ['Small', 'Medium', 'Large']:
            df.loc[df['model'] == f'{size} (ONNX)', 'speedup'] = sklearn_times[size] / onnx_times[size]
    
    # Display formatted results
    print("\nBenchmark Results:")
    print("=================")
    
    display_df = df[['model', 'avg_time_ms', 'texts_per_second']].copy()
    display_df['avg_time_ms'] = display_df['avg_time_ms'].round(2)
    
    if 'speedup' in df.columns:
        display_df = pd.concat([display_df, df[['model', 'speedup']]], axis=1)
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]  # Remove duplicate columns
        display_df['speedup'] = display_df['speedup'].round(2)
        display_df.loc[display_df['model'].str.contains('sklearn'), 'speedup'] = '-'
    
    # Print as table
    print("\n" + display_df.to_string(index=False))
    
    # Save results to JSON for future reference
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to benchmark_results.json")

if __name__ == "__main__":
    main()