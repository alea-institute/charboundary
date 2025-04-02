# Charboundary Scripts

This directory contains utility scripts for the charboundary library.

## Benchmark Scripts

The benchmark scripts have been consolidated for better organization and easier usage. There are now two main benchmark scripts:

### `benchmark_onnx_models.py`

Comprehensive ONNX benchmarking script that consolidates all ONNX-related benchmark functionality into a single tool. This script replaces several older benchmark scripts and provides multiple benchmark modes:

1. Benchmark a specific model with different ONNX optimization levels (0-3)
2. Benchmark all built-in models with their optimal optimization levels
3. Benchmark a custom model file with ONNX vs scikit-learn

Usage:

```bash
# Benchmark all built-in models with optimal optimization levels
python benchmark_onnx_models.py --built-in-models

# Benchmark a specific built-in model with all optimization levels
python benchmark_onnx_models.py --model-name small
python benchmark_onnx_models.py --model-name medium
python benchmark_onnx_models.py --model-name large

# Benchmark a specific model file with optimization levels
python benchmark_onnx_models.py --model-file path/to/model.skops.xz

# Additional parameters
python benchmark_onnx_models.py --runs 50 --batch-size 1000
```

Key features of this consolidated script:

- Tests all four ONNX optimization levels (0-3)
- Automatically validates that prediction quality is identical between sklearn and ONNX
- Provides detailed performance metrics and reports the optimal optimization level
- Handles all built-in models (small, medium, large)
- Supports custom model files
- Generates formatted tables for documentation

### `benchmark_optimizations.py`

Benchmarks the performance improvements of the optimized text segmentation without ONNX. This script focuses on measuring the pure Python processing performance of the segmentation process.

```bash
python benchmark_optimizations.py
```

This script:
- Tests various text sizes (100, 500, 1000, 5000, 10000 characters)
- Measures performance of all segmentation methods
- Calculates characters processed per second
- Generates a JSON report with detailed metrics

## Model Training Scripts

### `train_small_model.py`, `train_medium_model.py`, `train_large_model.py`

These scripts train the small, medium, and large models respectively.

### `train_all.sh`

Trains all models in sequence.

## Utility Scripts

### `compress_model.py`

Compresses model files.

### `profile_model.py`

Profiles model performance.

## Legacy Benchmark Scripts

Older benchmark scripts have been archived in the `archive` directory. The functionality of these scripts is now consolidated in `benchmark_onnx_models.py`.