# ONNX Model Directory

This directory stores ONNX versions of the charboundary models. These models provide faster inference compared to the standard scikit-learn models.

The ONNX models are not stored directly in the git repository due to their size. Instead, they are downloaded automatically when needed from the GitHub repository.

## Available Models

- `small_model.onnx`: A lightweight model with 19 features (5 token context window, 32 trees)
- `medium_model.onnx`: The default model with 21 features (7 token context window, 64 trees)
- `large_model.onnx`: The most accurate model with 27 features (9 token context window, 128 trees)

## Automatic Download

When you use functions like `get_small_onnx_segmenter()`, the library will:
1. Check if the ONNX model exists in this directory
2. If not, attempt to download it from GitHub
3. If the download fails, try to convert the standard model to ONNX

You can also manually download models using:

```python
from charboundary import download_onnx_model

# Download a specific model (small, medium, or large)
download_onnx_model("large", force=True)  # Force re-download
```

## Manual Conversion

If you prefer to convert models manually, use the conversion scripts:

```bash
# Convert all built-in models
python scripts/convert_models_to_onnx.py

# Convert a specific model
python scripts/convert_model_to_onnx.py --input model.skops.xz --output model.onnx
```

## Optimization Levels

ONNX models support different optimization levels (0-3):
- Level 0: No optimization (debugging)
- Level 1: Basic optimizations (default)
- Level 2: Extended optimizations (recommended for most cases)
- Level 3: All optimizations (best for large models)

See the main documentation for details on using optimization levels.