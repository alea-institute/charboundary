#!/usr/bin/env bash

cd "$(dirname "$0")/../.."  # Go to project root

# Train the models
echo "=== Training models ==="
uv run python3 scripts/train/train_small_model.py
uv run python3 scripts/train/train_medium_model.py
uv run python3 scripts/train/train_large_model.py

# Convert all models to ONNX format
echo ""
echo "=== Converting models to ONNX format ==="
uv run python3 scripts/onnx_utils.py convert --all-models

echo ""
echo "=== All models trained and converted to ONNX format ==="
