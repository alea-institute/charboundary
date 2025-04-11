#!/usr/bin/env bash

export PYTHONPATH=.
uv run python scripts/upload_to_hf.py --model small --format skops
uv run python scripts/upload_to_hf.py --model small --format onnx
uv run python scripts/upload_to_hf.py --model medium --format skops
uv run python scripts/upload_to_hf.py --model medium --format onnx
uv run python scripts/upload_to_hf.py --model large --format skops
uv run python scripts/upload_to_hf.py --model large --format onnx
