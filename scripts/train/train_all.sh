#!/usr/bin/env bash

cd "$(dirname "$0")/../.."  # Go to project root

uv run python3 scripts/train/train_small_model.py
uv run python3 scripts/train/train_medium_model.py
uv run python3 scripts/train/train_large_model.py
