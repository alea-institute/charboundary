#!/usr/bin/env bash


uv run python3 scripts/train_small_model.py
uv run python3 scripts/train_medium_model.py
uv run python3 scripts/train_large_model.py
