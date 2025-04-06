"""
CharBoundary: A modular library for segmenting text into sentences and paragraphs.

The library provides tools for segmenting text into sentences and paragraphs,
including character-level boundary detection and span information.
"""

import os
import sys
import warnings

# Import directly from submodules
from charboundary.constants import (
    SENTENCE_TAG,
    PARAGRAPH_TAG,
)
from charboundary.encoders import CharacterEncoder
from charboundary.features import FeatureExtractor
from charboundary.segmenters import (
    TextSegmenter,
    SegmenterConfig,
    TextSegmenterProtocol,
    MetricsResult,
)
from charboundary.utils import load_jsonl, save_jsonl

# Try to import ONNX support
try:
    from charboundary.onnx_support import check_onnx_available

    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

# Import remote model helpers
from charboundary.remote_models import (
    get_model_path,
    download_model,
    get_resource_dir,
    ensure_onnx_dir,
)


# Define helper functions for getting pre-trained models
def get_onnx_dir():
    """Get the ONNX directory path."""
    return ensure_onnx_dir()


def get_default_segmenter() -> TextSegmenter:
    """
    Get the default pre-trained medium-sized text segmenter.

    This loads the medium model, which offers a good balance between
    accuracy and resource usage. For smaller footprint, use the small model,
    and for potentially higher accuracy, use the large model.

    Note: The medium model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the medium model
    """

    # Try to get model path
    model_path = get_model_path("medium", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download medium model.\n"
            f"You can manually download the medium model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/medium_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def get_small_segmenter() -> TextSegmenter:
    """
    Get the small pre-trained text segmenter.

    The small model has a smaller memory footprint and faster inference
    but may have slightly lower accuracy than the medium or large models.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the small model
    """

    # Try to get model path
    model_path = get_model_path("small", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download small model.\n"
            f"You can manually download the small model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/small_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def get_large_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter.

    The large model has the highest accuracy but also the largest memory footprint
    and may be slower for inference than the small or medium models.

    Note: The large model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 62MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model
    """

    # Try to get model path
    model_path = get_model_path("large", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download large model.\n"
            f"You can manually download the large model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/large_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def download_onnx_model(model_name: str, force: bool = False) -> str:
    """
    Download an ONNX model if not available locally.

    Args:
        model_name: Name of the model ('small', 'medium', or 'large')
        force: Whether to force download even if the model exists locally

    Returns:
        str: Path to the downloaded model
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support is not available. "
            "Install it with: pip install charboundary[onnx]"
        )

    # Ensure model name is valid
    if model_name not in ["small", "medium", "large"]:
        raise ValueError(
            f"Unknown model name: {model_name}. Use 'small', 'medium', or 'large'."
        )

    # Get model file name
    model_file = f"{model_name}_model.onnx"

    # Download the model
    path = download_model(model_file, force=force)

    if path is None:
        raise RuntimeError(f"Failed to download {model_file}")

    return path


def get_onnx_segmenter(model_name: str = "small") -> TextSegmenter:
    """
    Get a pre-trained text segmenter that uses ONNX for inference.

    Args:
        model_name: Name of the model ('small', 'medium', or 'large')

    Returns:
        TextSegmenter: A pre-trained text segmenter using ONNX
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support is not available. "
            "Install it with: pip install charboundary[onnx]"
        )

    # Get the regular segmenter based on model name
    if model_name == "small":
        segmenter = get_small_segmenter()
    elif model_name == "medium":
        segmenter = get_default_segmenter()
    elif model_name == "large":
        segmenter = get_large_segmenter()
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Use 'small', 'medium', or 'large'."
        )

    # Check if the ONNX model is available
    onnx_path = get_model_path(model_name, use_onnx=True, download=True)

    if onnx_path is None:
        # If not available, try to convert the model to ONNX
        if segmenter.model.feature_count is None:
            # Infer feature count from model
            if (
                hasattr(segmenter.model, "selected_feature_indices")
                and segmenter.model.selected_feature_indices
            ):
                segmenter.model.feature_count = len(
                    segmenter.model.selected_feature_indices
                )
            else:
                # Default values
                feature_counts = {"small": 19, "medium": 21, "large": 25}
                segmenter.model.feature_count = feature_counts.get(model_name, 32)

        # Convert and save model
        segmenter.to_onnx()
        onnx_file = os.path.join(ensure_onnx_dir(), f"{model_name}_model.onnx")
        segmenter.save_onnx(onnx_file)
    else:
        # Load the ONNX model
        segmenter.load_onnx(onnx_path)

    # Enable ONNX inference
    segmenter.enable_onnx(True)

    return segmenter


def get_small_onnx_segmenter() -> TextSegmenter:
    """
    Get the small pre-trained text segmenter with ONNX inference.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the small model with ONNX
    """
    return get_onnx_segmenter("small")


def get_medium_onnx_segmenter() -> TextSegmenter:
    """
    Get the medium pre-trained text segmenter with ONNX inference.

    Note: The medium ONNX model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 33MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the medium model with ONNX
    """
    return get_onnx_segmenter("medium")


def get_large_onnx_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter with ONNX inference.

    Note: The large ONNX model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 188MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model with ONNX
    """
    return get_onnx_segmenter("large")


# Export the model loading functions as part of the public API
__all__ = [
    "SENTENCE_TAG",
    "PARAGRAPH_TAG",
    "CharacterEncoder",
    "FeatureExtractor",
    "TextSegmenter",
    "SegmenterConfig",
    "TextSegmenterProtocol",
    "MetricsResult",
    "load_jsonl",
    "save_jsonl",
    "get_default_segmenter",
    "get_small_segmenter",
    "get_large_segmenter",
    "get_onnx_segmenter",
    "get_small_onnx_segmenter",
    "get_medium_onnx_segmenter",
    "get_large_onnx_segmenter",
    "download_onnx_model",
    "cli",
    "get_onnx_dir",
]

# Check if ONNX is available and warn if needed
if not ONNX_AVAILABLE:
    warnings.warn(
        "ONNX support is not available. Install the optional dependencies with: "
        "pip install charboundary[onnx]"
    )


# Create a convenience function to run the CLI
def cli():
    """Run the charboundary command-line interface."""
    from charboundary.cli.main import main

    sys.exit(main())


# Create a function to load the default model
def get_default_segmenter() -> TextSegmenter:
    """
    Get the default pre-trained medium-sized text segmenter.

    This loads the medium model, which offers a good balance between
    accuracy and resource usage. For smaller footprint, use the small model,
    and for potentially higher accuracy, use the large model.

    Note: The medium model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the medium model
    """

    # Try to get model path
    model_path = get_model_path("medium", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download medium model.\n"
            f"You can manually download the medium model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/medium_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def get_small_segmenter() -> TextSegmenter:
    """
    Get the small pre-trained text segmenter.

    The small model has a smaller memory footprint and faster inference
    but may have slightly lower accuracy than the medium or large models.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the small model
    """

    # Try to get model path
    model_path = get_model_path("small", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download small model.\n"
            f"You can manually download the small model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/small_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def get_large_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter.

    The large model has the highest accuracy but also the largest memory footprint
    and may be slower for inference than the small or medium models.

    Note: The large model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 62MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model
    """

    # Try to get model path
    model_path = get_model_path("large", use_onnx=False, download=True)

    if model_path is None:
        raise RuntimeError(
            f"Failed to load or download large model.\n"
            f"You can manually download the large model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/large_model.skops.xz\n"
            f"and place it in: {get_resource_dir()}/"
        )

    # Load the model
    return TextSegmenter.load(model_path, trust_model=True)


def download_onnx_model(model_name: str, force: bool = False) -> str:
    """
    Download an ONNX model if not available locally.

    Args:
        model_name: Name of the model ('small', 'medium', or 'large')
        force: Whether to force download even if the model exists locally

    Returns:
        str: Path to the downloaded model
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support is not available. "
            "Install it with: pip install charboundary[onnx]"
        )

    # Ensure model name is valid
    if model_name not in ["small", "medium", "large"]:
        raise ValueError(
            f"Unknown model name: {model_name}. Use 'small', 'medium', or 'large'."
        )

    # Get model file name
    model_file = f"{model_name}_model.onnx"

    # Download the model
    path = download_model(model_file, force=force)

    if path is None:
        raise RuntimeError(f"Failed to download {model_file}")

    return path


def get_onnx_segmenter(model_name: str = "small") -> TextSegmenter:
    """
    Get a pre-trained text segmenter that uses ONNX for inference.

    Args:
        model_name: Name of the model ('small', 'medium', or 'large')

    Returns:
        TextSegmenter: A pre-trained text segmenter using ONNX
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support is not available. "
            "Install it with: pip install charboundary[onnx]"
        )

    # Get the regular segmenter based on model name
    if model_name == "small":
        segmenter = get_small_segmenter()
    elif model_name == "medium":
        segmenter = get_default_segmenter()
    elif model_name == "large":
        segmenter = get_large_segmenter()
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Use 'small', 'medium', or 'large'."
        )

    # Check if the ONNX model is available
    onnx_path = get_model_path(model_name, use_onnx=True, download=True)

    if onnx_path is None:
        # If not available, try to convert the model to ONNX
        if segmenter.model.feature_count is None:
            # Infer feature count from model
            if (
                hasattr(segmenter.model, "selected_feature_indices")
                and segmenter.model.selected_feature_indices
            ):
                segmenter.model.feature_count = len(
                    segmenter.model.selected_feature_indices
                )
            else:
                # Default values
                feature_counts = {"small": 19, "medium": 21, "large": 25}
                segmenter.model.feature_count = feature_counts.get(model_name, 32)

        # Convert and save model
        segmenter.to_onnx()
        onnx_file = os.path.join(ensure_onnx_dir(), f"{model_name}_model.onnx")
        segmenter.save_onnx(onnx_file)
    else:
        # Load the ONNX model
        segmenter.load_onnx(onnx_path)

    # Enable ONNX inference
    segmenter.enable_onnx(True)

    return segmenter


def get_small_onnx_segmenter() -> TextSegmenter:
    """
    Get the small pre-trained text segmenter with ONNX inference.

    Returns:
        TextSegmenter: A pre-trained text segmenter using the small model with ONNX
    """
    return get_onnx_segmenter("small")


def get_medium_onnx_segmenter() -> TextSegmenter:
    """
    Get the medium pre-trained text segmenter with ONNX inference.

    Note: The medium ONNX model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 33MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the medium model with ONNX
    """
    return get_onnx_segmenter("medium")


def get_large_onnx_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter with ONNX inference.

    Note: The large ONNX model is not included in the PyPI package to keep the
    package size reasonable. It will be automatically downloaded from GitHub
    when first used (approximately 188MB download).

    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model with ONNX
    """
    return get_onnx_segmenter("large")


__version__ = "0.5.0"
