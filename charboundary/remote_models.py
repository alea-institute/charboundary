"""
Support for remote model retrieval in charboundary.

This module provides functions to download models from remote sources
when they are not available locally.
"""

import os
import sys
import urllib.request
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger("charboundary.remote_models")

# GitHub repository information
GITHUB_REPO = "alea-institute/charboundary"
GITHUB_BRANCH = "main"

# Base URL for GitHub raw content
GITHUB_RAW_BASE = f"https://github.com/{GITHUB_REPO}/raw/refs/heads/{GITHUB_BRANCH}"

# Define model URLs (relative paths from repository root)
MODEL_URLS = {
    # Standard models
    "small_model.skops.xz": f"{GITHUB_RAW_BASE}/charboundary/resources/small_model.skops.xz",
    "medium_model.skops.xz": f"{GITHUB_RAW_BASE}/charboundary/resources/medium_model.skops.xz",
    "large_model.skops.xz": f"{GITHUB_RAW_BASE}/charboundary/resources/large_model.skops.xz",
    # ONNX models
    "small_model.onnx": f"{GITHUB_RAW_BASE}/charboundary/resources/onnx/small_model.onnx.xz",
    "medium_model.onnx": f"{GITHUB_RAW_BASE}/charboundary/resources/onnx/medium_model.onnx.xz",
    "large_model.onnx": f"{GITHUB_RAW_BASE}/charboundary/resources/onnx/large_model.onnx.xz",
}

# Model sizes in megabytes (approximate) for progress reporting
MODEL_SIZES = {
    "small_model.skops.xz": 1,  # ~1 MB
    "medium_model.skops.xz": 2,  # ~2 MB
    "large_model.skops.xz": 6,  # ~6 MB
    "small_model.onnx": 2,  # ~2 MB compressed (~5 MB uncompressed)
    "medium_model.onnx": 13,  # ~13 MB compressed (~33 MB uncompressed)
    "large_model.onnx": 75,  # ~75 MB compressed (~188 MB uncompressed)
}


def get_resource_dir() -> str:
    """
    Get the path to the resources directory.

    Returns:
        str: Path to resources directory
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(package_dir, "resources")
    return resource_dir


def get_onnx_dir() -> str:
    """
    Get the path to the ONNX resources directory.

    Returns:
        str: Path to ONNX resources directory
    """
    resource_dir = get_resource_dir()
    onnx_dir = os.path.join(resource_dir, "onnx")
    return onnx_dir


def ensure_onnx_dir() -> str:
    """
    Ensure the ONNX directory exists.

    Returns:
        str: Path to ONNX directory
    """
    onnx_dir = get_onnx_dir()
    os.makedirs(onnx_dir, exist_ok=True)
    return onnx_dir


def find_local_model(
    model_name: str, extensions: Optional[list] = None
) -> Optional[str]:
    """
    Check if a model is available locally.

    Args:
        model_name: Base name of the model (e.g., 'small_model')
        extensions: List of extensions to try (e.g., ['.skops', '.skops.xz'])
                   If None, uses ['.skops', '.skops.xz', '.skops.lzma']

    Returns:
        Optional[str]: Path to the model if found, None otherwise
    """
    if extensions is None:
        extensions = [".skops", ".skops.xz", ".skops.lzma"]

    resource_dir = get_resource_dir()

    # If model_name has an extension, use it directly
    if any(model_name.endswith(ext) for ext in extensions):
        # Check if ONNX model
        if model_name.endswith(".onnx"):
            path = os.path.join(resource_dir, "onnx", model_name)
        else:
            path = os.path.join(resource_dir, model_name)

        return path if os.path.exists(path) else None

    # Try each extension
    for ext in extensions:
        if model_name.endswith(".onnx"):
            # ONNX models are in a subdirectory
            path = os.path.join(resource_dir, "onnx", f"{model_name}{ext}")
        else:
            path = os.path.join(resource_dir, f"{model_name}{ext}")

        if os.path.exists(path):
            return path

    return None


class DownloadProgressReporter:
    """Simple progress reporter for downloads."""

    def __init__(self, total_size: int, model_name: str):
        """Initialize with total size."""
        self.total_size = total_size
        self.downloaded = 0
        self.model_name = model_name
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        """Report download progress."""
        if total_size > 0:
            self.total_size = total_size

        self.downloaded += block_size
        percent = min(100, int(self.downloaded * 100 / self.total_size))

        # Only update when percent changes to avoid too many updates
        if percent != self.last_percent and percent % 10 == 0:
            self.last_percent = percent
            sys.stdout.write(
                f"\rDownloading {self.model_name}: {percent}% ({self.downloaded / 1024 / 1024:.1f} MB)"
            )
            sys.stdout.flush()

        if percent >= 100:
            sys.stdout.write("\n")
            sys.stdout.flush()


def download_model(model_name: str, force: bool = False) -> Optional[str]:
    """
    Download a model from GitHub if not available locally.

    Args:
        model_name: Name of the model file (e.g., 'small_model.skops.xz')
        force: Whether to force download even if the model exists locally

    Returns:
        Optional[str]: Path to the downloaded model if successful, None otherwise
    """
    if model_name not in MODEL_URLS:
        logger.error(f"Unknown model: {model_name}")
        return None

    # Determine target path
    if model_name.endswith(".onnx"):
        # ONNX models go in the onnx subdirectory
        target_dir = ensure_onnx_dir()

        # Handle compressed ONNX models from GitHub
        url = MODEL_URLS[model_name]
        if url.endswith(".xz"):
            target_path = os.path.join(target_dir, model_name + ".xz")
        else:
            target_path = os.path.join(target_dir, model_name)
    else:
        target_dir = get_resource_dir()
        target_path = os.path.join(target_dir, model_name)

    # Skip download if file exists and force is not set
    if os.path.exists(target_path) and not force:
        logger.info(f"Model {model_name} already exists at {target_path}")
        return target_path

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Get download URL
    url = MODEL_URLS[model_name]

    # Estimate total size for progress reporting
    total_size = MODEL_SIZES.get(model_name, 1) * 1024 * 1024  # Convert MB to bytes

    print(f"Downloading {model_name} from {url}...")

    try:
        # Setup progress reporter
        progress_reporter = DownloadProgressReporter(total_size, model_name)

        # Download the file
        urllib.request.urlretrieve(url, target_path, progress_reporter)
        print(f"Download complete. Model saved to {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        # Clean up partial download
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass
        return None


def get_model_path(
    model_name: str, use_onnx: bool = False, download: bool = True
) -> Optional[str]:
    """
    Get the path to a model, downloading it if necessary and requested.

    Args:
        model_name: Base name of the model (e.g., 'small_model', 'small_model.skops.xz')
        use_onnx: Whether to get the ONNX version of the model
        download: Whether to download the model if not found locally

    Returns:
        Optional[str]: Path to the model if available, None otherwise
    """
    # Strip extensions if present
    base_name = model_name.split(".")[0]

    # Add appropriate extension
    if use_onnx:
        file_name = f"{base_name}_model.onnx"
    else:
        file_name = f"{base_name}_model.skops.xz"

    # Check if model exists locally
    local_path = find_local_model(file_name)

    if local_path is not None:
        return local_path

    # If not found and downloads are enabled, try to download
    if download:
        return download_model(file_name)

    return None
