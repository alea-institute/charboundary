"""
CharBoundary: A modular library for segmenting text into sentences and paragraphs.
"""

import os
import json
import sys

# Import directly from submodules
from charboundary.constants import (
    SENTENCE_TAG,
    PARAGRAPH_TAG,
)
from charboundary.encoders import CharacterEncoder
from charboundary.features import FeatureExtractor
from charboundary.segmenters import TextSegmenter
from charboundary.utils import load_jsonl, save_jsonl

# Export the model loading functions as part of the public API
__all__ = [
    'SENTENCE_TAG', 'PARAGRAPH_TAG',
    'CharacterEncoder', 'FeatureExtractor', 'TextSegmenter',
    'load_jsonl', 'save_jsonl',
    'get_default_segmenter', 'get_small_segmenter', 'get_large_segmenter',
    'cli'
]

# Create a convenience function to run the CLI
def cli():
    """Run the charboundary command-line interface."""
    from charboundary.cli.main import main
    import sys
    sys.exit(main())

# Create a function to load the default model
def get_default_segmenter() -> TextSegmenter:
    """
    Get the default pre-trained medium-sized text segmenter.
    
    This loads the medium model, which offers a good balance between
    accuracy and resource usage. For smaller footprint, use the small model,
    and for potentially higher accuracy, use the large model.
    
    Returns:
        TextSegmenter: A pre-trained text segmenter using the medium model
    """
    # Import needed types
    from charboundary.models import BinaryRandomForestModel
    from charboundary.segmenters import SegmenterConfig
    from skops.io import get_untrusted_types
    
    # Check possible model paths (compressed or uncompressed)
    # Medium model is the default
    model_base_name = 'medium_model.skops'
    
    # Directly use file paths instead of context managers
    package_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(package_dir, "resources")
    
    # Try uncompressed and compressed versions
    model_paths = [
        os.path.join(resource_dir, f"{model_base_name}"),
        os.path.join(resource_dir, f"{model_base_name}.xz"),
        os.path.join(resource_dir, f"{model_base_name}.lzma"),
    ]
    
    # Try each path until one works
    last_error = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                return TextSegmenter.load(model_path, trust_model=True)
            except Exception as e:
                last_error = e
    
    # If we get here, no paths worked
    raise RuntimeError(f"Failed to load default model. Last error: {last_error if last_error else 'No valid model paths found'}")


def get_small_segmenter() -> TextSegmenter:
    """
    Get the small pre-trained text segmenter.
    
    The small model has a smaller memory footprint and faster inference
    but may have slightly lower accuracy than the medium or large models.
    
    Returns:
        TextSegmenter: A pre-trained text segmenter using the small model
    """
    # Import needed types
    from charboundary.models import BinaryRandomForestModel
    from charboundary.segmenters import SegmenterConfig
    from skops.io import get_untrusted_types
    
    # Small model
    model_base_name = 'small_model.skops'
    
    # Directly use file paths instead of context managers
    package_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(package_dir, "resources")
    
    # Try uncompressed and compressed versions
    model_paths = [
        os.path.join(resource_dir, f"{model_base_name}"),
        os.path.join(resource_dir, f"{model_base_name}.xz"),
        os.path.join(resource_dir, f"{model_base_name}.lzma"),
    ]
    
    # Try each path until one works
    last_error = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                return TextSegmenter.load(model_path, trust_model=True)
            except Exception as e:
                last_error = e
    
    # If we get here, no paths worked
    raise RuntimeError(f"Failed to load small model. Last error: {last_error if last_error else 'No valid model paths found'}")


def get_large_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter.
    
    The large model has the highest accuracy but also the largest memory footprint
    and may be slower for inference than the small or medium models.
    
    If the large model is not available locally, this function will attempt to download it
    from the GitHub repository.
    
    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model
    """
    # Import needed types
    import urllib.request
    from charboundary.models import BinaryRandomForestModel
    from charboundary.segmenters import SegmenterConfig
    from skops.io import get_untrusted_types
    
    # Large model
    model_base_name = 'large_model.skops'
    
    # Directly use file paths instead of context managers
    package_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(package_dir, "resources")
    
    # Try uncompressed and compressed versions
    model_paths = [
        os.path.join(resource_dir, f"{model_base_name}"),
        os.path.join(resource_dir, f"{model_base_name}.xz"),
        os.path.join(resource_dir, f"{model_base_name}.lzma"),
    ]
    
    # Try each path until one works
    last_error = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                return TextSegmenter.load(model_path, trust_model=True)
            except Exception as e:
                last_error = e
    
    # If no local model found, try to download it
    print("Large model not found locally. Attempting to download from GitHub...")
    try:
        # Define GitHub URL for the compressed model
        github_url = "https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/large_model.skops.xz"
        
        # Choose the .xz path version for download
        download_path = os.path.join(resource_dir, "large_model.skops.xz")
        
        # Create resources directory if it doesn't exist
        os.makedirs(resource_dir, exist_ok=True)
        
        # Download the model
        print(f"Downloading from {github_url}...")
        urllib.request.urlretrieve(github_url, download_path)
        print(f"Download complete. Model saved to {download_path}")
        
        # Try to load the downloaded model
        return TextSegmenter.load(download_path, trust_model=True)
    except Exception as e:
        download_error = str(e)
        # If we get here, downloading or loading the downloaded model failed
        raise RuntimeError(
            f"Failed to load or download large model.\n"
            f"Local error: {last_error if last_error else 'No valid model paths found'}\n"
            f"Download error: {download_error}\n\n"
            f"You can manually download the large model from:\n"
            f"https://github.com/alea-institute/charboundary/raw/refs/heads/main/charboundary/resources/large_model.skops.xz\n"
            f"and place it in: {resource_dir}/"
        )


__version__ = "0.3.0"
