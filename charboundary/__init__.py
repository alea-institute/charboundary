"""
CharBoundary: A modular library for segmenting text into sentences and paragraphs.
"""

import os
from importlib import resources
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
    model_paths = []
    
    # Medium model is the default
    model_base_name = 'medium_model.skops'
    
    if sys.version_info >= (3, 9):
        # Use new resources API for Python 3.9+
        # Try both compressed and uncompressed files
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.files('charboundary.resources').joinpath(f'{model_base_name}{ext}'))
            except Exception:
                pass
    else:
        # Fallback for older Python versions
        # Try both compressed and uncompressed files
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.path('charboundary.resources', f'{model_base_name}{ext}').__enter__())
            except Exception:
                pass
    
    # Try each path until one works
    for model_resource in model_paths:
        try:
            model_path = str(model_resource)
            return TextSegmenter.load(model_path, trust_model=True)
        except Exception as e:
            last_error = e
            continue
    
    # If we get here, no paths worked
    raise RuntimeError(f"Failed to load default model. Last error: {last_error if 'last_error' in locals() else 'No valid model paths found'}")


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
    
    # Check possible model paths (compressed or uncompressed)
    model_paths = []
    
    # Small model
    model_base_name = 'small_model.skops'
    
    if sys.version_info >= (3, 9):
        # Use new resources API for Python 3.9+
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.files('charboundary.resources').joinpath(f'{model_base_name}{ext}'))
            except Exception:
                pass
    else:
        # Fallback for older Python versions
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.path('charboundary.resources', f'{model_base_name}{ext}').__enter__())
            except Exception:
                pass
    
    # Try each path until one works
    for model_resource in model_paths:
        try:
            model_path = str(model_resource)
            return TextSegmenter.load(model_path, trust_model=True)
        except Exception as e:
            last_error = e
            continue
    
    # If we get here, no paths worked
    raise RuntimeError(f"Failed to load small model. Last error: {last_error if 'last_error' in locals() else 'No valid model paths found'}")


def get_large_segmenter() -> TextSegmenter:
    """
    Get the large pre-trained text segmenter.
    
    The large model has the highest accuracy but also the largest memory footprint
    and may be slower for inference than the small or medium models.
    
    Returns:
        TextSegmenter: A pre-trained text segmenter using the large model
    """
    # Import needed types
    from charboundary.models import BinaryRandomForestModel
    from charboundary.segmenters import SegmenterConfig
    from skops.io import get_untrusted_types
    
    # Check possible model paths (compressed or uncompressed)
    model_paths = []
    
    # Large model
    model_base_name = 'large_model.skops'
    
    if sys.version_info >= (3, 9):
        # Use new resources API for Python 3.9+
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.files('charboundary.resources').joinpath(f'{model_base_name}{ext}'))
            except Exception:
                pass
    else:
        # Fallback for older Python versions
        for ext in ['', '.xz', '.lzma']:
            try:
                model_paths.append(resources.path('charboundary.resources', f'{model_base_name}{ext}').__enter__())
            except Exception:
                pass
    
    # Try each path until one works
    for model_resource in model_paths:
        try:
            model_path = str(model_resource)
            return TextSegmenter.load(model_path, trust_model=True)
        except Exception as e:
            last_error = e
            continue
    
    # If we get here, no paths worked
    raise RuntimeError(f"Failed to load large model. Last error: {last_error if 'last_error' in locals() else 'No valid model paths found'}")


__version__ = "0.2.0"
