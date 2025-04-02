#!/usr/bin/env python
"""
Test script to verify model downloading behavior.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from charboundary import (
    get_default_segmenter,
    get_large_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter,
    get_resource_dir,
    download_onnx_model
)

def remove_medium_model():
    """Remove medium model if it exists."""
    resource_dir = Path(get_resource_dir())
    medium_path = resource_dir / "medium_model.skops.xz"
    
    if medium_path.exists():
        print(f"Removing existing medium model: {medium_path}")
        medium_path.unlink()
        return True
    return False

def test_medium_download():
    """Test downloading the medium model."""
    had_model = remove_medium_model()
    
    print("\nTesting medium model download...")
    print("This should download the medium model from GitHub:")
    
    try:
        segmenter = get_default_segmenter()
        text = "Hello, world! This is a test."
        result = segmenter.segment_to_sentences(text)
        print(f"✓ Successfully downloaded and used medium model")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_medium_download()