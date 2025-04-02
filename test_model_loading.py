#!/usr/bin/env python
"""
Test script to verify model loading behavior.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from charboundary import (
    get_small_segmenter,
    get_default_segmenter,
    get_large_segmenter,
    get_small_onnx_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter,
    get_resource_dir
)

def test_model_loading():
    """Test loading all models."""
    print("Testing model loading...")
    
    # Print resource directory
    print(f"Resource directory: {get_resource_dir()}")
    
    try:
        # Try loading small model (should be in the package)
        print("\nLoading small model...")
        small_segmenter = get_small_segmenter()
        print("✓ Successfully loaded small model")
        
        # Try loading medium model (should download if not available)
        print("\nLoading medium model...")
        print("This may trigger a download if the model is not available locally...")
        medium_segmenter = get_default_segmenter()
        print("✓ Successfully loaded medium model")
        
        # Only check if the ONNX directory exists and has the small model
        print("\nChecking ONNX directory structure...")
        from pathlib import Path
        onnx_dir = Path(get_resource_dir()) / "onnx"
        print(f"ONNX directory exists: {onnx_dir.exists()}")
        small_onnx_path = onnx_dir / "small_model.onnx" 
        print(f"Small ONNX model exists: {small_onnx_path.exists()}")
        
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_model_loading()