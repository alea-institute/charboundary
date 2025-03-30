#!/usr/bin/env python3
"""
Script to compress the existing default model for the CharBoundary library.
"""

import os
import sys
import time

from charboundary import TextSegmenter


def compress_model():
    """Compress the existing default model."""
    
    # Set the default model path in the package resources directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(package_dir, "charboundary", "resources")
    model_path = os.path.join(model_dir, "default_model.skops")
    
    # Ensure model directory exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return 1
    
    # Load the existing model
    print(f"Loading existing model from {model_path}...")
    t0 = time.time()
    segmenter = TextSegmenter.load(model_path, trust_model=True)
    print(f"Model loaded in {time.time() - t0:.2f} seconds.")
    
    # Get the original file size
    original_size = os.path.getsize(model_path)
    print(f"Original model size: {original_size / (1024*1024):.2f} MB")
    
    # Save with compression
    print(f"Saving model with compression...")
    t0 = time.time()
    segmenter.save(model_path, compress=True, compression_level=9)
    print(f"Model saved in {time.time() - t0:.2f} seconds.")
    
    # Check the compression ratio achieved
    compressed_path = model_path + '.xz'
    if os.path.exists(compressed_path):
        compressed_size = os.path.getsize(compressed_path)
        print(f"Model saved successfully with compression!")
        
        # Make sure the original file is gone
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Removed uncompressed file {model_path}")
            except Exception as e:
                print(f"Note: Could not remove uncompressed file: {e}")
        
        ratio = original_size / compressed_size
        print(f"Compression ratio: {ratio:.2f}x (from {original_size/1024/1024:.1f}MB to {compressed_size/1024/1024:.1f}MB)")
    else:
        print(f"Warning: Compressed model not found at {compressed_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(compress_model())