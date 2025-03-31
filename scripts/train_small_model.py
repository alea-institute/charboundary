#!/usr/bin/env python3
"""
Script to train the default model for the CharBoundary library.
This script trains on the full training dataset and saves the model to be included in the package.
"""

import os
import sys
import time

from charboundary import TextSegmenter


def train_default_model():
    """Train the default model for the CharBoundary library."""
    
    # Create a segmenter with optimized parameters
    segmenter = TextSegmenter()
    
    # Set the small model path in the package resources directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(package_dir, "charboundary", "resources")
    model_path = os.path.join(model_dir, "small_model.skops")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Use the full training data
    training_data_path = os.path.join(package_dir, "data", "train.jsonl.gz")
    
    if not os.path.exists(training_data_path):
        print(f"Error: Training data not found at {training_data_path}")
        return 1
    
    # Train the segmenter with optimized parameters
    print(f"Training small model using data from {training_data_path}...")
    t0 = time.time()
    metrics = segmenter.train(
        data=training_data_path,
        model_params={
            "n_estimators": 32,
            "max_depth": 16,
            "min_samples_split": 16,
            "min_samples_leaf": 8,
            "n_jobs": -1,
            "class_weight": "balanced"
        },
        sample_rate=0.001,  # Sample rate to get good class balance
        left_window=5,  # Optimized window sizes
        right_window=5
    )
    training_time = time.time() - t0
    
    # Display training metrics
    print("Training completed in {:.2f} seconds.".format(training_time))
    print(f"Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")
    
    # Save the model
    print(f"Saving model to {model_path}...")
    
    # Import and register trusted types for skops
    try:
        from skops.io import register_trusted_types
        from charboundary.models import BinaryRandomForestModel
        from charboundary.segmenters import SegmenterConfig
        
        # Register the custom types as trusted for skops
        register_trusted_types(BinaryRandomForestModel, SegmenterConfig)
        print("Registered custom types as trusted for skops")
    except ImportError:
        print("Warning: Could not register trusted types for skops")
    
    # Save with compression for smaller file size
    original_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    
    segmenter.save(model_path, compress=True, compression_level=9)
    
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
        
        if original_size > 0:
            ratio = original_size / compressed_size
            print(f"Compression ratio: {ratio:.2f}x (from {original_size/1024/1024:.1f}MB to {compressed_size/1024/1024:.1f}MB)")
        else:
            print(f"Compressed size: {compressed_size/1024/1024:.1f}MB")
    else:
        print(f"Model saved successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(train_default_model())
