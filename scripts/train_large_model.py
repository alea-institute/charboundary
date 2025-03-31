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
    
    # Set the default model path in the package resources directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(package_dir, "charboundary", "resources")
    model_path = os.path.join(model_dir, "large_model.skops")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Use the full training data
    training_data_path = os.path.join(package_dir, "data", "train.jsonl.gz")
    
    if not os.path.exists(training_data_path):
        print(f"Error: Training data not found at {training_data_path}")
        return 1
    
    # Train the segmenter with optimized parameters and feature selection
    print(f"Training large model using data from {training_data_path}...")
    print("Enabling feature selection for improved performance...")
    t0 = time.time()
    metrics = segmenter.train(
        data=training_data_path,
        model_params={
            "n_estimators": 256,
            "max_depth": 64,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "class_weight": "balanced_subsample"
        },
        sample_rate=0.001,  # Sample rate to get good class balance
        left_window=9,  # Optimized window sizes
        right_window=9,
        use_feature_selection=False,  # Enable feature selection
        feature_selection_threshold=0.001,  # Keep features with importance >= 0.5%
        max_features=None  # No upper limit on number of features (use threshold only)
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
    
    # Get feature selection info if available
    if hasattr(segmenter, 'model') and hasattr(segmenter.model, 'get_feature_importances'):
        try:
            feature_info = segmenter.model.get_feature_importances()
            orig_features = feature_info.get("original_num_features", 0)
            selected_features = feature_info.get("selected_num_features", 0)
            
            if orig_features > 0:
                print("\nFeature Selection Summary:")
                print(f"  Original features:      {orig_features}")
                print(f"  Selected features:      {selected_features}")
                print(f"  Reduction:              {(1 - selected_features/orig_features)*100:.1f}%")
                
                # Print top 15 most important features
                if "selected_indices" in feature_info and "original_importances" in feature_info:
                    print("\nTop 15 most important features:")
                    indices = feature_info["selected_indices"][:15]  # Get top 15
                    importances = feature_info["original_importances"]
                    
                    for i, idx in enumerate(indices, 1):
                        print(f"  {i}. Feature {idx}: importance={importances[idx]:.6f}")
        except Exception as e:
            print(f"Note: Could not retrieve feature selection information: {e}")
    
    # Save the model
    print(f"Saving model to {model_path}...")
    
    # Import and register trusted types for skops
    try:
        from skops.io import register_trusted_types
        from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel
        from charboundary.segmenters import SegmenterConfig
        
        # Register the custom types as trusted for skops
        register_trusted_types(BinaryRandomForestModel, FeatureSelectedRandomForestModel, SegmenterConfig)
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
