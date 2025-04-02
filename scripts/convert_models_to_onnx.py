#!/usr/bin/env python
"""
Script to convert all built-in charboundary models to ONNX format.

This script loads each of the standard models shipped with charboundary
and converts them to ONNX format.
"""

import os
import sys
import tempfile
import warnings

# Add the parent directory to the path to import charboundary
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Check if ONNX is available
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

if not ONNX_AVAILABLE:
    print("Error: ONNX support is not available.")
    print("Please install the ONNX dependencies with: pip install onnx skl2onnx onnxruntime")
    sys.exit(1)

from charboundary import get_small_segmenter, get_default_segmenter, get_large_segmenter


def infer_feature_count(model):
    """Try to infer the feature count from a model."""
    try:
        # First check if the model has feature_importances_ which indicates n_features
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            return len(model.model.feature_importances_)
        
        # For sklearn RandomForest
        if hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            return model.model.n_features_in_
            
        # For feature-selected models, check selected features
        if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices:
            return len(model.selected_feature_indices)
            
        # For models with a FeatureSelectedRandomForestModel attribute
        if hasattr(model, 'model') and hasattr(model.model, 'selected_feature_indices') and model.model.selected_feature_indices:
            return len(model.model.selected_feature_indices)
    
    except Exception as e:
        print(f"Error inferring feature count: {e}")
    
    # Defaults for known models
    model_sizes = {
        "small": 20,   # Small model typically has ~20 features
        "medium": 32,  # Medium model typically has ~32 features
        "large": 48    # Large model typically has ~48 features
    }
    
    # If we can identify the model type from its name
    for model_type, feature_count in model_sizes.items():
        if model_type in str(model):
            print(f"Using default feature count of {feature_count} for {model_type} model")
            return feature_count
    
    # Default fallback
    print("Using default feature count of 32 (couldn't determine from model)")
    return 32


def convert_segmenter_model(segmenter_getter, model_name, optimization_level=2):
    """
    Convert a segmenter model to ONNX format.
    
    Args:
        segmenter_getter: Function that returns a segmenter
        model_name: Name of the model (small, medium, large)
        optimization_level: ONNX optimization level (0-3)
    """
    print(f"Loading {model_name} model...")
    
    # Get the segmenter
    segmenter = segmenter_getter()
    
    # Output path
    output_dir = os.path.join(PROJECT_DIR, "charboundary", "resources", "onnx")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_model.onnx")
    
    # Get the model from the segmenter
    model = segmenter.model
    
    # Set the feature count - this is required for ONNX conversion
    feature_count = infer_feature_count(model)
    model.feature_count = feature_count
    print(f"Set feature count to {feature_count}")
    
    # Set optimization level
    model.onnx_optimization_level = optimization_level
    print(f"Using optimization level: {optimization_level}")
    
    # Convert to ONNX
    print(f"Converting {model_name} model to ONNX...")
    try:
        # Convert to ONNX
        onnx_model = model.to_onnx()
        
        if onnx_model is None:
            print(f"Error: ONNX conversion failed for {model_name} model. No model was generated.")
            return False
            
        # Save the ONNX model
        print(f"Saving ONNX model to {output_path}...")
        if model.save_onnx(output_path):
            print(f"Successfully saved {model_name} ONNX model to {output_path}")
            print(f"Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return True
        else:
            print(f"Error: Failed to save {model_name} ONNX model.")
            return False
            
    except Exception as e:
        print(f"Error converting {model_name} model to ONNX: {e}")
        return False


def main():
    """Convert all built-in models to ONNX format."""
    print("Converting charboundary models to ONNX format...")
    
    # Convert the small model - Level 2 (Extended) recommended
    convert_segmenter_model(get_small_segmenter, "small", optimization_level=2)
    
    # Convert the medium model - Level 2 (Extended) recommended
    convert_segmenter_model(get_default_segmenter, "medium", optimization_level=2)
    
    # Convert the large model - Level 3 (All) recommended
    convert_segmenter_model(get_large_segmenter, "large", optimization_level=3)
    
    print("\nAll models have been converted and saved to charboundary/resources/onnx/")


if __name__ == "__main__":
    main()