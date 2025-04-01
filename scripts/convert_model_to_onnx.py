#!/usr/bin/env python
"""
Script to convert an existing charboundary model to ONNX format.

This script loads a saved charboundary model file and converts it to ONNX format.
It's useful for existing models that were trained without ONNX support.

Usage:
    python convert_model_to_onnx.py --input path/to/model.skops.xz --output path/to/model.onnx [--feature-count 200]

If feature-count is not provided, the script will try to infer it from the model.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

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
    print("Please install the ONNX dependencies with: pip install charboundary[onnx]")
    sys.exit(1)

from charboundary import TextSegmenter
from charboundary.models import BinaryRandomForestModel, FeatureSelectedRandomForestModel


def infer_feature_count(model) -> int:
    """Try to infer the feature count from the model."""
    if isinstance(model, FeatureSelectedRandomForestModel):
        if hasattr(model, "selected_feature_indices") and model.selected_feature_indices:
            # For feature-selected models, get the count of selected features
            return len(model.selected_feature_indices)
    
    # For standard RandomForest models, get the feature count from the first tree
    if hasattr(model, "model") and hasattr(model.model, "n_features_in_"):
        return model.model.n_features_in_
    
    # If we can't determine the feature count, return a default value and warn
    warnings.warn(
        "Could not determine feature count from model. Using default value of 200. "
        "If conversion fails, please specify the feature count manually with --feature-count."
    )
    return 200


def convert_model(input_path: str, output_path: str, feature_count: int = None) -> None:
    """
    Convert a charboundary model to ONNX format.
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the ONNX model
        feature_count: Number of features in the model (optional)
    """
    print(f"Loading model from {input_path}...")
    
    try:
        # Try to load as a segmenter (which contains the full model)
        segmenter = TextSegmenter.load(input_path)
        model = segmenter.model
        print("Successfully loaded model from TextSegmenter.")
    except Exception as e:
        print(f"Error loading as TextSegmenter: {e}")
        print("Trying to load as raw model...")
        
        try:
            # Try to use pickle loading for standalone models
            import pickle
            import lzma
            
            if input_path.endswith('.xz'):
                with lzma.open(input_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                with open(input_path, 'rb') as f:
                    model = pickle.load(f)
                    
            print("Successfully loaded raw model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    # Verify we have a proper model
    if not isinstance(model, (BinaryRandomForestModel, FeatureSelectedRandomForestModel)):
        print(f"Error: Expected a charboundary model, but got {type(model)}.")
        sys.exit(1)
    
    # Infer feature count if not provided
    if feature_count is None:
        feature_count = infer_feature_count(model)
        print(f"Inferred feature count: {feature_count}")
    
    # Set the feature count on the model (needed for ONNX conversion)
    model.feature_count = feature_count
    
    # Convert to ONNX
    print(f"Converting model to ONNX with {feature_count} features...")
    try:
        onnx_model = model.to_onnx()
        
        if onnx_model is None:
            print("Error: ONNX conversion failed. No model was generated.")
            sys.exit(1)
            
        # Save the ONNX model
        print(f"Saving ONNX model to {output_path}...")
        if model.save_onnx(output_path):
            print(f"Successfully saved ONNX model to {output_path}")
            print(f"Model size: {os.path.getsize(output_path) / 1024:.2f} KB")
        else:
            print("Error: Failed to save ONNX model.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
        sys.exit(1)


def main():
    """Parse arguments and convert the model."""
    parser = argparse.ArgumentParser(description="Convert a charboundary model to ONNX format.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input model file (.skops, .skops.xz, .pickle, etc.)")
    parser.add_argument("--output", "-o", required=True, help="Path to save the ONNX model (.onnx)")
    parser.add_argument("--feature-count", "-f", type=int, help="Number of features in the model (if not provided, will try to infer)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the model
    convert_model(args.input, args.output, args.feature_count)


if __name__ == "__main__":
    main()