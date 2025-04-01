"""
Example of using ONNX model conversion and inference with charboundary models.
This example demonstrates how to:
1. Convert a trained charboundary model to ONNX format
2. Save and load ONNX models
3. Use ONNX for inference
"""

import os
import sys
import random
import tempfile
from pathlib import Path
import time

# Add the parent directory to the path to import charboundary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if ONNX is available
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False
    
if not ONNX_AVAILABLE:
    print("ONNX support is not available. Install it with: pip install charboundary[onnx]")
    sys.exit(1)

# Import charboundary modules
from charboundary.models import create_model
from charboundary.features import extract_features


def generate_dummy_data(num_samples=1000, feature_dim=20):
    """Generate dummy data for training and testing."""
    X = []
    y = []
    
    for _ in range(num_samples):
        # Create random feature vector
        features = [random.randint(0, 1) for _ in range(feature_dim)]
        
        # Random label (more 0s than 1s to simulate imbalanced data)
        label = 1 if random.random() < 0.2 else 0
        
        X.append(features)
        y.append(label)
        
    return X, y


def main():
    print("ONNX Model Example")
    print("-----------------")
    
    # Generate dummy data
    print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=5000, feature_dim=20)
    X_test, y_test = generate_dummy_data(num_samples=1000, feature_dim=20)
    
    # Create and train a model
    print("Training model...")
    model = create_model(
        model_type="feature_selected_rf",
        n_estimators=50,  # Small model for fast training
        feature_selection_threshold=0.05
    )
    model.fit(X_train, y_train)
    
    # Get baseline performance (sklearn)
    print("\nEvaluating sklearn model performance...")
    start_time = time.time()
    predictions = model.predict(X_test)
    sklearn_time = time.time() - start_time
    metrics = model.get_metrics(X_test, y_test)
    
    print(f"Sklearn model metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
    print(f"  Prediction time: {sklearn_time:.4f} seconds")
    
    # Convert to ONNX
    print("\nConverting model to ONNX...")
    model.to_onnx()
    
    # Save ONNX model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name
    
    print(f"Saving ONNX model to {onnx_path}...")
    model.save_onnx(onnx_path)
    
    # Create a new model and load the ONNX model
    print("Creating new model and loading ONNX model...")
    new_model = create_model(model_type="feature_selected_rf")
    new_model.load_onnx(onnx_path)
    new_model.enable_onnx(True)
    
    # Evaluate ONNX model performance
    print("\nEvaluating ONNX model performance...")
    start_time = time.time()
    onnx_predictions = new_model.predict(X_test)
    onnx_time = time.time() - start_time
    
    # Calculate accuracy manually to verify ONNX predictions
    onnx_accuracy = sum(1 for true, pred in zip(y_test, onnx_predictions) if true == pred) / len(y_test)
    
    print(f"ONNX model metrics:")
    print(f"  Accuracy: {onnx_accuracy:.4f}")
    print(f"  Prediction time: {onnx_time:.4f} seconds")
    print(f"  Speedup: {sklearn_time / onnx_time:.2f}x")
    
    # Print prediction comparison
    prediction_match = sum(1 for a, b in zip(predictions, onnx_predictions) if a == b)
    print(f"\nPrediction comparison:")
    print(f"  Match: {prediction_match}/{len(predictions)} ({prediction_match/len(predictions):.2%})")
    
    # Clean up
    print(f"\nCleaning up temporary file {onnx_path}...")
    Path(onnx_path).unlink()
    
    print("\nONNX example completed.")


if __name__ == "__main__":
    main()