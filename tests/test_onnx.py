"""Tests for ONNX conversion and inference in charboundary."""

import os
import tempfile
import pytest
from pathlib import Path

from charboundary.models import create_model, BinaryRandomForestModel
from charboundary.segmenters import TextSegmenter
from charboundary import TextSegmenter as PublicTextSegmenter

# Try to import ONNX support - if not available, skip tests
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False


# Skip all tests in this module if ONNX is not available
pytestmark = pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX dependencies not installed")


def test_onnx_model_conversion(trained_model: BinaryRandomForestModel):
    """Test converting a model to ONNX format."""
    # Set the feature count and enable ONNX
    trained_model.feature_count = 20  # Sample feature count
    trained_model.use_onnx = True
    
    # Convert to ONNX
    onnx_model = trained_model.to_onnx()
    
    # Verify the conversion was successful
    assert onnx_model is not None
    assert isinstance(onnx_model, bytes)
    assert len(onnx_model) > 0


def test_onnx_model_save_load(trained_model: BinaryRandomForestModel):
    """Test saving and loading an ONNX model."""
    # Set the feature count and enable ONNX
    trained_model.feature_count = 20  # Sample feature count
    trained_model.use_onnx = True
    
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name
    
    try:
        # Convert and save the model
        trained_model.to_onnx()
        result = trained_model.save_onnx(onnx_path)
        
        # Verify the model was saved successfully
        assert result is True
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
        
        # Create a new model and load the ONNX model
        new_model = create_model(model_type="random_forest")
        result = new_model.load_onnx(onnx_path)
        
        # Verify the model was loaded successfully
        assert result is True
        assert new_model.onnx_model is not None
        assert new_model.onnx_session is not None
        
        # Enable ONNX inference
        result = new_model.enable_onnx(True)
        assert result is True
        assert new_model.use_onnx is True
        
    finally:
        # Clean up the temporary file
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


def test_onnx_inference(trained_model: BinaryRandomForestModel):
    """Test inference using ONNX model."""
    # Set the feature count and enable ONNX
    trained_model.feature_count = 20  # Sample feature count
    
    # Prepare test data
    test_features = [[0] * 20 for _ in range(10)]  # Create 10 test samples with 20 features each
    
    # Get predictions using the regular model
    regular_predictions = trained_model.predict(test_features)
    assert len(regular_predictions) == 10
    
    # Convert to ONNX and enable
    trained_model.to_onnx()
    trained_model.enable_onnx(True)
    
    # Get predictions using the ONNX model
    onnx_predictions = trained_model.predict(test_features)
    assert len(onnx_predictions) == 10
    
    # Predictions should match (or be close enough)
    # In some cases there might be slight differences due to numerical precision
    # We'll consider this a strong test if all predictions match exactly
    for reg_pred, onnx_pred in zip(regular_predictions, onnx_predictions):
        assert reg_pred == onnx_pred


def test_onnx_segmenter_methods(trained_segmenter: TextSegmenter):
    """Test ONNX methods in the TextSegmenter class."""
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name
    
    try:
        # Convert the segmenter's model to ONNX
        onnx_model = trained_segmenter.to_onnx()
        assert onnx_model is not None
        
        # Save the ONNX model
        result = trained_segmenter.save_onnx(onnx_path)
        assert result is True
        assert os.path.exists(onnx_path)
        
        # Create a new segmenter
        new_segmenter = TextSegmenter(model=trained_segmenter.model)
        
        # Load the ONNX model
        result = new_segmenter.load_onnx(onnx_path)
        assert result is True
        
        # Enable ONNX inference
        result = new_segmenter.enable_onnx(True)
        assert result is True
        assert new_segmenter.config.use_onnx is True
        
        # Segment a sample text using ONNX
        text = "This is a test sentence. This is another sentence."
        segmented_text = new_segmenter.segment_text(text)
        
        # The segmentation should still work with ONNX
        assert "This is a test sentence." in segmented_text
        
    finally:
        # Clean up the temporary file
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


def test_train_with_onnx(sample_annotated_text, sample_model_params):
    """Test training a segmenter with ONNX enabled."""
    segmenter = TextSegmenter()
    
    # Train with ONNX enabled
    segmenter.train(
        data=[sample_annotated_text],
        model_params=sample_model_params,
        left_window=3,
        right_window=3,
        use_onnx=True
    )
    
    # Verify ONNX is enabled in the config
    assert segmenter.config.use_onnx is True
    
    # Verify the model has ONNX attributes
    assert hasattr(segmenter.model, 'onnx_model')
    
    # Should be converted to ONNX during training
    assert segmenter.model.onnx_model is not None
    
    # Test segmentation with ONNX
    text = "This is a test. This is only a test."
    segmented_text = segmenter.segment_text(text)
    
    # Basic check that segmentation still works
    assert len(segmenter.segment_to_sentences(text)) > 0


def test_public_segmenter_onnx():
    """Test ONNX support with the public TextSegmenter interface."""
    # Import error won't happen if ONNX is available
    # This test confirms the public API can enable ONNX
    from charboundary import TextSegmenter
    
    # Create a segmenter with ONNX enabled
    segmenter = TextSegmenter()
    
    # Train a minimal model
    sample_text = f"This is a test.{SENTENCE_TAG} This is another test.{SENTENCE_TAG}"
    segmenter.train(
        data=[sample_text],
        model_params={"n_estimators": 10, "max_depth": 3},
        use_onnx=True
    )
    
    # Verify ONNX is enabled
    assert segmenter.config.use_onnx is True
    
    # Test both normal inference and ONNX inference
    text = "Hello. World."
    # Disable ONNX first to test normal inference
    segmenter.enable_onnx(False)
    normal_result = segmenter.segment_text(text)
    
    # Enable ONNX to test ONNX inference
    segmenter.enable_onnx(True)
    onnx_result = segmenter.segment_text(text)
    
    # Results should be the same
    assert normal_result == onnx_result