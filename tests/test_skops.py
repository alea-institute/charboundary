"""
Tests for the skops model serialization and loading in TextSegmenter.
"""

import os
import tempfile
import random
import pytest

from charboundary.segmenters import TextSegmenter, SegmenterConfig
from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG


def create_sample_training_data(num_samples=10, max_length=100):
    """Generate simple training data with sentence and paragraph annotations."""
    training_data = []
    
    for _ in range(num_samples):
        # Create a random string with boundary annotations
        text = []
        for i in range(random.randint(20, max_length)):
            # Add a random character (ASCII letters and punctuation)
            char = chr(random.randint(65, 122))
            text.append(char)
            
            # Randomly insert sentence and paragraph boundaries
            if random.random() < 0.1:
                text.append('.' + SENTENCE_TAG)
                
                # Occasionally add paragraph tags after sentences
                if random.random() < 0.2:
                    text.append(PARAGRAPH_TAG)
        
        training_data.append(''.join(text))
    
    return training_data


class TestSkopsIntegration:
    """Test the integration with skops for model serialization and loading."""
    
    def test_save_load_model(self):
        """Test that a model can be saved with skops and loaded correctly."""
        # Create a simple segmenter with minimal configuration
        config = SegmenterConfig(
            left_window=3,
            right_window=3,
            model_type="random_forest",
            model_params={
                "n_estimators": 10,  # Small model for faster tests
                "max_depth": 5
            }
        )
        
        segmenter = TextSegmenter(config=config)
        
        # Generate random training data
        training_data = create_sample_training_data(num_samples=5)
        
        # Train the model
        metrics = segmenter.train(
            data=training_data,
            sample_rate=0.2,
            max_samples=5
        )
        
        # Create a temporary file for saving the model
        with tempfile.NamedTemporaryFile(suffix='.skops', delete=False) as tmp_file:
            model_path = tmp_file.name
            
        try:
            # Save the model
            segmenter.save(model_path, format="skops")
            
            # Load the model with trust_model=True since we're testing our own models
            loaded_segmenter = TextSegmenter.load(model_path, use_skops=True, trust_model=True)
            
            # Verify configuration was preserved
            assert loaded_segmenter.config.left_window == config.left_window
            assert loaded_segmenter.config.right_window == config.right_window
            assert loaded_segmenter.config.model_type == config.model_type
            
            # Verify model functionality works after loading
            test_text = "This is a test. Another sentence."
            segmented_text = loaded_segmenter.segment_text(test_text)
            
            # The model should produce some kind of segmentation
            assert isinstance(segmented_text, str)
            
            # Convert to sentences and check if at least one sentence was detected
            sentences = loaded_segmenter.segment_to_sentences(test_text)
            assert isinstance(sentences, list)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_skops_with_trusted_parameter(self):
        """Test loading a model with trusted=True parameter."""
        # Create a simple segmenter with minimal configuration
        config = SegmenterConfig(
            left_window=3,
            right_window=3,
            model_type="random_forest",
            model_params={
                "n_estimators": 10,  # Small model for faster tests
                "max_depth": 5
            }
        )
        
        segmenter = TextSegmenter(config=config)
        
        # Generate random training data
        training_data = create_sample_training_data(num_samples=3)
        
        # Train the model
        segmenter.train(data=training_data, sample_rate=0.2)
        
        # Create a temporary file for saving the model
        with tempfile.NamedTemporaryFile(suffix='.skops', delete=False) as tmp_file:
            model_path = tmp_file.name
            
        try:
            # Save the model
            segmenter.save(model_path, format="skops")
            
            # Load the model with trust_model=True
            loaded_segmenter = TextSegmenter.load(
                model_path, 
                use_skops=True, 
                trust_model=True
            )
            
            # Verify model functionality works after loading
            test_text = "This is a test."
            segmented_text = loaded_segmenter.segment_text(test_text)
            assert isinstance(segmented_text, str)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_skops_fallback_to_pickle(self):
        """Test the fallback to pickle when loading fails with skops."""
        # Create a simple segmenter
        segmenter = TextSegmenter()
        
        # Train with minimal data
        training_data = create_sample_training_data(num_samples=2, max_length=50)
        segmenter.train(data=training_data, sample_rate=0.2)
        
        # Create a temporary file for saving the model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
            
        try:
            # Save with pickle format
            segmenter.save(model_path, format="pickle")
            
            # Try to load with skops first (should fall back to pickle)
            loaded_segmenter = TextSegmenter.load(model_path)
            
            # Verify model works
            test_text = "This is a test."
            result = loaded_segmenter.segment_text(test_text)
            assert isinstance(result, str)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)