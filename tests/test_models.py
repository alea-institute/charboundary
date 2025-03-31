"""Tests for the models module of CharSentence."""

import pytest
from typing import List

from charboundary.models import BinaryRandomForestModel, create_model


class TestModels:
    """Test the models in CharSentence."""

    def test_create_model(self, sample_model_params):
        """Test model creation."""
        # Test creating a model with default parameters
        model = create_model()
        assert isinstance(model, BinaryRandomForestModel)
        
        # Test creating a model with specific parameters
        model = create_model(model_type="random_forest", **sample_model_params)
        assert isinstance(model, BinaryRandomForestModel)
        
        # Test with explicit binary random forest type
        model = create_model(model_type="binary_random_forest", **sample_model_params)
        assert isinstance(model, BinaryRandomForestModel)

    def test_invalid_model_type(self):
        """Test creating a model with an invalid type."""
        with pytest.raises(ValueError):
            create_model(model_type="invalid_model_type")

    def test_model_binary_classification(self, trained_model):
        """Test binary classification with the model."""
        
        # Create some test features matching the model's expected feature count (15 features)
        test_features = [[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        # Get predictions
        predictions = trained_model.predict(test_features)
        
        # Check that predictions are binary
        for pred in predictions:
            assert pred in [0, 1]

    def test_model_metrics(self, trained_model):
        """Test model metrics calculation."""
        
        # Create test data matching model's feature dimensions (15 features)
        test_features = [
            [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        test_labels = [0, 1, 0]
        
        # Calculate metrics
        metrics = trained_model.get_metrics(test_features, test_labels)
        
        # Check that we have the expected metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "boundary_accuracy" in metrics
        assert "binary_mode" in metrics
        
        # Check that binary_mode is True
        assert metrics["binary_mode"] is True

    def test_feature_importances(self, trained_model):
        """Test feature importance calculation."""
        
        # Get feature importances
        importances = trained_model.get_feature_importances()
        
        # The actual number of importances depends on the window size used in training
        # For the trained_model fixture with optimizations, expected feature count is 15
        expected_features = 15
        assert len(importances) == expected_features
        
        # Importances should sum to approximately 1
        assert abs(sum(importances) - 1.0) < 0.001
        
        # All importances should be non-negative
        for importance in importances:
            assert importance >= 0

    def test_benchmark_prediction(self, benchmark, trained_model):
        """Benchmark model prediction performance."""
        # Create some test features matching the model's feature count (15)
        test_features = []
        for i in range(100):
            test_features.append([i % 10, (i + 1) % 10, (i + 2) % 10] + [0] * 12)
        
        # Define a benchmark function
        def predict_features():
            trained_model.predict(test_features)
        
        # Run the benchmark
        benchmark(predict_features)