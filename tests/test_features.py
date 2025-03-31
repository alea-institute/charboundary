"""Tests for the features module of CharSentence."""

import pytest
from typing import List

from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG
from charboundary.features import FeatureExtractor


class TestFeatureExtractor:
    """Test the FeatureExtractor class."""

    def test_basic_feature_extraction(self, feature_extractor: FeatureExtractor):
        """Test basic feature extraction."""
        # Test with a simple text
        text = "abc"
        features = feature_extractor.get_char_features(text, left_window=1, right_window=1)
        
        # Should have a feature vector for each character
        assert len(features) == len(text)
        
        # Feature vector has window*2+1 chars features + 8 special features
        assert len(features[0]) == (2*1+1) + 8  # window*2+1 + 8 special features
        
        # Check the middle character features
        middle_features = features[1]
        
        # First feature should be the code for 'a'
        assert middle_features[0] == feature_extractor.encoder.encode('a')
        
        # Second feature should be the code for 'b'
        assert middle_features[1] == feature_extractor.encoder.encode('b')
        
        # Third feature should be the code for 'c'
        assert middle_features[2] == feature_extractor.encoder.encode('c')
        
        # Last feature should be the abbreviation flag (0 for 'b')
        assert middle_features[3] == 0

    def test_abbreviation_detection(self, feature_extractor: FeatureExtractor):
        """Test abbreviation detection."""
        # Add a test abbreviation
        feature_extractor.abbreviations.add("Dr.")
        
        # Test with a text containing the abbreviation
        text = "Dr. Smith"
        features = feature_extractor.get_char_features(text, left_window=1, right_window=1)
        
        # Find the feature vector for the period
        period_index = text.find('.')
        period_features = features[period_index]
        
        # The abbreviation flag is at window_size index (index 3 for window=1)
        assert period_features[3] == 1
        
        # Test with a non-abbreviation period
        text = "End of sentence."
        features = feature_extractor.get_char_features(text, left_window=1, right_window=1)
        
        # Find the feature vector for the period
        period_index = text.find('.')
        period_features = features[period_index]
        
        # The abbreviation flag is at window_size index (index 3 for window=1)
        assert period_features[3] == 0

    def test_mark_annotation_positions(self, feature_extractor: FeatureExtractor):
        """Test marking of annotation positions."""
        # Test with a text containing sentence and paragraph tags
        text = f"First sentence.{SENTENCE_TAG} Second sentence.{SENTENCE_TAG}{PARAGRAPH_TAG}New paragraph."
        
        labels = feature_extractor.mark_annotation_positions(text)
        
        # Labels should have length equal to the number of chars in clean text
        clean_text = text.replace(SENTENCE_TAG, "").replace(PARAGRAPH_TAG, "")
        assert len(labels) == len(clean_text)
        
        # Check that sentence boundaries are marked correctly
        first_period_index = clean_text.find('.')
        assert labels[first_period_index] == 1
        
        # Check that second sentence (also paragraph) boundary is marked
        second_period_index = clean_text.find('.', first_period_index + 1)
        assert labels[second_period_index] == 1

    def test_process_annotated_text(self, feature_extractor: FeatureExtractor, sample_annotated_text: str):
        """Test processing of annotated text."""
        clean_text, features, labels = feature_extractor.process_annotated_text(
            sample_annotated_text,
            left_window=2,
            right_window=2
        )
        
        # Check that the text is cleaned properly
        assert SENTENCE_TAG not in clean_text
        assert PARAGRAPH_TAG not in clean_text
        
        # Check that features and labels have the right length
        assert len(features) == len(clean_text)
        assert len(labels) == len(clean_text)
        
        # Check feature vector size - window*2+1 char features + 8 special features
        assert len(features[0]) == 2*2+1+8  # 2*window+1+8

    def test_benchmark_feature_extraction(self, benchmark, feature_extractor: FeatureExtractor):
        """Benchmark feature extraction performance."""
        text = "This is a sample text for benchmarking feature extraction performance."
        
        # Define a benchmark function
        def extract_features():
            feature_extractor.get_char_features(text, left_window=3, right_window=3)
        
        # Run the benchmark
        benchmark(extract_features)

    def test_benchmark_process_annotated_text(self, benchmark, feature_extractor: FeatureExtractor, sample_annotated_text: str):
        """Benchmark processing of annotated text."""
        # Define a benchmark function
        def process_text():
            feature_extractor.process_annotated_text(
                sample_annotated_text,
                left_window=3,
                right_window=3
            )
        
        # Run the benchmark
        benchmark(process_text)

    def test_numpy_vs_list_processing(self, sample_annotated_text: str):
        """Test and compare NumPy and list-based processing."""
        # Only run test if NumPy is available
        try:
            import numpy as np
            numpy_available = True
        except ImportError:
            pytest.skip("NumPy not available, skipping test")
            return
        
        # Create feature extractors with and without NumPy
        numpy_extractor = FeatureExtractor(use_numpy=True)
        list_extractor = FeatureExtractor(use_numpy=False)
        
        # Process the same text with both extractors
        numpy_result = numpy_extractor.process_annotated_text(
            sample_annotated_text,
            left_window=3,
            right_window=3
        )
        
        list_result = list_extractor.process_annotated_text(
            sample_annotated_text,
            left_window=3,
            right_window=3
        )
        
        # Results should be identical
        assert numpy_result[0] == list_result[0]  # Clean text
        assert numpy_result[2] == list_result[2]  # Labels
        
        # Features should be equivalent
        for numpy_feat, list_feat in zip(numpy_result[1], list_result[1]):
            assert numpy_feat == list_feat