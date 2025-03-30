"""Pytest configuration and fixtures for testing CharSentence."""

import pytest
from typing import List, Dict, Any

from charboundary import TextSegmenter
from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG
from charboundary.encoders import CharacterEncoder
from charboundary.features import FeatureExtractor
from charboundary.models import create_model, BinaryRandomForestModel


@pytest.fixture
def sample_annotated_text() -> str:
    """Return a sample annotated text for testing."""
    return (
        f"This is a test sentence.{SENTENCE_TAG} "
        f"This is another test sentence.{SENTENCE_TAG}{PARAGRAPH_TAG}"
        f"This is a new paragraph.{SENTENCE_TAG} "
        f"This has a Dr. abbreviation.{SENTENCE_TAG}"
    )


@pytest.fixture
def sample_texts() -> List[str]:
    """Return a list of sample texts for testing."""
    return [
        "Hello world. This is a simple test.",
        "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson.",
        "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that "
        "racial segregation was unconstitutional. This landmark decision changed "
        "the course of history."
    ]


@pytest.fixture
def sample_model_params() -> Dict[str, Any]:
    """Return sample model parameters for testing."""
    return {
        "n_estimators": 10,  # Small value for testing
        "max_depth": 5,      # Small value for testing
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": 1,         # Single job for consistent testing
        "class_weight": "balanced"
    }


@pytest.fixture
def character_encoder() -> CharacterEncoder:
    """Return a character encoder instance for testing."""
    return CharacterEncoder()


@pytest.fixture
def feature_extractor(character_encoder) -> FeatureExtractor:
    """Return a feature extractor instance for testing."""
    return FeatureExtractor(encoder=character_encoder)


@pytest.fixture
def trained_model(sample_annotated_text, sample_model_params) -> BinaryRandomForestModel:
    """Return a small trained model for testing."""
    model = create_model(model_type="random_forest", **sample_model_params)
    
    # Create a small feature extractor for training
    feature_extractor = FeatureExtractor()
    
    # Process the sample text to get features and labels
    _, features, labels = feature_extractor.process_annotated_text(
        sample_annotated_text, left_window=3, right_window=3
    )
    
    # Train the model
    model.fit(features, labels)
    
    return model


@pytest.fixture
def trained_segmenter(sample_annotated_text, sample_model_params) -> TextSegmenter:
    """Return a trained text segmenter for testing."""
    segmenter = TextSegmenter()
    
    # Train with minimal data
    segmenter.train(
        data=[sample_annotated_text],
        model_params=sample_model_params,
        left_window=3,
        right_window=3
    )
    
    return segmenter