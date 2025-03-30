"""Explicit benchmarks for CharSentence.

This module contains benchmarks that can be run explicitly with:
pytest tests/test_benchmarks.py -xvs --benchmark-only

Regular test runs will skip these by default due to the "--benchmark-skip" option
in the pytest configuration.
"""

import pytest
from typing import List

from charboundary import TextSegmenter
from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG
from charboundary.encoders import CharacterEncoder
from charboundary.features import FeatureExtractor


@pytest.mark.benchmark(
    group="encoders",
    min_time=1,
    max_time=5,
    min_rounds=5,
    timer=lambda: 0,
    disable_gc=True,
    warmup=True
)
def test_encoder_benchmark(benchmark):
    """Benchmark the CharacterEncoder performance."""
    encoder = CharacterEncoder()
    text = "The quick brown fox jumps over the lazy dog. " * 10
    
    @benchmark
    def result():
        for char in text:
            encoder.encode(char)
    
    # Verify result
    assert result is None


@pytest.mark.benchmark(
    group="features",
    min_time=1,
    max_time=5,
    min_rounds=5,
    timer=lambda: 0,
    disable_gc=True,
    warmup=True
)
def test_feature_extraction_benchmark(benchmark):
    """Benchmark the feature extraction performance."""
    extractor = FeatureExtractor()
    text = "The quick brown fox jumps over the lazy dog. " * 10
    
    @benchmark
    def result():
        return extractor.get_char_features(text, left_window=5, right_window=5)
    
    # Verify result
    assert len(result) == len(text)


@pytest.mark.benchmark(
    group="segmentation",
    min_time=1,
    max_time=5,
    min_rounds=5,
    timer=lambda: 0,
    disable_gc=True,
    warmup=True
)
def test_segmentation_benchmark(benchmark, trained_segmenter):
    """Benchmark the text segmentation performance."""
    text = "The quick brown fox jumps over the lazy dog. " * 10
    text += "This is another sentence. And yet another one! How about a question? "
    text += "Let's add more text to make it longer. " * 5
    
    @benchmark
    def result():
        return trained_segmenter.segment_text(text)
    
    # Verify result
    assert SENTENCE_TAG in result or PARAGRAPH_TAG in result


@pytest.mark.benchmark(
    group="segmentation",
    min_time=1,
    max_time=5,
    min_rounds=5,
    timer=lambda: 0,
    disable_gc=True,
    warmup=True
)
def test_sentence_extraction_benchmark(benchmark, trained_segmenter):
    """Benchmark the sentence extraction performance."""
    text = "The quick brown fox jumps over the lazy dog. " * 5
    text += "This is another sentence. And yet another one! How about a question? "
    text += "Let's add more text to make it longer. " * 3
    
    @benchmark
    def result():
        return trained_segmenter.segment_to_sentences(text)
    
    # Verify result
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(s, str) for s in result)