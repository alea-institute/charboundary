"""Tests for the top-level module of CharSentence."""

import pytest

import charboundary


def test_top_level_imports():
    """Test that key classes are importable from the top level."""
    # Check that TextSegmenter can be imported from the top level
    assert hasattr(charboundary, 'TextSegmenter')
    
    # Create an instance to make sure it works
    segmenter = charboundary.TextSegmenter()
    assert not segmenter.is_trained