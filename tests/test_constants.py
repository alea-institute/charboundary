"""Tests for the constants module of CharSentence."""

import pytest

from charboundary.constants import (
    SENTENCE_TAG, 
    PARAGRAPH_TAG, 
    TERMINAL_SENTENCE_CHAR_LIST, 
    TERMINAL_PARAGRAPH_CHAR_LIST,
    DEFAULT_ABBREVIATIONS
)


def test_constants_values():
    """Test that constants have expected values."""
    # Tags should be non-empty strings
    assert isinstance(SENTENCE_TAG, str) and len(SENTENCE_TAG) > 0
    assert isinstance(PARAGRAPH_TAG, str) and len(PARAGRAPH_TAG) > 0
    
    # Tags should be different from each other
    assert SENTENCE_TAG != PARAGRAPH_TAG
    
    # Terminal character lists should be non-empty
    assert len(TERMINAL_SENTENCE_CHAR_LIST) > 0
    assert len(TERMINAL_PARAGRAPH_CHAR_LIST) > 0
    
    # Default abbreviations should exist
    assert len(DEFAULT_ABBREVIATIONS) > 0


def test_terminal_characters():
    """Test that terminal character lists contain expected characters."""
    # Sentence terminals should include basic punctuation
    assert '.' in TERMINAL_SENTENCE_CHAR_LIST
    assert '?' in TERMINAL_SENTENCE_CHAR_LIST
    assert '!' in TERMINAL_SENTENCE_CHAR_LIST
    
    # Paragraph terminals should include newlines
    assert '\n' in TERMINAL_PARAGRAPH_CHAR_LIST