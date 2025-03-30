"""Tests for the encoders module of CharSentence."""

import pytest
from typing import Dict

from charboundary.encoders import CharacterEncoder


class TestCharacterEncoder:
    """Test the CharacterEncoder class."""

    def test_basic_encoding(self, character_encoder: CharacterEncoder):
        """Test basic character encoding."""
        # Test basic encoding
        assert character_encoder.encode('a') == 1
        assert character_encoder.encode('b') == 2
        assert character_encoder.encode('c') == 3
        
        # Test consistency
        assert character_encoder.encode('a') == 1  # Should return the same code

    def test_whitespace_encoding(self, character_encoder: CharacterEncoder):
        """Test whitespace character encoding."""
        assert character_encoder.encode(' ') == -3  # Whitespace
        assert character_encoder.encode('\t') == -3  # Whitespace
        assert character_encoder.encode('\n') == -2  # Terminal paragraph
        
    def test_special_character_encoding(self, character_encoder: CharacterEncoder):
        """Test special character encoding."""
        assert character_encoder.encode('.') == -1  # Terminal sentence char
        assert character_encoder.encode('!') == -1  # Terminal sentence char
        assert character_encoder.encode('?') == -1  # Terminal sentence char

    def test_cache_functionality(self):
        """Test the caching functionality."""
        encoder = CharacterEncoder()
        
        # Fill the cache
        encoder.encode('a')
        encoder.encode('b')
        encoder.encode('c')
        
        # Check cache content
        assert 'a' in encoder.cache
        assert 'b' in encoder.cache
        assert 'c' in encoder.cache
        
        # Add one more 
        encoder.encode('d')
        
        # All characters should be in the cache
        assert 'a' in encoder.cache
        assert 'd' in encoder.cache

    def test_cache_manual_clear(self, character_encoder: CharacterEncoder):
        """Test manually clearing the cache."""
        # Add a few characters to the cache
        character_encoder.encode('a')
        character_encoder.encode('b')
        
        # Manually clear the cache
        character_encoder.cache.clear()
        
        # Cache should be empty
        assert len(character_encoder.cache) == 0
        
        # But encoding should still work
        assert character_encoder.encode('a') == 1

    def test_cache_consistency(self, character_encoder: CharacterEncoder):
        """Test that cache entries are consistent with direct encoding."""
        # Encode some characters
        code_a = character_encoder.encode('a')
        code_b = character_encoder.encode('b')
        
        # Check cache consistency
        assert character_encoder.cache['a'] == code_a
        assert character_encoder.cache['b'] == code_b

    def test_is_terminal_functions(self, character_encoder: CharacterEncoder):
        """Test the terminal character check functions."""
        # Test sentence terminal characters
        assert character_encoder.is_terminal_sentence_char('.')
        assert character_encoder.is_terminal_sentence_char('!')
        assert character_encoder.is_terminal_sentence_char('?')
        assert not character_encoder.is_terminal_sentence_char('a')
        
        # Test paragraph terminal characters
        assert character_encoder.is_terminal_paragraph_char('\n')
        assert character_encoder.is_terminal_paragraph_char('\r')
        assert not character_encoder.is_terminal_paragraph_char('.')

    def test_benchmark_encoding(self, benchmark):
        """Benchmark character encoding performance."""
        encoder = CharacterEncoder()
        
        # Define a benchmark function that encodes a sentence
        def encode_sentence():
            for char in "This is a sample sentence for benchmarking encoding performance.":
                encoder.encode(char)
        
        # Run the benchmark
        benchmark(encode_sentence)

    def test_benchmark_with_cache(self, benchmark):
        """Benchmark encoding with a warm cache."""
        encoder = CharacterEncoder()
        
        # Pre-fill the cache
        sentence = "This is a sample sentence for benchmarking encoding performance."
        for char in sentence:
            encoder.encode(char)
        
        # Define a benchmark function that re-encodes the same sentence
        def encode_with_cache():
            for char in sentence:
                encoder.encode(char)
        
        # Run the benchmark
        benchmark(encode_with_cache)