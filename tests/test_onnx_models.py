"""Test loading and using the pre-trained ONNX models."""

import pytest

from charboundary import (
    get_small_onnx_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter,
    SENTENCE_TAG
)

# Try to import ONNX support - if not available, skip tests
try:
    from charboundary.onnx_support import check_onnx_available
    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

# Skip all tests in this module if ONNX is not available
pytestmark = pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX dependencies not installed")


def test_small_onnx_model():
    """Test loading and using the small ONNX model."""
    # Get the small ONNX segmenter
    segmenter = get_small_onnx_segmenter()
    
    # Verify ONNX is enabled
    assert segmenter.config.use_onnx is True
    
    # Test segmentation
    text = "Hello world. This is a test."
    sentences = segmenter.segment_to_sentences(text)
    
    # Should have at least one sentence
    assert len(sentences) >= 1
    assert "Hello world." in sentences
    
    # Test spans
    spans = segmenter.get_sentence_spans(text)
    assert spans[0][0] == 0  # First span should start at 0
    assert spans[-1][1] == len(text)  # Last span should end at text length


def test_medium_onnx_model():
    """Test loading and using the medium ONNX model."""
    # Get the medium ONNX segmenter
    segmenter = get_medium_onnx_segmenter()
    
    # Verify ONNX is enabled
    assert segmenter.config.use_onnx is True
    
    # Test segmentation
    text = "Hello world. This is a test."
    sentences = segmenter.segment_to_sentences(text)
    
    # Should have at least one sentence
    assert len(sentences) >= 1
    assert "Hello world." in sentences
    
    # Test more complex text
    complex_text = "Mr. Smith visited Washington, D.C. He met with Sen. Jones."
    complex_sentences = segmenter.segment_to_sentences(complex_text)
    assert len(complex_sentences) >= 2  # Should recognize abbreviations


def test_large_onnx_model():
    """Test loading and using the large ONNX model."""
    # Get the large ONNX segmenter
    segmenter = get_large_onnx_segmenter()
    
    # Verify ONNX is enabled
    assert segmenter.config.use_onnx is True
    
    # Test segmentation
    text = "Hello world. This is a test."
    sentences = segmenter.segment_to_sentences(text)
    
    # Should have at least one sentence
    assert len(sentences) >= 1
    assert "Hello world." in sentences
    
    # Test longer text segmentation
    long_text = """
    First sentence. Second sentence. Third sentence.
    
    Another paragraph with another sentence.
    """
    long_sentences = segmenter.segment_to_sentences(long_text)
    assert len(long_sentences) >= 4  # Should have at least 4 sentences


def test_custom_text_segmentation():
    """Test segmenting custom text with all models."""
    # Legal text with citations and abbreviations
    legal_text = """
    The court in Brown v. Board of Education, 347 U.S. 483 (1954), declared 
    that racial segregation in public schools was unconstitutional. This 
    landmark decision was delivered by Chief Justice Earl Warren.
    
    After the decision, implementation was delegated to district courts 
    with orders to desegregate "with all deliberate speed."
    """
    
    # Test all three models
    models = [
        get_small_onnx_segmenter(),
        get_medium_onnx_segmenter(),
        get_large_onnx_segmenter()
    ]
    
    for i, segmenter in enumerate(models):
        model_name = ["small", "medium", "large"][i]
        print(f"Testing {model_name} model...")
        
        # Test sentence segmentation
        sentences = segmenter.segment_to_sentences(legal_text)
        assert len(sentences) >= 2, f"{model_name} model failed to segment sentences"
        
        # Test paragraph segmentation 
        paragraphs = segmenter.segment_to_paragraphs(legal_text)
        assert len(paragraphs) >= 1, f"{model_name} model failed to segment paragraphs"
        
        # Test character spans
        sentence_spans = segmenter.get_sentence_spans(legal_text)
        assert len(sentence_spans) >= 2, f"{model_name} model failed to get sentence spans"
        
        # Verify spans cover the entire text
        assert sentence_spans[0][0] == 0, f"{model_name} model: first span doesn't start at 0"
        assert sentence_spans[-1][1] == len(legal_text), f"{model_name} model: last span doesn't end at text length"
        
        # Check full coverage of spans
        total_coverage = sum(end - start for start, end in sentence_spans)
        assert total_coverage == len(legal_text), f"{model_name} model: spans don't cover entire text"