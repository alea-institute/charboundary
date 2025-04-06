"""
Text segmentation functionality for the charboundary library.

This module re-exports the main classes from the segmenters package.
"""

from charboundary.segmenters.types import (
    SegmenterConfig,
    TextSegmenterProtocol,
    MetricsResult,
)
from charboundary.segmenters.base import TextSegmenter
from charboundary.segmenters.paragraphs import ParagraphSegmenter
from charboundary.segmenters.sentences import SentenceSegmenter
from charboundary.segmenters.spans import SpanHandler

__all__ = [
    "TextSegmenter",
    "SegmenterConfig",
    "TextSegmenterProtocol",
    "MetricsResult",
    "ParagraphSegmenter",
    "SentenceSegmenter",
    "SpanHandler",
]
