"""
Text segmentation functionality for the charboundary library.
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
