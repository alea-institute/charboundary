"""
Type definitions for the segmenters module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Protocol, TypedDict

from charboundary.constants import DEFAULT_ABBREVIATIONS


class MetricsResult(TypedDict):
    """Type definition for metrics result dictionary."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    boundary_accuracy: float
    binary_mode: bool


@dataclass
class SegmenterConfig:
    """Configuration parameters for TextSegmenter."""

    left_window: int = 5
    right_window: int = 5
    abbreviations: List[str] = field(
        default_factory=lambda: DEFAULT_ABBREVIATIONS.copy()
    )
    model_type: str = "random_forest"
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
    )
    threshold: float = 0.5  # Probability threshold for classification
    use_numpy: bool = True
    cache_size: int = 1024
    num_workers: int = 0  # Auto-detect

    # Feature selection parameters
    use_feature_selection: bool = False
    feature_selection_threshold: float = 0.01
    max_features: Optional[int] = None

    # ONNX parameters
    use_onnx: bool = False  # Whether to use ONNX for inference if available
    onnx_optimization_level: int = 1  # ONNX optimization level (0-3)


class TextSegmenterProtocol(Protocol):
    """Protocol defining the interface for text segmenters."""

    def train(self, data: Union[str, List[str]], **kwargs) -> MetricsResult:
        """Train a new model for text segmentation."""
        ...

    def segment_text(self, text: str) -> str:
        """Segment text into sentences and paragraphs."""
        ...

    def segment_to_sentences(self, text: str) -> List[str]:
        """Segment text into a list of sentences."""
        ...

    def segment_to_paragraphs(self, text: str) -> List[str]:
        """Segment text into a list of paragraphs."""
        ...

    def evaluate(self, data: Union[str, List[str]], **kwargs) -> MetricsResult:
        """Evaluate the model on a dataset."""
        ...
