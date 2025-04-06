"""
Base text segmentation functionality for the charboundary library.
"""

import gzip
import json
import random
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple, ClassVar

from charboundary.constants import (
    SENTENCE_TAG,
    PARAGRAPH_TAG,
    TERMINAL_SENTENCE_CHAR_LIST,
    TERMINAL_PARAGRAPH_CHAR_LIST,
    PRIMARY_TERMINATORS,
)
from charboundary.encoders import CharacterEncoder, CharacterEncoderProtocol
from charboundary.features import (
    FeatureExtractor,
    FeatureExtractorProtocol,
    FeatureMatrix,
    PositionLabels,
)
from charboundary.models import create_model, TextSegmentationModel

from charboundary.segmenters.types import SegmenterConfig, MetricsResult
from charboundary.segmenters.model_io import ModelIO
from charboundary.segmenters.sentences import SentenceSegmenter
from charboundary.segmenters.paragraphs import ParagraphSegmenter
from charboundary.segmenters.spans import SpanHandler
from charboundary.segmenters.evaluation import Evaluator

# Segmentation tuning parameters
PATTERN_CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for pattern matching
CACHE_USE_THRESHOLD = 50  # Number of indices below which to use cached prediction


class TextSegmenter:
    """
    High-level interface for training, saving, loading, and using text segmentation models.

    This simplified implementation only supports binary classification (boundary/non-boundary).

    Key features:
    - Character-level text segmentation
    - Support for sentence and paragraph boundaries
    - Customizable window sizes for context
    - Support for feature selection to improve performance
    - Trained using RandomForest classifiers
    - Support for retrieving character spans for segments

    The segmenter can be used with default parameters or customized for specific needs
    through the configuration parameters, including window sizes, model parameters,
    and feature selection options.
    """

    # Class constants for tag markers
    SENTENCE_TAG: ClassVar[str] = SENTENCE_TAG
    PARAGRAPH_TAG: ClassVar[str] = PARAGRAPH_TAG

    def __init__(
        self,
        model: Optional[TextSegmentationModel] = None,
        encoder: Optional[CharacterEncoderProtocol] = None,
        feature_extractor: Optional[FeatureExtractorProtocol] = None,
        config: Optional[SegmenterConfig] = None,
        prediction_cache_size: int = 10000,
    ):
        """
        Initialize the TextSegmenter.

        Args:
            model (TextSegmentationModel, optional): Model to use.
                If None, a model will be created when training.
            encoder (CharacterEncoderProtocol, optional): Character encoder to use.
                If None, a new one will be created.
            feature_extractor (FeatureExtractorProtocol, optional): Feature extractor to use.
                If None, a new one will be created.
            config (SegmenterConfig, optional): Configuration parameters.
                If None, default configuration will be used.
            prediction_cache_size (int, optional): Size of the prediction cache.
                Larger values use more memory but can improve performance. Defaults to 10000.
        """
        self.config = config or SegmenterConfig()

        self.encoder = encoder or CharacterEncoder()

        self.feature_extractor = feature_extractor or FeatureExtractor(
            encoder=self.encoder,
            abbreviations=self.config.abbreviations,
            use_numpy=self.config.use_numpy,
            cache_size=self.config.cache_size,
        )

        self.model = model
        self.is_trained = model is not None

        # Set up prediction cache
        self.prediction_cache_size = prediction_cache_size
        self._setup_prediction_cache(prediction_cache_size)

    def _setup_prediction_cache(self, cache_size: int) -> None:
        """Set up LRU cache for predictions."""
        # Create cache for single position predictions
        self._cached_predict = lru_cache(maxsize=cache_size)(self._predict_for_position)

    def _predict_for_position(
        self,
        text_hash: int,
        position: int,
        left_context: str,
        right_context: str,
        threshold: float,
    ) -> int:
        """
        Make a prediction for a specific position with context.

        Args:
            text_hash: Hash of the original text (to avoid collisions between different texts)
            position: Position in the original text
            left_context: Text context before and including the character at position
            right_context: Text context after the character at position
            threshold: Probability threshold for classification

        Returns:
            int: Prediction (0 or 1)
        """
        if not self.model:
            return 0

        # Combine contexts
        context = left_context + right_context

        # Calculate the position of the target character in the combined context
        target_pos = len(left_context) - 1

        # Extract features for this position and context
        features = self.feature_extractor.get_char_features(
            context,
            self.config.left_window,
            self.config.right_window,
            positions=[target_pos],
        )

        # Make prediction
        return self.model.predict(features, threshold=threshold)[0]

    def train(
        self,
        data: Union[str, List[str]],
        sample_rate: float = 0.1,
        max_samples: Optional[int] = None,
        model_type: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        left_window: Optional[int] = None,
        right_window: Optional[int] = None,
        num_workers: Optional[int] = None,
        threshold: Optional[float] = None,
        use_feature_selection: bool = False,
        feature_selection_threshold: float = 0.01,
        max_features: Optional[int] = None,
        use_onnx: bool = False,
        onnx_optimization_level: Optional[int] = None,
    ) -> MetricsResult:
        """
        Train a new model for text segmentation.

        Args:
            data (Union[str, List[str]]):
                - Path to a training data file
                - List of annotated texts
            sample_rate (float, optional): Rate at which to sample non-terminal positions.
                Defaults to 0.1.
            max_samples (int, optional): Maximum number of samples to process.
                If None, process all samples.
            model_type (str, optional): Type of model to use.
                If None, use the value from config.
            model_params (Dict[str, Any], optional): Parameters for the model.
                If None, use the values from config.
            left_window (int, optional): Size of left context window.
                If None, use the value from config.
            right_window (int, optional): Size of right context window.
                If None, use the value from config.
            num_workers (int, optional): Number of worker processes for parallel processing.
                If None, use the value from config.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                Values below 0.5 favor recall (fewer false negatives),
                values above 0.5 favor precision (fewer false positives).
                Defaults to None (which means 0.5).
            use_feature_selection (bool, optional): Whether to use feature selection.
                If True, selects important features and retrains the model.
                Defaults to False.
            feature_selection_threshold (float, optional): Importance threshold for selecting features.
                Features with importance below this threshold will be filtered out.
                Only used if use_feature_selection is True.
                Defaults to 0.01.
            max_features (int, optional): Maximum number of features to select.
                If None, use all features above the threshold.
                Only used if use_feature_selection is True.
                Defaults to None.
            use_onnx (bool, optional): Whether to use ONNX for inference if available.
                If True, the model will be converted to ONNX format after training for faster inference.
                Requires the 'onnx' optional dependency.
                Defaults to False.
            onnx_optimization_level (int, optional): ONNX optimization level (0-3) to use.
                0: No optimization
                1: Basic optimizations (default)
                2: Extended optimizations
                3: All optimizations including extended memory reuse
                Only used if use_onnx is True.
                Defaults to None (which uses the default value from config).

        Returns:
            MetricsResult: Training metrics
        """
        # Update config with new values, if provided
        if left_window is not None:
            self.config.left_window = left_window
        if right_window is not None:
            self.config.right_window = right_window
        if num_workers is not None:
            self.config.num_workers = num_workers
        if model_type is not None:
            self.config.model_type = model_type
        if model_params is not None:
            self.config.model_params.update(model_params)
        if threshold is not None:
            self.config.threshold = threshold

        # Store feature selection settings
        self.config.use_feature_selection = use_feature_selection
        self.config.feature_selection_threshold = feature_selection_threshold
        self.config.max_features = max_features

        # Store ONNX settings
        self.config.use_onnx = use_onnx
        if onnx_optimization_level is not None:
            self.config.onnx_optimization_level = onnx_optimization_level

        features: FeatureMatrix = []
        labels: PositionLabels = []

        # Process data
        if isinstance(data, str):
            # Path to a file
            if data.endswith(".jsonl.gz"):
                # Handle gzipped jsonl files
                with gzip.open(data, "rt", encoding="utf-8") as f:
                    i = 0
                    for line in f:
                        if max_samples is not None and i >= max_samples:
                            break
                        try:
                            json_obj = json.loads(line.strip())
                            if "text" in json_obj:
                                self._process_text_for_training(
                                    json_obj["text"], features, labels, sample_rate
                                )
                                i += 1
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line in {data}")
            else:
                # Handle regular text files
                with open(data, "r", encoding="utf-8") as input_file:
                    texts = input_file.readlines()
                    for i, text in enumerate(texts):
                        if max_samples is not None and i >= max_samples:
                            break
                        self._process_text_for_training(
                            text, features, labels, sample_rate
                        )
        elif isinstance(data, list):
            for i, text in enumerate(data):
                if max_samples is not None and i >= max_samples:
                    break
                self._process_text_for_training(text, features, labels, sample_rate)

        # Create and train the model
        if self.config.use_feature_selection:
            # Use feature selection model
            print(
                f"Using feature selection with threshold {self.config.feature_selection_threshold}"
            )
            self.model = create_model(
                model_type="feature_selected_rf",
                threshold=self.config.threshold,
                feature_selection_threshold=self.config.feature_selection_threshold,
                max_features=self.config.max_features,
                use_onnx=self.config.use_onnx,
                onnx_optimization_level=self.config.onnx_optimization_level,
                **(self.config.model_params),
            )
        else:
            # Use regular model
            self.model = create_model(
                model_type=self.config.model_type,
                threshold=self.config.threshold,
                use_onnx=self.config.use_onnx,
                onnx_optimization_level=self.config.onnx_optimization_level,
                **(self.config.model_params),
            )

        # Print debug info about the training data
        print(f"Training on {len(features)} samples...")
        print(
            f"Window sizes: left={self.config.left_window}, right={self.config.right_window}"
        )
        print(f"Positive samples (boundaries): {labels.count(1)}")
        print(f"Negative samples (non-boundaries): {labels.count(0)}")
        print(f"Positive ratio: {labels.count(1) / len(labels) if labels else 0:.4f}")

        # Fit the model
        self.model.fit(X=features, y=labels)
        self.is_trained = True

        # Print feature selection info if available
        if self.config.use_feature_selection and hasattr(
            self.model, "get_feature_importances"
        ):
            feature_info = self.model.get_feature_importances()
            orig_features = feature_info.get("original_num_features", 0)
            selected_features = feature_info.get("selected_num_features", 0)

            if orig_features > 0:
                print(
                    f"Feature selection reduced dimensions from {orig_features} to {selected_features} features "
                    f"({selected_features / orig_features:.1%} of original)"
                )

                # Print top 10 most important features
                if (
                    "selected_indices" in feature_info
                    and "original_importances" in feature_info
                ):
                    indices = feature_info["selected_indices"][:10]  # Get top 10
                    importances = feature_info["original_importances"]

                    print("\nTop 10 most important features:")
                    for i, idx in enumerate(indices, 1):
                        print(
                            f"  {i}. Feature {idx}: importance={importances[idx]:.4f}"
                        )
                    print("")

        # Evaluate on training data
        report = self.model.get_metrics(features, labels)

        return report

    def _process_text_for_training(
        self,
        text: str,
        features: FeatureMatrix,
        labels: PositionLabels,
        sample_rate: float = 0.1,
    ) -> None:
        """
        Process a text for training and add its features and labels to the provided lists.

        Args:
            text (str): Annotated text
            features (FeatureMatrix): List to which features will be added
            labels (PositionLabels]): List to which labels will be added
            sample_rate (float, optional): Rate at which to sample non-terminal positions.
                Defaults to 0.1.
        """
        clean_text, text_features, text_labels = (
            self.feature_extractor.process_annotated_text(
                text,
                self.config.left_window,
                self.config.right_window,
                self.config.num_workers,
            )
        )

        # Always include terminal characters and a sample of non-terminal characters
        for j, (char, feature_vec, label) in enumerate(
            zip(clean_text, text_features, text_labels)
        ):
            is_terminal = (
                char in TERMINAL_SENTENCE_CHAR_LIST
                or char in TERMINAL_PARAGRAPH_CHAR_LIST
            )

            # Use modern Python 3.11 pattern matching for cleaner code
            match (label, is_terminal, random.random() < sample_rate):
                case (1, _, _):  # Always include positive samples (boundaries)
                    features.append(feature_vec)
                    labels.append(label)
                case (_, True, _):  # Always include terminal characters
                    features.append(feature_vec)
                    labels.append(label)
                case (_, _, True):  # Sample some non-terminals based on rate
                    features.append(feature_vec)
                    labels.append(label)
                case _:  # Skip other non-terminal characters
                    pass

    def save(
        self,
        path: str,
        format: str = "skops",
        compress: bool = True,
        compression_level: int = 9,
    ) -> None:
        """
        Save the model and configuration to a file.

        Args:
            path (str): Path to save the model
            format (str, optional): Serialization format to use ('skops' or 'pickle').
                                    Defaults to 'skops' for secure serialization.
            compress (bool, optional): Whether to use compression. Defaults to True.
            compression_level (int, optional): Compression level (0-9, where 9 is highest).
                                              Defaults to 9.
        """
        ModelIO.save(self, path, format, compress, compression_level)

    @classmethod
    def load(
        cls, path: str, use_skops: bool = True, trust_model: bool = False
    ) -> "TextSegmenter":
        """
        Load a model and configuration from a file.

        Args:
            path (str): Path to load the model from
            use_skops (bool, optional): Whether to use skops to load the model. Defaults to True.
            trust_model (bool, optional): Whether to trust all types in the model file.
                                         Set to True only if you trust the source of the model file.
                                         Defaults to False.

        Returns:
            TextSegmenter: Loaded TextSegmenter instance
        """
        return ModelIO.load(path, cls, use_skops, trust_model)

    def segment_text(self, text: str, threshold: Optional[float] = None) -> str:
        """
        Segment text into sentences and paragraphs.

        Args:
            text (str): Text to segment
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            str: Text with sentence and paragraph annotations
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        # Use the model's threshold if none is provided
        threshold_to_use = threshold if threshold is not None else self.config.threshold

        # Extract features for terminal characters only - optimized approach
        terminal_indices = []

        # Pre-identify all terminal characters to batch process them
        for i, char in enumerate(text):
            if (
                char in TERMINAL_SENTENCE_CHAR_LIST
                or char in TERMINAL_PARAGRAPH_CHAR_LIST
            ):
                terminal_indices.append(i)

        # Skip feature extraction if no terminal characters found
        if not terminal_indices:
            return text

        # Generate a hash of the text to avoid cache collisions between different texts
        text_hash = hash(text)

        # First, use pattern hash to quickly process common patterns if feature extractor supports it
        pattern_matched_indices = []
        pattern_matched_predictions = []
        remaining_indices = []

        if (
            hasattr(self.feature_extractor, "pre_compute_patterns")
            and self.feature_extractor.pre_compute_patterns
        ):
            for i, pos in enumerate(terminal_indices):
                # Try to match common patterns first
                pattern_found = False

                # Check if this character can form a pattern (must be one of primary terminators)
                if pos < len(text) and text[pos] in PRIMARY_TERMINATORS:
                    # For each possible pattern length (typically 2-6 chars)
                    for pattern_len in range(
                        2, 7
                    ):  # Look for patterns up to 6 chars long
                        # Make sure we have enough characters to check
                        if pos + pattern_len <= len(text):
                            # Extract the potential pattern
                            pattern = text[pos : pos + pattern_len]
                            if pattern in self.feature_extractor.pattern_hash:
                                # Found a pattern match
                                is_boundary, confidence = (
                                    self.feature_extractor.pattern_hash[pattern]
                                )

                                # Only use the pattern if confidence exceeds threshold
                                if confidence > PATTERN_CONFIDENCE_THRESHOLD:
                                    pattern_matched_indices.append(pos)
                                    pattern_matched_predictions.append(
                                        1 if is_boundary else 0
                                    )
                                    pattern_found = True
                                    break

                # If no pattern match was found, add to remaining indices
                if not pattern_found:
                    remaining_indices.append(pos)
        else:
            # If pattern matching is not enabled, process all indices
            remaining_indices = terminal_indices

        # Decide whether to use the cache or batch processing for remaining positions
        # For small numbers of positions, cached position-by-position prediction might be faster
        # For large numbers of positions, batch processing with vectorization might be better
        use_cache = len(remaining_indices) <= CACHE_USE_THRESHOLD

        remaining_predictions = []
        if remaining_indices:  # Only process if we have remaining indices
            if use_cache:
                # Use the cached predictor for each position
                context_window = (
                    max(self.config.left_window, self.config.right_window) * 2
                )

                for pos in remaining_indices:
                    # Extract context around the position
                    left_start = max(0, pos - context_window)
                    right_end = min(len(text), pos + context_window + 1)
                    left_context = text[left_start : pos + 1]
                    right_context = text[pos + 1 : right_end]

                    # Get prediction from cache
                    pred = self._cached_predict(
                        text_hash, pos, left_context, right_context, threshold_to_use
                    )
                    remaining_predictions.append(pred)
            else:
                # Batch process all remaining terminal characters at once for better performance
                terminal_features = self.feature_extractor.get_char_features(
                    text,
                    self.config.left_window,
                    self.config.right_window,
                    positions=remaining_indices,
                )

                # Predict for all terminal characters in one batch
                remaining_predictions = self.model.predict(
                    terminal_features, threshold=threshold_to_use
                )

        # Reassemble predictions in the correct order
        if pattern_matched_indices:
            # Need to reconstruct the predictions array in the right order
            combined_indices = pattern_matched_indices + remaining_indices
            combined_predictions = pattern_matched_predictions + remaining_predictions

            # Create mapping from position to prediction
            pos_to_pred = {
                pos: pred for pos, pred in zip(combined_indices, combined_predictions)
            }

            # Reassemble predictions in the original terminal_indices order
            predictions = [pos_to_pred[pos] for pos in terminal_indices]
        else:
            # All predictions came from the regular path
            predictions = remaining_predictions

        # Optimization: only create result list if we have boundaries
        if not any(predictions):
            return text

        # Apply segmentation with special handling for quotes
        result = list(text)

        # Insert tags from end to beginning to maintain correct indices
        # Pre-reverse the arrays for better performance
        reversed_indices = terminal_indices[::-1]
        reversed_predictions = predictions[::-1]

        # Track quote positions to handle special cases with quotation marks
        quote_positions = set()
        for i, char in enumerate(text):
            if char == '"' or char == '"' or char == '"':
                quote_positions.add(i)

        # Boundary detection and tag insertion
        for pos, pred in zip(reversed_indices, reversed_predictions):
            if pred == 1:
                char = text[pos]
                # Insert tags after this character
                insert_pos = pos + 1

                # Add paragraph tag for paragraph terminators (after sentence tag)
                if char in TERMINAL_PARAGRAPH_CHAR_LIST:
                    result.insert(insert_pos, self.PARAGRAPH_TAG)

                # Handle quotation marks that shouldn't be sentence boundaries
                # If this is a quote followed by text without whitespace, it's likely not a boundary
                if pos in quote_positions and pos + 1 < len(text):
                    # If quote is followed by another character that should continue the sentence,
                    # don't treat this as a boundary
                    next_char = text[pos + 1]
                    if next_char != " " and next_char != "\n" and next_char != "\t":
                        continue

                # Add sentence tag for all boundaries
                result.insert(insert_pos, self.SENTENCE_TAG)

        return "".join(result)

    def segment_text_streaming(
        self, text: str, chunk_size: int = 10000, overlap: int = 100
    ) -> Iterator[str]:
        """
        Memory-efficient streaming text segmentation.

        Args:
            text (str): Text to segment
            chunk_size (int, optional): Size of chunks to process at a time. Defaults to 10000.
            overlap (int, optional): Overlap between chunks to maintain context. Defaults to 100.

        Yields:
            Iterator[str]: Stream of segmented text fragments
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        # For small texts, just use the regular segmenter
        if len(text) <= chunk_size:
            yield self.segment_text(text)
            return

        # Process text in overlapping chunks
        text_len = len(text)
        position = 0

        while position < text_len:
            # Calculate chunk bounds with overlap
            chunk_end = min(position + chunk_size, text_len)

            # Extract chunk with context
            if position > 0:
                # Ensure we have enough context for the first chunk
                context_start = max(0, position - overlap)
                chunk = text[context_start:chunk_end]
                context_len = position - context_start
            else:
                # First chunk has no left context
                chunk = text[position:chunk_end]
                context_len = 0

            # Add right context if not at the end
            if chunk_end < text_len:
                right_context_end = min(text_len, chunk_end + overlap)
                chunk += text[chunk_end:right_context_end]

            # Segment the chunk
            segmented_chunk = self.segment_text(chunk)

            # Remove context from the output
            if context_len > 0:
                # Find the first boundary after the context
                i = context_len
                while i < len(segmented_chunk) and not (
                    segmented_chunk[i : i + len(self.SENTENCE_TAG)] == self.SENTENCE_TAG
                    or segmented_chunk[i : i + len(self.PARAGRAPH_TAG)]
                    == self.PARAGRAPH_TAG
                ):
                    i += 1

                segmented_chunk = segmented_chunk[i:]

            # Find the position to cut off the right context
            if chunk_end < text_len:
                # Find the last boundary before the right context
                actual_chunk_len = chunk_end - position
                if context_len > 0:
                    actual_chunk_len = chunk_end - context_start

                i = actual_chunk_len
                while i > 0 and not (
                    segmented_chunk[i - len(self.SENTENCE_TAG) : i] == self.SENTENCE_TAG
                    or segmented_chunk[i - len(self.PARAGRAPH_TAG) : i]
                    == self.PARAGRAPH_TAG
                ):
                    i -= 1

                if i > 0:
                    segmented_chunk = segmented_chunk[:i]

            # Yield the cleaned-up segmented chunk
            yield segmented_chunk

            # Move to the next chunk, accounting for any boundaries we found
            position = chunk_end

    # Sentence methods
    def segment_to_sentences(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Segment text into a list of sentences.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                       Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[str]: List of sentences
        """
        return SentenceSegmenter.segment_to_sentences(self, text, streaming, threshold)

    def segment_to_sentences_with_spans(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of sentences with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[str, tuple[int, int]]]: List of tuples containing (sentence, (start_index, end_index))
        """
        return SpanHandler.get_sentence_spans_with_text(
            self, text, streaming, threshold
        )

    def get_sentence_spans(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each sentence in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[int, int]]: List of character spans (start_index, end_index)
        """
        return SpanHandler.get_sentence_spans(self, text, streaming, threshold)

    # Paragraph methods
    def segment_to_paragraphs(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Segment text into a list of paragraphs.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                       Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[str]: List of paragraphs
        """
        return ParagraphSegmenter.segment_to_paragraphs(
            self, text, streaming, threshold
        )

    def segment_to_paragraphs_with_spans(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of paragraphs with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[str, tuple[int, int]]]: List of tuples containing (paragraph, (start_index, end_index))
        """
        return ParagraphSegmenter.get_paragraph_spans_with_text(
            self, text, streaming, threshold
        )

    def get_paragraph_spans(
        self, text: str, streaming: bool = False, threshold: Optional[float] = None
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each paragraph in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                      Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[tuple[int, int]]: List of character spans (start_index, end_index)
        """
        return ParagraphSegmenter.get_paragraph_spans(self, text, streaming, threshold)

    # Abbreviation management
    def get_abbreviations(self) -> List[str]:
        """
        Get the current list of abbreviations.

        Returns:
            List[str]: The current list of abbreviations
        """
        return sorted(list(self.config.abbreviations))

    def add_abbreviation(self, abbreviation: str) -> None:
        """
        Add a new abbreviation to the list.

        Args:
            abbreviation (str): The abbreviation to add (must end with a period)
        """
        if not abbreviation.endswith("."):
            abbreviation = abbreviation + "."

        # Update both the segmenter's abbreviation list and the feature extractor's
        self.config.abbreviations.append(abbreviation)
        self.feature_extractor.abbreviations.add(abbreviation)

    def remove_abbreviation(self, abbreviation: str) -> bool:
        """
        Remove an abbreviation from the list.

        Args:
            abbreviation (str): The abbreviation to remove

        Returns:
            bool: True if the abbreviation was removed, False if it wasn't in the list
        """
        if not abbreviation.endswith("."):
            abbreviation = abbreviation + "."

        if abbreviation in self.config.abbreviations:
            self.config.abbreviations.remove(abbreviation)
            if abbreviation in self.feature_extractor.abbreviations:
                self.feature_extractor.abbreviations.remove(abbreviation)
            return True
        return False

    def set_abbreviations(self, abbreviations: List[str]) -> None:
        """
        Set the complete list of abbreviations, replacing the current list.

        Args:
            abbreviations (List[str]): The new list of abbreviations
        """
        # Ensure all abbreviations end with periods
        self.config.abbreviations = [
            abbr if abbr.endswith(".") else abbr + "." for abbr in abbreviations
        ]

        # Update the feature extractor's abbreviations
        self.feature_extractor.abbreviations = set(self.config.abbreviations)

    # Evaluation methods
    def evaluate(
        self, data: Union[str, List[str]], max_samples: Optional[int] = None
    ) -> MetricsResult:
        """
        Evaluate the model on a dataset.

        Args:
            data (Union[str, List[str]]):
                - Path to a test data file
                - List of annotated texts
            max_samples (int, optional): Maximum number of samples to process.
                If None, process all samples.

        Returns:
            MetricsResult: Evaluation metrics
        """
        return Evaluator.evaluate(self, data, max_samples)

    # ONNX support
    def to_onnx(self) -> Optional[bytes]:
        """
        Convert the model to ONNX format.

        Returns:
            Optional[bytes]: Serialized ONNX model if conversion was successful, None otherwise

        Raises:
            ImportError: If ONNX dependencies are not installed
            ValueError: If the model has not been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        if not hasattr(self.model, "to_onnx"):
            raise NotImplementedError("ONNX conversion not supported for this model.")

        return self.model.to_onnx()

    def save_onnx(self, path: str, compress: bool = True) -> bool:
        """
        Save the model in ONNX format, optionally with XZ compression.

        Args:
            path (str): Path to save the ONNX model
            compress (bool): Whether to compress the model with XZ (default: True)

        Returns:
            bool: True if the model was saved successfully, False otherwise

        Raises:
            ImportError: If ONNX is not installed
            ValueError: If the model has not been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        if not hasattr(self.model, "save_onnx"):
            raise NotImplementedError("ONNX conversion not supported for this model.")

        return self.model.save_onnx(path, compress=compress)

    def load_onnx(self, path: str) -> bool:
        """
        Load an ONNX model.

        Args:
            path (str): Path to the ONNX model file

        Returns:
            bool: True if the model was loaded successfully, False otherwise

        Raises:
            ImportError: If ONNX is not installed
        """
        if not hasattr(self.model, "load_onnx"):
            raise NotImplementedError("ONNX loading not supported for this model.")

        return self.model.load_onnx(path)

    def enable_onnx(self, enable: bool = True) -> bool:
        """
        Enable or disable ONNX inference.

        Args:
            enable (bool): Whether to enable ONNX inference

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not hasattr(self.model, "enable_onnx"):
            raise NotImplementedError("ONNX inference not supported for this model.")

        result = self.model.enable_onnx(enable)

        if result:
            # Update config to match the model's ONNX state
            self.config.use_onnx = enable

        return result
