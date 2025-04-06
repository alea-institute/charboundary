"""
Character span functionality for text segmentation.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING

from charboundary.constants import (
    PRIMARY_TERMINATORS,
    TERMINAL_SENTENCE_CHAR_LIST,
    TERMINAL_PARAGRAPH_CHAR_LIST,
)

# Constants for optimization
PATTERN_CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for pattern matching
CACHE_USE_THRESHOLD = 50  # Number of indices below which to use cached prediction

if TYPE_CHECKING:
    from charboundary.segmenters.base import TextSegmenter


class SpanHandler:
    """
    Handles character span functionality for text segmentation.
    """

    @staticmethod
    def find_boundary_positions(
        segmenter: "TextSegmenter", text: str, threshold: Optional[float] = None
    ) -> List[int]:
        """
        Find positions of sentence boundaries directly in the original text.

        Each position represents the index AFTER a sentence boundary character.
        For example, in "Hello. World", the boundary position would be 6 (after the period).

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            threshold (float, optional): Probability threshold for classification

        Returns:
            List[int]: List of boundary positions
        """
        if not segmenter.is_trained:
            raise ValueError("Model has not been trained yet.")

        # Use the model's threshold if none is provided
        threshold_to_use = (
            threshold if threshold is not None else segmenter.config.threshold
        )

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
            return []

        # Generate a hash of the text to avoid cache collisions between different texts
        text_hash = hash(text)

        # First, use pattern hash to quickly process common patterns if feature extractor supports it
        pattern_matched_indices = []
        pattern_matched_predictions = []
        remaining_indices = []

        if (
            hasattr(segmenter.feature_extractor, "pre_compute_patterns")
            and segmenter.feature_extractor.pre_compute_patterns
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
                            if pattern in segmenter.feature_extractor.pattern_hash:
                                # Found a pattern match
                                is_boundary, confidence = (
                                    segmenter.feature_extractor.pattern_hash[pattern]
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
        use_cache = len(remaining_indices) <= CACHE_USE_THRESHOLD

        remaining_predictions = []
        if remaining_indices:  # Only process if we have remaining indices
            if use_cache:
                # Use the cached predictor for each position
                context_window = (
                    max(segmenter.config.left_window, segmenter.config.right_window) * 2
                )

                for pos in remaining_indices:
                    # Extract context around the position
                    left_start = max(0, pos - context_window)
                    right_end = min(len(text), pos + context_window + 1)
                    left_context = text[left_start : pos + 1]
                    right_context = text[pos + 1 : right_end]

                    # Get prediction from cache
                    pred = segmenter._cached_predict(
                        text_hash, pos, left_context, right_context, threshold_to_use
                    )
                    remaining_predictions.append(pred)
            else:
                # Batch process all remaining terminal characters at once for better performance
                terminal_features = segmenter.feature_extractor.get_char_features(
                    text,
                    segmenter.config.left_window,
                    segmenter.config.right_window,
                    positions=remaining_indices,
                )

                # Predict for all terminal characters in one batch
                remaining_predictions = segmenter.model.predict(
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

        # Find positions where boundaries occur (after the terminal character)
        boundary_positions = []

        for pos, pred in zip(terminal_indices, predictions):
            if pred == 1:
                boundary_positions.append(
                    pos + 1
                )  # Position after the terminal character

        return sorted(boundary_positions)

    @staticmethod
    def get_sentence_spans_with_text(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of sentences with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
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
        if not text:
            return []

        # Find boundary positions directly
        boundary_positions = SpanHandler.find_boundary_positions(
            segmenter, text, threshold=threshold
        )

        # If no boundaries found, return the whole text as one span
        if not boundary_positions:
            return [(text, (0, len(text)))]

        # Create spans from boundary positions
        result = []
        start_idx = 0

        # Build spans from boundaries (directly converting boundaries to spans)
        for end_idx in boundary_positions:
            sentence = text[start_idx:end_idx]
            result.append((sentence, (start_idx, end_idx)))
            start_idx = end_idx

        # Add final segment if needed (for text after the last boundary)
        if start_idx < len(text):
            sentence = text[start_idx:]
            result.append((sentence, (start_idx, len(text))))

        return result

    @staticmethod
    def get_sentence_spans(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each sentence in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
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
        segments_with_spans = SpanHandler.get_sentence_spans_with_text(
            segmenter, text, streaming=streaming, threshold=threshold
        )
        return [span for _, span in segments_with_spans]

    @staticmethod
    def get_paragraph_spans_with_text(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Segment text into a list of paragraphs with their character spans.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
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
        from charboundary.segmenters.paragraphs import ParagraphSegmenter

        # Delegate to ParagraphSegmenter
        return ParagraphSegmenter.get_paragraph_spans_with_text(
            segmenter, text, streaming, threshold
        )

    @staticmethod
    def get_paragraph_spans(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get the character spans for each paragraph in the text.

        Each span is a tuple of (start_idx, end_idx) where start_idx is inclusive
        and end_idx is exclusive (following Python's slicing convention).
        The spans are guaranteed to cover the entire input text without gaps.

        Args:
            segmenter: The TextSegmenter to use
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
        segments_with_spans = SpanHandler.get_paragraph_spans_with_text(
            segmenter, text, streaming=streaming, threshold=threshold
        )
        return [span for _, span in segments_with_spans]
