"""
Feature extraction functionality for the charboundary library.
"""

from typing import List, Tuple, Optional, Protocol, TypeAlias
from functools import lru_cache
import multiprocessing
from multiprocessing import Pool
from functools import partial

# Try to import numpy - if not available, we'll use list-based processing
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from charboundary.constants import (
    SENTENCE_TAG,
    PARAGRAPH_TAG,
    TERMINAL_SENTENCE_CHAR_LIST,
    DEFAULT_ABBREVIATIONS,
    PRIMARY_TERMINATORS,
    SECONDARY_TERMINATORS,
    OPENING_QUOTES,
    CLOSING_QUOTES,
    PUNCTUATION_CHAR_LIST,
    LIST_MARKERS,
    LIST_CONJUNCTIONS,
    LIST_INTROS,
)
from charboundary.encoders import CharacterEncoder, CharacterEncoderProtocol

# Type aliases for better readability
FeatureVector: TypeAlias = List[int]
FeatureMatrix: TypeAlias = List[FeatureVector]
PositionLabels: TypeAlias = List[int]
TextEncoding: TypeAlias = List[int]


class FeatureExtractorProtocol(Protocol):
    """Protocol defining the interface for feature extractors."""

    def get_char_features(
        self, text: str, left_window: int = 5, right_window: int = 5
    ) -> FeatureMatrix:
        """Extract character-level features from text."""
        ...

    def mark_annotation_positions(self, text: str) -> PositionLabels:
        """Mark positions of sentence and paragraph annotations in text."""
        ...

    def process_annotated_text(
        self, text: str, left_window: int = 5, right_window: int = 5
    ) -> Tuple[str, FeatureMatrix, PositionLabels]:
        """Process annotated text to extract features and labels."""
        ...


class FeatureExtractor:
    """
    Extracts character-level features from text using windowing.

    This extractor creates a window of encoded characters around each position
    in the text, generating a fixed-size feature vector for each character.
    """

    def __init__(
        self,
        encoder: Optional[CharacterEncoderProtocol] = None,
        abbreviations: Optional[List[str]] = None,
        use_numpy: bool = NUMPY_AVAILABLE,
        cache_size: int = 1024,
    ):
        """
        Initialize the FeatureExtractor.

        Args:
            encoder (CharacterEncoderProtocol, optional): Character encoder to use.
                If None, a new CharacterEncoder will be created.
            abbreviations (List[str], optional): List of abbreviations to use for feature extraction.
                If None, the default abbreviations list will be used.
            use_numpy (bool, optional): Whether to use NumPy for vectorized operations
                if available. Defaults to True if NumPy is installed.
            cache_size (int, optional): Size of internal LRU caches. Defaults to 1024.
        """
        self.encoder = encoder or CharacterEncoder()
        # Use a set for O(1) lookup time
        abbr_list = (
            abbreviations if abbreviations is not None else DEFAULT_ABBREVIATIONS.copy()
        )
        self.abbreviations = set(abbr_list)
        self.use_numpy = use_numpy and NUMPY_AVAILABLE
        self._setup_caches(cache_size)

    def _setup_caches(self, cache_size: int) -> None:
        """Setup LRU caches for frequently used methods."""
        # Create LRU cached version of is_in_abbreviation for better performance
        self.is_in_abbreviation = lru_cache(maxsize=cache_size)(
            self._is_in_abbreviation
        )

    def _is_in_abbreviation(self, text: str, position: int) -> bool:
        """
        Check if a character at a given position is part of a common abbreviation.

        Args:
            text (str): The text to check
            position (int): The position of the character to check

        Returns:
            bool: True if the character is part of an abbreviation, False otherwise
        """
        # If not a period, can't be part of an abbreviation ending
        if position >= len(text) or text[position] != ".":
            return False

        # Look back to find the start of the word
        word_start = position
        while word_start > 0 and (
            text[word_start - 1].isalnum() or text[word_start - 1] == "."
        ):
            word_start -= 1

        # Extract the potential abbreviation (including the period)
        potential_abbr = text[word_start : position + 1]

        # Check if it's in our configured abbreviations set
        return potential_abbr in self.abbreviations

    def _is_quote_balanced(self, text: str, position: int) -> bool:
        """
        Check if quotes are balanced at a given position.

        Args:
            text (str): The text to check
            position (int): The position to check

        Returns:
            bool: True if the quotation marks are balanced, False otherwise
        """
        if position >= len(text):
            return True

        # Only check for quote characters
        if (
            text[position] not in OPENING_QUOTES
            and text[position] not in CLOSING_QUOTES
        ):
            return True

        # Count quotes up to this position
        straight_double_count = text[: position + 1].count('"')
        curly_double_open_count = text[: position + 1].count(
            "\u201c"
        )  # left double quote
        curly_double_close_count = text[: position + 1].count(
            "\u201d"
        )  # right double quote
        straight_single_count = text[: position + 1].count("'")
        curly_single_open_count = text[: position + 1].count(
            "\u2018"
        )  # left single quote
        curly_single_close_count = text[: position + 1].count(
            "\u2019"
        )  # right single quote

        # Check if quote counts are balanced
        is_straight_double_balanced = straight_double_count % 2 == 0
        is_curly_double_balanced = curly_double_open_count == curly_double_close_count
        is_straight_single_balanced = straight_single_count % 2 == 0
        is_curly_single_balanced = curly_single_open_count == curly_single_close_count

        return (
            is_straight_double_balanced
            and is_curly_double_balanced
            and is_straight_single_balanced
            and is_curly_single_balanced
        )

    def _is_word_likely_complete(self, text: str, position: int) -> bool:
        """
        Check if a word appears to be complete at a given position.

        Args:
            text (str): The text to check
            position (int): The position to check

        Returns:
            bool: True if the word appears complete, False if it seems cut off
        """
        if position >= len(text):
            return True

        # If the character is not a quote, don't apply this rule
        if text[position] not in CLOSING_QUOTES:
            return True

        # Check if there's more text after this position
        if position + 1 >= len(text):
            return True  # End of text, word is complete

        # If next character is whitespace, punctuation, or terminal, word is likely complete
        next_char = text[position + 1]
        if (
            next_char.isspace()
            or next_char in TERMINAL_SENTENCE_CHAR_LIST
            or next_char in PUNCTUATION_CHAR_LIST
        ):
            return True

        # If the next character is lowercase, the word might be cut off
        if position + 1 < len(text) and text[position + 1].islower():
            return False

        return True

    def _is_in_list_item(self, text: str, position: int, window_size: int = 20) -> bool:
        """
        Check if a character at a given position is part of a list item.

        Args:
            text (str): The text to check
            position (int): The position to check
            window_size (int): Size of the window to look for list markers

        Returns:
            bool: True if the character is part of a list item, False otherwise
        """
        if position >= len(text):
            return False

        # Cache text length for faster repeated access
        text_len = len(text)

        # Optimize common case: check if character is part of list marker with fixed-length substrings
        char = text[position] if position < text_len else ""

        # Skip checking if character is not likely part of a list marker
        if char not in "()0123456789abcdefghijklmnopqrstuvwxyz.•·○●■□▪▫":
            # If not a character typically found in list markers, do quick checks for context
            if position > 0 and position < text_len - 1:
                # Only check for list context if the character is a boundary character
                if char in ".,;:" or text[position - 1] in ".,;:":
                    # Check only right after potential list markers or boundaries
                    pass
                else:
                    return False
            else:
                return False

        # Check if this character is part of a list marker
        # Use a more efficient approach by checking fixed patterns
        for marker in LIST_MARKERS:
            marker_len = len(marker)

            # Check if the marker appears right before this position
            marker_start = position - marker_len
            if marker_start >= 0 and text[marker_start:position] == marker:
                return True

            # Check if the marker appears right after this position
            marker_end = position + marker_len
            if marker_end <= text_len and text[position:marker_end] == marker:
                return True

        # Only do expensive context checks if at a potential list boundary
        if (
            char in ".,;:()[]"
            or (position > 0 and text[position - 1] in ".,;:()")
            or (position < text_len - 1 and text[position + 1] in ".,;:()[]")
        ):
            # Check for broader context - is this part of a list structure?
            # Look for beginning of list within window - but only if likely to be a list
            start_idx = max(0, position - window_size)
            context_before = text[start_idx:position]

            # Use substring search instead of 'in' for better performance
            # Precompute lower bound lengths to avoid unnecessary searches
            min_intro_len = 5  # Minimum length of a list intro
            has_colon_before = (
                ":" in context_before[-min_intro_len:]
                if len(context_before) >= min_intro_len
                else False
            )

            # Only search for list intros if there's a colon nearby
            if has_colon_before:
                for intro in LIST_INTROS:
                    if intro in context_before:
                        return True

            # Look for list conjunction (like "and", "or") that often indicates a list
            # Only if character is a semicolon or comma
            if char in ";," or (position > 0 and text[position - 1] in ";,"):
                end_idx = min(text_len, position + window_size)
                context_after = text[position:end_idx]

                for conj in LIST_CONJUNCTIONS:
                    conj_pos = context_after.find(conj)
                    if (
                        conj_pos > 0
                        and conj_pos < 5
                        and (
                            context_after[conj_pos - 1] == ";"
                            or context_after[conj_pos - 1] == ","
                        )
                    ):
                        return True

        return False

    def _is_semicolon_in_list(
        self, text: str, position: int, window_size: int = 50
    ) -> bool:
        """
        Check if a semicolon at a given position is part of a list structure.

        Args:
            text (str): The text to check
            position (int): The position to check
            window_size (int): Size of the window to look for list context

        Returns:
            bool: True if the semicolon is part of a list, False otherwise
        """
        if position >= len(text) or text[position] != ";":
            return False

        # Look for list markers within the window before and after
        start_idx = max(0, position - window_size)
        end_idx = min(len(text), position + window_size)

        context = text[start_idx:end_idx]

        # Count semicolons in the context - multiple semicolons often indicate a list
        semicolon_count = context.count(";")
        if semicolon_count >= 2:  # This semicolon plus at least one more
            return True

        # Look for list markers around this semicolon
        for marker in LIST_MARKERS:
            if marker in context:
                return True

        # Check for list introduction markers
        for intro in LIST_INTROS:
            if intro in text[start_idx:position]:
                return True

        # Check for conjunctions that might indicate the end of a list
        for conj in LIST_CONJUNCTIONS:
            if conj in text[position:end_idx]:
                return True

        return False

    def _is_near_colon(self, text: str, position: int, window_size: int = 10) -> bool:
        """
        Check if a character is near a colon, which often introduces a list.

        Args:
            text (str): The text to check
            position (int): The position to check
            window_size (int): Size of the window to look for a colon

        Returns:
            bool: True if there's a colon nearby before the position, False otherwise
        """
        start_idx = max(0, position - window_size)
        context = text[start_idx:position]
        return ":" in context

    def get_char_features(
        self,
        text: str,
        left_window: int = 5,
        right_window: int = 5,
        positions: Optional[List[int]] = None,
    ) -> FeatureMatrix:
        """
        Extract character-level features from a given text.

        Args:
            text (str): The text to extract features from
            left_window (int, optional): Size of left context window. Defaults to 5.
            right_window (int, optional): Size of right context window. Defaults to 5.
            positions (List[int], optional): Specific character positions to extract features for.
                If None, extract features for all characters. Defaults to None.

        Returns:
            FeatureMatrix: Character-level features for each character in the text (or specified positions)
        """
        n = len(text)
        window_size = left_window + right_window + 1

        # Add more features for better boundary detection:
        # 1. Abbreviation detection
        # 2. Primary vs secondary terminator
        # 3. Quote balance
        # 4. Word completion (for quotes)
        # 5. Is followed by lowercase letter
        # 6. Is in list item
        # 7. Is semicolon in list
        # 8. Is near colon
        feature_size = window_size + 8

        # Use NumPy implementation if available and enabled
        if self.use_numpy:
            return self._get_char_features_numpy(
                text, left_window, right_window, positions
            )

        # Pre-encode all characters to avoid repeated function calls
        encoded_chars = [self.encoder.encode(c) for c in text]

        # Pre-compute abbreviation flags for the entire text
        abbr_flags = [self.is_in_abbreviation(text, i) for i in range(n)]

        # Pre-compute additional features
        primary_term_flags = [
            1 if (i < n and text[i] in PRIMARY_TERMINATORS) else 0 for i in range(n)
        ]
        secondary_term_flags = [
            1 if (i < n and text[i] in SECONDARY_TERMINATORS) else 0 for i in range(n)
        ]

        # Compute quote balance for all positions
        quote_balanced_flags = [self._is_quote_balanced(text, i) for i in range(n)]

        # Check word completion status for quotes
        word_complete_flags = [self._is_word_likely_complete(text, i) for i in range(n)]

        # Check if followed by lowercase (useful for quotes)
        followed_by_lowercase_flags = [
            1 if (i < n - 1 and text[i + 1].islower()) else 0 for i in range(n)
        ]

        # Check list-related features
        in_list_item_flags = [self._is_in_list_item(text, i) for i in range(n)]
        semicolon_in_list_flags = [
            self._is_semicolon_in_list(text, i) for i in range(n)
        ]
        near_colon_flags = [self._is_near_colon(text, i) for i in range(n)]

        # Determine which positions to extract features for
        if positions is None:
            position_indices = list(range(n))
        else:
            position_indices = [p for p in positions if 0 <= p < n]

        features = [[0] * feature_size for _ in range(len(position_indices))]

        # Fill in features with sliding window for specified positions
        for feature_idx, i in enumerate(position_indices):
            # Add character window features
            for j in range(-left_window, right_window + 1):
                idx = j + left_window
                pos = i + j

                if pos < 0 or pos >= n:
                    features[feature_idx][
                        idx
                    ] = -3  # Out-of-bounds placeholder (whitespace value)
                else:
                    features[feature_idx][idx] = encoded_chars[pos]

            # Add additional features at the end of the feature vector
            window_end = window_size

            # Feature 1: Abbreviation
            features[feature_idx][window_end] = 1 if abbr_flags[i] else 0

            # Feature 2: Primary/secondary terminator
            features[feature_idx][window_end + 1] = (
                1 if primary_term_flags[i] else (-1 if secondary_term_flags[i] else 0)
            )

            # Feature 3: Quote balance
            features[feature_idx][window_end + 2] = 1 if quote_balanced_flags[i] else 0

            # Feature 4: Word completion status
            features[feature_idx][window_end + 3] = 1 if word_complete_flags[i] else 0

            # Feature 5: Followed by lowercase
            features[feature_idx][window_end + 4] = (
                1 if followed_by_lowercase_flags[i] else 0
            )

            # Feature 6: Is in list item
            features[feature_idx][window_end + 5] = 1 if in_list_item_flags[i] else 0

            # Feature 7: Is semicolon in list
            features[feature_idx][window_end + 6] = (
                1 if semicolon_in_list_flags[i] else 0
            )

            # Feature 8: Is near colon
            features[feature_idx][window_end + 7] = 1 if near_colon_flags[i] else 0

        return features

    def _get_char_features_numpy(
        self,
        text: str,
        left_window: int = 5,
        right_window: int = 5,
        positions: Optional[List[int]] = None,
    ) -> FeatureMatrix:
        """
        NumPy-optimized version of feature extraction.

        Args:
            text (str): The text to extract features from
            left_window (int): Size of left context window
            right_window (int): Size of right context window
            positions (List[int], optional): Specific positions to extract features for

        Returns:
            FeatureMatrix: Character-level features
        """
        n = len(text)
        window_size = left_window + right_window + 1
        feature_size = window_size + 8  # +8 for additional features

        # Pre-encode all characters as a NumPy array - memoized for performance
        encoded_chars = np.array([self.encoder.encode(c) for c in text], dtype=np.int32)

        # Determine positions early to avoid unnecessary computation for unused positions
        if positions is None:
            position_indices = np.arange(n, dtype=np.int32)
        else:
            position_indices = np.array(
                [p for p in positions if 0 <= p < n], dtype=np.int32
            )

        # If positions is specified and small, only compute features for those positions
        if (
            positions is not None and len(position_indices) < n / 4
        ):  # Only compute for positions if less than 25% of text
            # More efficient to compute features only for needed positions
            num_positions = len(position_indices)

            # Pre-compute feature flags only for positions we care about
            abbr_flags = np.array(
                [self.is_in_abbreviation(text, i) for i in position_indices],
                dtype=np.int8,
            )
            primary_term_flags = np.array(
                [1 if text[i] in PRIMARY_TERMINATORS else 0 for i in position_indices],
                dtype=np.int8,
            )
            secondary_term_flags = np.array(
                [
                    1 if text[i] in SECONDARY_TERMINATORS else 0
                    for i in position_indices
                ],
                dtype=np.int8,
            )
            quote_balanced_flags = np.array(
                [self._is_quote_balanced(text, i) for i in position_indices],
                dtype=np.int8,
            )
            word_complete_flags = np.array(
                [self._is_word_likely_complete(text, i) for i in position_indices],
                dtype=np.int8,
            )
            followed_by_lowercase_flags = np.array(
                [
                    1 if (i < n - 1 and text[i + 1].islower()) else 0
                    for i in position_indices
                ],
                dtype=np.int8,
            )

            # List-related features - compute only for potential list positions
            in_list_item_flags = np.zeros(num_positions, dtype=np.int8)
            semicolon_in_list_flags = np.zeros(num_positions, dtype=np.int8)
            near_colon_flags = np.zeros(num_positions, dtype=np.int8)

            # Only compute expensive list-related features for positions that might be in lists
            for idx, pos in enumerate(position_indices):
                char = text[pos]
                # Only check for lists if this could be a list-related character
                if (
                    char in ".,;:()[]0123456789abcdefghijklmnopqrstuvwxyz•·○●■□▪▫"
                    or (pos > 0 and text[pos - 1] in ".,;:()")
                    or (pos < n - 1 and text[pos + 1] in ".,;:()")
                ):
                    in_list_item_flags[idx] = self._is_in_list_item(text, pos)
                    if char == ";":
                        semicolon_in_list_flags[idx] = self._is_semicolon_in_list(
                            text, pos
                        )
                    if (
                        char == ":"
                        or (pos > 0 and text[pos - 1] == ":")
                        or (pos < n - 1 and text[pos + 1] == ":")
                    ):
                        near_colon_flags[idx] = self._is_near_colon(text, pos)
        else:
            # For full text or large number of positions, traditional vectorized approach
            # Pre-compute feature flags for all positions
            abbr_flags = np.array(
                [self.is_in_abbreviation(text, i) for i in range(n)], dtype=np.int8
            )
            primary_term_flags = np.array(
                [
                    1 if (i < n and text[i] in PRIMARY_TERMINATORS) else 0
                    for i in range(n)
                ],
                dtype=np.int8,
            )
            secondary_term_flags = np.array(
                [
                    1 if (i < n and text[i] in SECONDARY_TERMINATORS) else 0
                    for i in range(n)
                ],
                dtype=np.int8,
            )
            quote_balanced_flags = np.array(
                [self._is_quote_balanced(text, i) for i in range(n)], dtype=np.int8
            )
            word_complete_flags = np.array(
                [self._is_word_likely_complete(text, i) for i in range(n)],
                dtype=np.int8,
            )
            followed_by_lowercase_flags = np.array(
                [1 if (i < n - 1 and text[i + 1].islower()) else 0 for i in range(n)],
                dtype=np.int8,
            )

            # List-related features - focus optimization on these expensive ones
            in_list_item_flags = np.zeros(n, dtype=np.int8)
            semicolon_in_list_flags = np.zeros(n, dtype=np.int8)
            near_colon_flags = np.zeros(n, dtype=np.int8)

            # Only compute expensive features for potential list positions
            for i in range(n):
                char = text[i] if i < n else ""
                # Early filtering for list-related computation
                if (
                    char in ".,;:()[]0123456789abcdefghijklmnopqrstuvwxyz•·○●■□▪▫"
                    or (i > 0 and text[i - 1] in ".,;:()")
                    or (i < n - 1 and text[i + 1] in ".,;:()")
                ):
                    in_list_item_flags[i] = self._is_in_list_item(text, i)
                    if char == ";":
                        semicolon_in_list_flags[i] = self._is_semicolon_in_list(text, i)
                    if (
                        char == ":"
                        or (i > 0 and text[i - 1] == ":")
                        or (i < n - 1 and text[i + 1] == ":")
                    ):
                        near_colon_flags[i] = self._is_near_colon(text, i)

        # Calculate terminator feature (-1 for secondary, 0 for non-terminator, 1 for primary)
        # Consistent definition for all paths
        terminator_feature = primary_term_flags - secondary_term_flags

        # Initialize features matrix
        num_positions = len(position_indices)
        features = np.full(
            (num_positions, feature_size), -3, dtype=np.int16
        )  # Fill with whitespace placeholder

        # Process each position
        for i, pos in enumerate(position_indices):
            # Calculate valid window range
            start_idx = max(0, pos - left_window)
            end_idx = min(n, pos + right_window + 1)

            # Calculate corresponding indices in the feature vector
            feat_start = left_window - (pos - start_idx)
            feat_end = feat_start + (end_idx - start_idx)

            # Copy encoded characters to the feature vector
            features[i, feat_start:feat_end] = encoded_chars[start_idx:end_idx]

            # Add additional features
            if positions is not None and len(position_indices) < n / 4:
                # For selective positions optimization path
                idx = i  # For selective positions, the array index matches feature array index
                features[i, window_size] = abbr_flags[idx]  # Abbreviation flag
                features[i, window_size + 1] = terminator_feature[
                    idx
                ]  # Terminator type
                features[i, window_size + 2] = quote_balanced_flags[
                    idx
                ]  # Quote balance
                features[i, window_size + 3] = word_complete_flags[
                    idx
                ]  # Word completion
                features[i, window_size + 4] = followed_by_lowercase_flags[
                    idx
                ]  # Followed by lowercase
                features[i, window_size + 5] = in_list_item_flags[
                    idx
                ]  # Is in list item
                features[i, window_size + 6] = semicolon_in_list_flags[
                    idx
                ]  # Is semicolon in list
                features[i, window_size + 7] = near_colon_flags[idx]  # Is near colon
            else:
                # For full text processing path
                features[i, window_size] = abbr_flags[pos]  # Abbreviation flag
                features[i, window_size + 1] = terminator_feature[
                    pos
                ]  # Terminator type
                features[i, window_size + 2] = quote_balanced_flags[
                    pos
                ]  # Quote balance
                features[i, window_size + 3] = word_complete_flags[
                    pos
                ]  # Word completion
                features[i, window_size + 4] = followed_by_lowercase_flags[
                    pos
                ]  # Followed by lowercase
                features[i, window_size + 5] = in_list_item_flags[
                    pos
                ]  # Is in list item
                features[i, window_size + 6] = semicolon_in_list_flags[
                    pos
                ]  # Is semicolon in list
                features[i, window_size + 7] = near_colon_flags[pos]  # Is near colon

        # Convert back to list for API compatibility
        return features.tolist()

    def _process_chunk(
        self,
        chunk_range: Tuple[int, int],
        text: str,
        left_window: int,
        right_window: int,
    ) -> FeatureMatrix:
        """Process a chunk of text for parallel feature extraction."""
        start, end = chunk_range
        positions = list(range(start, end))
        return self.get_char_features(text, left_window, right_window, positions)

    def mark_annotation_positions(self, text: str) -> PositionLabels:
        """
        Mark positions of sentence and paragraph annotations in text.

        Args:
            text (str): Text with sentence and paragraph annotations

        Returns:
            PositionLabels: Position markers (0: non-terminal, 1: boundary)
        """
        result = []
        i = 0
        text_len = len(text)

        # Get tag lengths
        sentence_len = len(SENTENCE_TAG)
        paragraph_len = len(PARAGRAPH_TAG)

        while i < text_len:
            # Check for sentence tag
            if (
                i + sentence_len <= text_len
                and text[i : i + sentence_len] == SENTENCE_TAG
            ):
                if result:  # Mark the previous character
                    result[-1] = 1  # Binary classification: 1 for any boundary
                i += sentence_len
            # Check for paragraph tag
            elif (
                i + paragraph_len <= text_len
                and text[i : i + paragraph_len] == PARAGRAPH_TAG
            ):
                if result:  # Mark the previous character
                    result[-1] = 1  # Binary classification: 1 for any boundary
                i += paragraph_len
            else:
                # Regular character
                result.append(0)
                i += 1

        return result

    def process_annotated_text(
        self,
        text: str,
        left_window: int = 5,
        right_window: int = 5,
        num_workers: int = 0,
    ) -> Tuple[str, FeatureMatrix, PositionLabels]:
        """
        Process an annotated text to extract features and labels with optional parallelization.

        Args:
            text (str): Text with sentence and paragraph annotations
            left_window (int, optional): Size of left context window. Defaults to 5.
            right_window (int, optional): Size of right context window. Defaults to 5.
            num_workers (int, optional): Number of worker processes for parallel feature extraction.
                If 0, automatic detection based on text length. Defaults to 0.

        Returns:
            Tuple[str, FeatureMatrix, PositionLabels]:
                - clean_text: Text with annotations removed
                - features: Character-level features
                - labels: Position markers (0: non-terminal, 1: boundary)
        """
        # Get labels first
        labels = self.mark_annotation_positions(text)

        # Clean text (remove annotations)
        clean_text = text.replace(SENTENCE_TAG, "").replace(PARAGRAPH_TAG, "")

        # Determine if we should use parallel processing
        # Only parallelize for long texts, short texts are faster with direct processing
        text_len = len(clean_text)

        # Automatic worker detection if num_workers is 0
        if num_workers == 0:
            if text_len > 100000:  # Very long text
                num_workers = min(multiprocessing.cpu_count(), 8)
            elif text_len > 10000:  # Medium length text
                num_workers = min(multiprocessing.cpu_count() // 2, 4)
            else:  # Short text, no parallelization
                num_workers = 0

        # Extract features, potentially in parallel
        if num_workers > 1 and text_len > 10000:
            # Split the text into chunks for parallel processing
            chunk_size = text_len // num_workers
            chunks = [
                (i, min(i + chunk_size, text_len))
                for i in range(0, text_len, chunk_size)
            ]

            # Create a partial function with fixed parameters
            process_chunk = partial(
                self._process_chunk,
                text=clean_text,
                left_window=left_window,
                right_window=right_window,
            )

            # Process chunks in parallel
            with Pool(num_workers) as pool:
                feature_chunks = pool.map(process_chunk, chunks)

            # Combine chunks
            features = [feat for chunk in feature_chunks for feat in chunk]
        else:
            # For shorter texts, use the regular approach
            features = self.get_char_features(clean_text, left_window, right_window)

        return clean_text, features, labels

    def print_text_analysis(
        self, text: str, left_window: int = 5, right_window: int = 5
    ) -> None:
        """
        Analyze and print a formatted report of character features and labels.

        Args:
            text (str): Text with sentence and paragraph annotations
            left_window (int, optional): Size of left context window. Defaults to 5.
            right_window (int, optional): Size of right context window. Defaults to 5.
        """
        clean_text, features, labels = self.process_annotated_text(
            text, left_window, right_window
        )

        print("Character | Features | Label")
        print("-" * 50)

        for idx, (char, feature_vector, label) in enumerate(
            zip(clean_text, features, labels)
        ):
            feature_str = " ".join(f"{x:2}" for x in feature_vector)
            print(f"{char:10} | {feature_str:40} | {label}")

        # Summary
        print(f"\nTotal characters: {len(clean_text)}")
        print(f"Boundary positions: {labels.count(1)}")
