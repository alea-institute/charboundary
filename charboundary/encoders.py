"""
Character encoding functionality for the charboundary library.
"""

from typing import Dict, Protocol

from charboundary.constants import (
    PUNCTUATION_CHAR_LIST,
    WS_CHAR_LIST,
    TERMINAL_SENTENCE_CHAR_LIST,
    TERMINAL_PARAGRAPH_CHAR_LIST,
)


class CharacterEncoderProtocol(Protocol):
    """Protocol defining the interface for character encoders."""

    def encode(self, char: str) -> int:
        """Encode a character into a numerical representation."""
        ...

    def is_terminal_sentence_char(self, char: str) -> bool:
        """Check if a character can possibly end a sentence."""
        ...

    def is_terminal_paragraph_char(self, char: str) -> bool:
        """Check if a character can indicate the end of a paragraph."""
        ...


class CharacterEncoder:
    """
    Encodes characters into numerical representations based on their type.

    This is the default implementation that categorizes characters into:
    - Alphabetic: values > 0 (1-26 for a-z)
    - Numeric: value 0
    - Terminal sentence: value -1
    - Terminal paragraph: value -2
    - Whitespace: value -3
    - Punctuation: value -4
    - Other: value -5
    """

    def __init__(self):
        """Initialize the CharacterEncoder with an empty cache."""
        self.cache: Dict[str, int] = {}

    def encode(self, char: str) -> int:
        """
        Encode a character into a numerical representation based on its type.
        Uses caching for better performance.

        Args:
            char (str): The character to encode

        Returns:
            int: Numerical representation of the character
        """
        # Check if we've already encoded this character
        if char in self.cache:
            return self.cache[char]

        # Encode based on character type
        if char.isalpha():
            value = ord(char.lower()) - ord("a") + 1
        elif char.isdigit():
            value = 0
        elif char in TERMINAL_SENTENCE_CHAR_LIST:
            value = -1
        elif char in TERMINAL_PARAGRAPH_CHAR_LIST:
            value = -2
        elif char in WS_CHAR_LIST:
            value = -3
        elif char in PUNCTUATION_CHAR_LIST:
            value = -4
        else:
            value = -5

        # Cache the result for future use
        self.cache[char] = value
        return value

    def is_terminal_sentence_char(self, char: str) -> bool:
        """
        Check if a character can possibly end a sentence.

        Args:
            char (str): The character to check

        Returns:
            bool: True if the character can end a sentence, False otherwise
        """
        return char in TERMINAL_SENTENCE_CHAR_LIST

    def is_terminal_paragraph_char(self, char: str) -> bool:
        """
        Check if a character can indicate the end of a paragraph.

        Args:
            char (str): The character to check

        Returns:
            bool: True if the character can end a paragraph, False otherwise
        """
        return char in TERMINAL_PARAGRAPH_CHAR_LIST


class OneHotCharacterEncoder(CharacterEncoder):
    """
    A character encoder that uses one-hot encoding.

    This provides an alternative encoding strategy where each unique character
    gets its own distinct integer value, rather than being categorized.
    """

    def encode(self, char: str) -> int:
        """
        Encode a character using one-hot style encoding.

        Args:
            char (str): The character to encode

        Returns:
            int: Unique integer for this character
        """
        # Check cache first
        if char in self.cache:
            return self.cache[char]

        # Use unicode code point as the value
        value = ord(char)

        # Cache and return
        self.cache[char] = value
        return value
