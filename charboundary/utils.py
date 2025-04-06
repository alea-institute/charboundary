"""
Utility functions for the charboundary library.
"""

import gzip
import json
from typing import List, Dict, Any, Optional

from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG


def load_jsonl(
    path: str,
    key: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Load texts from a gzipped JSONL file.

    Args:
        path (str): Path to the gzipped JSONL file
        key (str, optional): Key for accessing the text in each record. Defaults to "text".
        max_samples (int, optional): Maximum number of samples to load.
            If None, load all samples.

    Returns:
        List[str]: Texts from the file
    """
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            record = json.loads(line)
            text = record.get(key, "")
            texts.append(text)
    return texts


def save_jsonl(
    data: List[Dict[str, Any]],
    path: str,
    compress: bool = True,
) -> None:
    """
    Save data to a JSONL file.

    Args:
        data (List[Dict[str, Any]]): Data to save
        path (str): Path to save the data
        compress (bool, optional): Whether to compress the output file. Defaults to True.
    """
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")


def annotate_text(sentences: List[str]) -> str:
    """
    Create an annotated text from a list of sentences.

    Args:
        sentences (List[str]): List of sentences

    Returns:
        str: Annotated text with sentence tags and paragraph tags between each sentence
    """
    result = []

    for sentence in sentences:
        result.append(sentence)
        result.append(SENTENCE_TAG)

    # Add paragraph tag at the end
    if result:
        result.append(PARAGRAPH_TAG)

    return "".join(result)


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read text from a file.

    Args:
        path (str): Path to the file
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        str: Text content of the file
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(text: str, path: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file.

    Args:
        text (str): Text to write
        path (str): Path to the file
        encoding (str, optional): File encoding. Defaults to "utf-8".
    """
    with open(path, "w", encoding=encoding) as f:
        f.write(text)
