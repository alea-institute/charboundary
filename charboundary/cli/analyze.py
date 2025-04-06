"""
Analysis command for the charboundary CLI.
"""

import json
import os

from charboundary.segmenters import TextSegmenter


def add_analyze_args(subparsers) -> None:
    """
    Add analysis command arguments to the subparsers.

    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser("analyze", help="Analyze text using a trained model")

    # Required arguments
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--input", required=True, help="Path to the input text file")

    # Optional arguments
    parser.add_argument("--output", help="Path to save the segmented text")
    parser.add_argument(
        "--format",
        choices=["annotated", "sentences", "paragraphs"],
        default="annotated",
        help="Output format (default: annotated)",
    )
    parser.add_argument("--metrics", help="Path to save metrics as JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Probability threshold for classification (0.0-1.0). Values below 0.5 favor recall, values above 0.5 favor precision.",
    )


def handle_analyze(args) -> int:
    """
    Handle the analyze command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    print(f"Loading model from {args.model}")
    try:
        segmenter = TextSegmenter.load(args.model, trust_model=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print(f"Reading text from {args.input}")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1

    # Process based on the requested format
    if args.format == "annotated":
        result = segmenter.segment_text(text, threshold=args.threshold)
    elif args.format == "sentences":
        sentences = segmenter.segment_to_sentences(text, threshold=args.threshold)
        result = "\n".join(sentences)
    elif args.format == "paragraphs":
        paragraphs = segmenter.segment_to_paragraphs(text, threshold=args.threshold)
        result = "\n\n".join(paragraphs)

    # Save to output file if specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Output saved to {args.output}")
    else:
        print("\nSegmentation Result:")
        print("=" * 40)
        print(result)
        print("=" * 40)

    # Calculate and save metrics if requested
    if args.metrics:
        if args.format == "sentences":
            metrics = {
                "num_sentences": len(segmenter.segment_to_sentences(text)),
                "avg_sentence_length": sum(
                    len(s) for s in segmenter.segment_to_sentences(text)
                )
                / len(segmenter.segment_to_sentences(text))
                if segmenter.segment_to_sentences(text)
                else 0,
            }
        elif args.format == "paragraphs":
            metrics = {
                "num_paragraphs": len(segmenter.segment_to_paragraphs(text)),
                "avg_paragraph_length": sum(
                    len(p) for p in segmenter.segment_to_paragraphs(text)
                )
                / len(segmenter.segment_to_paragraphs(text))
                if segmenter.segment_to_paragraphs(text)
                else 0,
            }
        else:
            metrics = {
                "num_sentences": len(segmenter.segment_to_sentences(text)),
                "num_paragraphs": len(segmenter.segment_to_paragraphs(text)),
            }

        os.makedirs(os.path.dirname(os.path.abspath(args.metrics)), exist_ok=True)
        with open(args.metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics}")

    return 0
