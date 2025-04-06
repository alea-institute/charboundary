"""
Main CLI entry point for the charboundary library.
"""

import argparse
import sys
from typing import List, Optional

from charboundary.cli.train import add_train_args, handle_train
from charboundary.cli.analyze import add_analyze_args, handle_analyze
from charboundary.cli.best_model import add_best_model_args, handle_best_model


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CharBoundary: A library for segmenting text into sentences."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add subcommands
    add_train_args(subparsers)
    add_analyze_args(subparsers)
    add_best_model_args(subparsers)

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:] if None)

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    # Dispatch to the appropriate handler
    if parsed_args.command == "train":
        return handle_train(parsed_args)
    elif parsed_args.command == "analyze":
        return handle_analyze(parsed_args)
    elif parsed_args.command == "best-model":
        return handle_best_model(parsed_args)
    else:
        print(f"Error: Unknown command {parsed_args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
