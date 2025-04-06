"""
Best model selection command for the charboundary CLI.
"""

import json
import os

from charboundary.segmenters import TextSegmenter


def add_best_model_args(subparsers) -> None:
    """
    Add best model selection command arguments to the subparsers.

    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser(
        "best-model", help="Train the best model based on evaluation data"
    )

    # Required arguments
    parser.add_argument("--data", required=True, help="Path to the training data file")
    parser.add_argument(
        "--output", required=True, help="Path to save the best trained model"
    )

    # Optional arguments
    parser.add_argument(
        "--validation", help="Path to validation data for model selection"
    )
    parser.add_argument("--metrics-file", help="Path to save metrics as JSON")

    # Model parameter ranges for grid search
    parser.add_argument(
        "--left-window-values",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Left window sizes to try (default: 3, 5, 7)",
    )
    parser.add_argument(
        "--right-window-values",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Right window sizes to try (default: 3, 5, 7)",
    )
    parser.add_argument(
        "--n-estimators-values",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Number of estimators to try (default: 50, 100, 200)",
    )
    parser.add_argument(
        "--max-depth-values",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="Max depth values to try (default: 8, 16, 24)",
    )
    parser.add_argument(
        "--threshold-values",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Probability threshold values to try (default: 0.3, 0.5, 0.7)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="Sample rate for non-terminal characters (default: 0.1)",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to use for training"
    )


def handle_best_model(args) -> int:
    """
    Handle the best-model command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    print("Starting best model search")
    print(f"Training data: {args.data}")

    # Parameters to try
    left_windows = args.left_window_values
    right_windows = args.right_window_values
    n_estimators_values = args.n_estimators_values
    max_depth_values = args.max_depth_values
    threshold_values = args.threshold_values

    print(f"Left window values: {left_windows}")
    print(f"Right window values: {right_windows}")
    print(f"Number of estimators values: {n_estimators_values}")
    print(f"Max depth values: {max_depth_values}")
    print(f"Threshold values: {threshold_values}")

    best_metrics = None
    best_params = None
    best_segmenter = None

    # Try all parameter combinations
    total_combinations = (
        len(left_windows)
        * len(right_windows)
        * len(n_estimators_values)
        * len(max_depth_values)
        * len(threshold_values)
    )
    print(f"Total parameter combinations: {total_combinations}")

    current_combination = 0

    for left_window in left_windows:
        for right_window in right_windows:
            for n_estimators in n_estimators_values:
                for max_depth in max_depth_values:
                    for threshold in threshold_values:
                        current_combination += 1

                        print(
                            f"\nCombination {current_combination}/{total_combinations}:"
                        )
                        print(f"  Left window: {left_window}")
                        print(f"  Right window: {right_window}")
                        print(f"  N estimators: {n_estimators}")
                        print(f"  Max depth: {max_depth}")
                        print(f"  Threshold: {threshold}")

                        # Create and train segmenter
                        segmenter = TextSegmenter()

                        model_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                        }

                        try:
                            metrics = segmenter.train(
                                data=args.data,
                                model_params=model_params,
                                sample_rate=args.sample_rate,
                                max_samples=args.max_samples,
                                left_window=left_window,
                                right_window=right_window,
                                threshold=threshold,
                            )

                            # Evaluate on validation data if provided
                            if args.validation:
                                metrics = segmenter.evaluate(
                                    data=args.validation,
                                    max_samples=args.max_samples,
                                )

                            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                            print(f"  F1-score: {metrics.get('f1_score', 0):.4f}")

                            # Track best model
                            if best_metrics is None or metrics.get(
                                "f1_score", 0
                            ) > best_metrics.get("f1_score", 0):
                                best_metrics = metrics
                                best_params = {
                                    "left_window": left_window,
                                    "right_window": right_window,
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth,
                                    "threshold": threshold,
                                }
                                best_segmenter = segmenter

                                print("  New best model!")

                        except Exception as e:
                            print(f"  Error training model: {e}")

    # Save the best model
    if best_segmenter:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        best_segmenter.save(args.output)
        print(f"\nBest model saved to {args.output}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best accuracy: {best_metrics.get('accuracy', 0):.4f}")
        print(f"Best F1-score: {best_metrics.get('f1_score', 0):.4f}")

        # Save metrics if requested
        if args.metrics_file:
            result = {
                "best_params": best_params,
                "best_metrics": best_metrics,
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(args.metrics_file)), exist_ok=True
            )
            with open(args.metrics_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Best model metrics saved to {args.metrics_file}")

        return 0
    else:
        print("Error: No valid model found")
        return 1
