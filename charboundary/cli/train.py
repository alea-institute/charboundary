"""
Training command for the charboundary CLI.
"""

import json
import os
import time

from charboundary.segmenters import TextSegmenter


def add_train_args(subparsers) -> None:
    """
    Add training command arguments to the subparsers.

    Args:
        subparsers: Subparsers object from argparse
    """
    parser = subparsers.add_parser("train", help="Train a text segmentation model")

    # Required arguments
    parser.add_argument("--data", required=True, help="Path to the training data file")
    parser.add_argument(
        "--output", required=True, help="Path to save the trained model"
    )

    # Optional arguments
    parser.add_argument(
        "--left-window",
        type=int,
        default=5,
        help="Size of left context window (default: 5)",
    )
    parser.add_argument(
        "--right-window",
        type=int,
        default=5,
        help="Size of right context window (default: 5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=16, help="Maximum depth of trees (default: 16)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="Sampling rate for non-terminal characters (default: 0.1)",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to use for training"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classification (0.0-1.0). Values below 0.5 favor recall, values above 0.5 favor precision. (default: 0.5)",
    )
    parser.add_argument(
        "--metrics-file", help="Optional path to save training metrics as JSON"
    )

    # Feature selection arguments
    feature_selection_group = parser.add_argument_group("Feature Selection")
    feature_selection_group.add_argument(
        "--use-feature-selection",
        action="store_true",
        help="Use feature selection to identify and use only important features",
    )
    feature_selection_group.add_argument(
        "--feature-selection-threshold",
        type=float,
        default=0.01,
        help="Importance threshold for feature selection (default: 0.01)",
    )
    feature_selection_group.add_argument(
        "--max-features", type=int, help="Maximum number of features to select"
    )


def handle_train(args) -> int:
    """
    Handle the train command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    print(f"Training model with data from {args.data}")
    print(f"Window size: {args.left_window} (left), {args.right_window} (right)")
    print(
        f"Model parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}"
    )

    # Create segmenter
    segmenter = TextSegmenter()

    # Start timing
    start_time = time.time()

    # Train the model
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }

    # Print feature selection info if enabled
    if args.use_feature_selection:
        print(
            f"Using feature selection with threshold {args.feature_selection_threshold}"
        )
        if args.max_features:
            print(f"Maximum features: {args.max_features}")

    metrics = segmenter.train(
        data=args.data,
        model_params=model_params,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        left_window=args.left_window,
        right_window=args.right_window,
        threshold=args.threshold,
        use_feature_selection=args.use_feature_selection,
        feature_selection_threshold=args.feature_selection_threshold,
        max_features=args.max_features,
    )

    # Calculate training time
    training_time = time.time() - start_time
    metrics["training_time_seconds"] = training_time

    # Save the model
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    segmenter.save(args.output)
    print(f"Model saved to {args.output}")

    # Print metrics
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Overall accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"Boundary accuracy:    {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"Boundary precision:   {metrics.get('precision', 0):.4f}")
    print(f"Boundary recall:      {metrics.get('recall', 0):.4f}")
    print(f"Boundary F1-score:    {metrics.get('f1_score', 0):.4f}")

    # Save metrics to file if requested
    if args.metrics_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_file)), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_file}")

    return 0
