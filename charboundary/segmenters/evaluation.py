"""
Evaluation functionality for text segmenters.
"""

import gzip
import json
from typing import List, Union, Optional, TYPE_CHECKING

import sklearn.metrics

from charboundary.features import PositionLabels
from charboundary.segmenters.types import MetricsResult

if TYPE_CHECKING:
    from charboundary.segmenters.base import TextSegmenter


class Evaluator:
    """
    Handles evaluation of text segmenters.
    """

    @staticmethod
    def evaluate(
        segmenter: "TextSegmenter",
        data: Union[str, List[str]],
        max_samples: Optional[int] = None,
    ) -> MetricsResult:
        """
        Evaluate the model on a dataset.

        Args:
            segmenter: The TextSegmenter to evaluate
            data (Union[str, List[str]]):
                - Path to a test data file
                - List of annotated texts
            max_samples (int, optional): Maximum number of samples to process.
                If None, process all samples.

        Returns:
            MetricsResult: Evaluation metrics
        """
        if not segmenter.is_trained:
            raise ValueError("Model has not been trained yet.")

        all_true_labels = []
        all_predictions = []

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
                                Evaluator._evaluate_text(
                                    segmenter,
                                    json_obj["text"],
                                    all_true_labels,
                                    all_predictions,
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
                        Evaluator._evaluate_text(
                            segmenter, text, all_true_labels, all_predictions
                        )
        elif isinstance(data, list):
            for i, text in enumerate(data):
                if max_samples is not None and i >= max_samples:
                    break
                Evaluator._evaluate_text(
                    segmenter, text, all_true_labels, all_predictions
                )

        # Generate evaluation report
        report: MetricsResult = {
            "accuracy": 0.0,
            "boundary_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "binary_mode": True,
        }

        if all_true_labels and all_predictions:
            # Calculate metrics
            accuracy = sklearn.metrics.accuracy_score(all_true_labels, all_predictions)

            # Classification report
            class_report = sklearn.metrics.classification_report(
                y_true=all_true_labels,
                y_pred=all_predictions,
                output_dict=True,
                zero_division=0,
            )

            # Extract metrics for the boundary class (1)
            if "1" in class_report:
                precision = class_report["1"]["precision"]
                recall = class_report["1"]["recall"]
                f1_score = class_report["1"]["f1-score"]
            else:
                precision = 0.0
                recall = 0.0
                f1_score = 0.0

            # Update report
            report["accuracy"] = accuracy
            report["precision"] = precision
            report["recall"] = recall
            report["f1_score"] = f1_score

            # Also calculate boundary-specific metrics (on positions where either true or predicted is a boundary)
            boundary_indices = [
                i
                for i, (t, p) in enumerate(zip(all_true_labels, all_predictions))
                if t == 1 or p == 1
            ]

            if boundary_indices:
                boundary_true = [all_true_labels[i] for i in boundary_indices]
                boundary_pred = [all_predictions[i] for i in boundary_indices]
                boundary_accuracy = sklearn.metrics.accuracy_score(
                    boundary_true, boundary_pred
                )
                report["boundary_accuracy"] = boundary_accuracy

        return report

    @staticmethod
    def _evaluate_text(
        segmenter: "TextSegmenter",
        text: str,
        all_true_labels: PositionLabels,
        all_predictions: PositionLabels,
    ) -> None:
        """
        Evaluate a text and add its true labels and predictions to the provided lists.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Annotated text
            all_true_labels (PositionLabels]): List to which true labels will be added
            all_predictions (PositionLabels]): List to which predictions will be added
        """
        clean_text, features, true_labels = (
            segmenter.feature_extractor.process_annotated_text(
                text,
                segmenter.config.left_window,
                segmenter.config.right_window,
                segmenter.config.num_workers,
            )
        )

        predictions = segmenter.model.predict(features)

        # Add all predictions and labels for proper evaluation
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
