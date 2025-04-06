"""
Model definitions and interfaces for the charboundary library.
"""

from typing import List, Dict, Any, Protocol, Optional, Union
from pathlib import Path
import sklearn.ensemble
import sklearn.metrics

# Try to import ONNX support (optional dependency)
try:
    from charboundary.onnx_support import (
        convert_to_onnx,
        save_onnx_model,
        load_onnx_model,
        create_onnx_inference_session,
        onnx_predict,
        onnx_predict_proba,
    )

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TextSegmentationModel(Protocol):
    """Protocol defining the interface for text segmentation models."""

    def fit(self, X: List[List[int]], y: List[int]) -> None:
        """Fit the model to the data."""
        ...

    def predict(self, X: List[List[int]]) -> List[int]:
        """Predict segmentation labels for the given features."""
        ...

    def get_metrics(self, X: List[List[int]], y: List[int]) -> Dict[str, Any]:
        """Evaluate the model on the given data."""
        ...

    @property
    def is_binary(self) -> bool:
        """Whether the model uses binary classification (boundary/non-boundary)."""
        ...


class BinaryRandomForestModel:
    """
    A text segmentation model based on RandomForest for binary classification.
    Only distinguishes between boundary (1) and non-boundary (0) positions.

    This model supports conversion to ONNX format when the 'onnx' optional
    dependency is installed. ONNX models can be used for faster inference,
    especially when deployed in production environments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_onnx: bool = False,
        onnx_optimization_level: int = 1,
        **kwargs,
    ):
        """
        Initialize the BinaryRandomForestModel.

        Args:
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        Defaults to 0.5.
            use_onnx (bool, optional): Whether to use ONNX for inference if available.
                                      Defaults to False.
            onnx_optimization_level (int, optional): ONNX optimization level (0-3).
                                                    0: No optimization
                                                    1: Basic optimizations (default)
                                                    2: Extended optimizations
                                                    3: All optimizations including extended memory reuse
                                                    Defaults to 1.
            **kwargs: Parameters to pass to the underlying RandomForestClassifier
        """
        self.threshold = threshold
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.onnx_optimization_level = onnx_optimization_level
        self.onnx_model = None
        self.onnx_session = None
        self.model_params = (
            kwargs.copy()
            if kwargs
            else {
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "n_jobs": -1,
            }
        )

        # Set class weight to 'balanced' to handle imbalanced data
        if "class_weight" not in self.model_params:
            self.model_params["class_weight"] = "balanced"

        self.model = sklearn.ensemble.RandomForestClassifier(**self.model_params)

        # ONNX related attributes
        self.onnx_model = None
        self.onnx_session = None
        self.feature_count = None

    @property
    def is_binary(self) -> bool:
        """
        Whether the model uses binary classification (boundary/non-boundary).

        Returns:
            bool: Always True for this model
        """
        return True

    def fit(self, X: List[List[int]], y: List[int]) -> None:
        """
        Fit the model to the data.

        Args:
            X (List[List[int]]): Feature vectors
            y (List[int]): Target labels (0 for non-boundary, 1 for boundary)
        """
        # Ensure binary labels
        y_binary = [1 if label > 0 else 0 for label in y]

        # Store feature count for ONNX conversion
        if X and len(X) > 0:
            self.feature_count = len(X[0])

        self.model.fit(X=X, y=y_binary)

        # Convert to ONNX if requested and available
        if self.use_onnx and ONNX_AVAILABLE:
            self.to_onnx()

    def predict(
        self, X: List[List[int]], threshold: Optional[float] = None
    ) -> List[int]:
        """
        Predict segmentation labels for the given features.

        Args:
            X (List[List[int]]): Feature vectors
            threshold (float, optional): Custom probability threshold to use for this prediction.
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[int]: Predicted labels (0 for non-boundary, 1 for boundary)
        """
        # Use custom threshold if provided, otherwise use the model's default
        thresh = threshold if threshold is not None else self.threshold

        # Use ONNX inference if enabled and available
        if self.use_onnx and self.onnx_session is not None:
            return onnx_predict(self.onnx_session, X, threshold=thresh)

        # Otherwise use scikit-learn inference
        if thresh == 0.5:
            # Use the default scikit-learn prediction for the default threshold
            return self.model.predict(X)
        else:
            # Get class probabilities and apply custom threshold
            probas = self.model.predict_proba(X)
            # Class 1 (boundary) is typically the second column
            return [1 if proba[1] >= thresh else 0 for proba in probas]

    def predict_proba(self, X: List[List[int]]) -> List[List[float]]:
        """
        Predict class probabilities for the given features.

        Args:
            X (List[List[int]]): Feature vectors

        Returns:
            List[List[float]]: Predicted probabilities for each class
        """
        # Use ONNX inference if enabled and available
        if self.use_onnx and self.onnx_session is not None:
            return onnx_predict_proba(self.onnx_session, X)

        # Otherwise use scikit-learn inference
        return self.model.predict_proba(X).tolist()

    def get_metrics(self, X: List[List[int]], y: List[int]) -> Dict[str, Any]:
        """
        Evaluate the model on the given data.

        Args:
            X (List[List[int]]): Feature vectors
            y (List[int]): True labels

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Convert labels to binary
        y_binary = [1 if label > 0 else 0 for label in y]

        predictions = self.predict(X)

        # Default report structure
        report = {
            "accuracy": sklearn.metrics.accuracy_score(y_binary, predictions),
            "binary_mode": True,
        }

        try:
            # Calculate metrics specific to the boundary class (label=1)
            boundary_precision = sklearn.metrics.precision_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )
            boundary_recall = sklearn.metrics.recall_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )
            boundary_f1 = sklearn.metrics.f1_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )

            # Update report with boundary metrics
            report["precision"] = boundary_precision
            report["recall"] = boundary_recall
            report["f1_score"] = boundary_f1

            # Calculate boundary-specific accuracy
            boundary_indices = [
                i
                for i, (t, p) in enumerate(zip(y_binary, predictions))
                if t == 1 or p == 1
            ]

            if boundary_indices:
                boundary_true = [y_binary[i] for i in boundary_indices]
                boundary_pred = [predictions[i] for i in boundary_indices]
                boundary_accuracy = sklearn.metrics.accuracy_score(
                    boundary_true, boundary_pred
                )
                report["boundary_accuracy"] = boundary_accuracy
            else:
                report["boundary_accuracy"] = 0.0

            # Create full classification report
            full_report = sklearn.metrics.classification_report(
                y_true=y_binary,
                y_pred=predictions,
                target_names=["Non-boundary", "Boundary"],
                labels=[0, 1],
                zero_division=0,
                output_dict=True,
            )

            # Add class-specific metrics
            for k, v in full_report.items():
                if k not in [
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                    "Non-boundary",
                    "Boundary",
                ]:
                    report[f"class_{k}"] = v

        except Exception as e:
            print(f"Warning: Error generating metrics: {e}")

        return report

    def get_feature_importances(self) -> List[float]:
        """
        Get feature importances from the model.

        Returns:
            List[float]: Feature importance scores
        """
        return self.model.feature_importances_.tolist()

    def to_onnx(self) -> Optional[bytes]:
        """
        Convert the model to ONNX format.

        Returns:
            Optional[bytes]: Serialized ONNX model if conversion was successful, None otherwise

        Raises:
            ImportError: If ONNX dependencies are not installed
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX conversion requires 'onnx' and 'skl2onnx' packages. "
                "Install them with: pip install charboundary[onnx]"
            )

        if self.feature_count is None:
            raise ValueError("Model must be fitted before conversion to ONNX")

        try:
            # Convert model to ONNX
            self.onnx_model = convert_to_onnx(
                model=self.model,
                feature_count=self.feature_count,
                model_name="charboundary_model",
            )

            # Create inference session with optimization level
            self.onnx_session = create_onnx_inference_session(
                self.onnx_model, optimization_level=self.onnx_optimization_level
            )

            return self.onnx_model
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to convert model to ONNX: {str(e)}")
            self.use_onnx = False
            return None

    def save_onnx(self, file_path: Union[str, Path], compress: bool = True) -> bool:
        """
        Save the ONNX model to a file, optionally with XZ compression.

        Args:
            file_path: Path to save the model
            compress: Whether to compress the model with XZ (default: True)

        Returns:
            bool: True if the model was saved successfully, False otherwise

        Raises:
            ImportError: If ONNX is not installed
            ValueError: If the model has not been converted to ONNX
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX support requires 'onnx' package. "
                "Install it with: pip install charboundary[onnx]"
            )

        if self.onnx_model is None:
            # Try to convert first
            if self.to_onnx() is None:
                raise ValueError(
                    "Model has not been converted to ONNX and conversion failed"
                )

        try:
            save_onnx_model(self.onnx_model, file_path, compress=compress)
            return True
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to save ONNX model: {str(e)}")
            return False

    def load_onnx(self, file_path: Union[str, Path]) -> bool:
        """
        Load an ONNX model from a file.

        Args:
            file_path: Path to the ONNX model file

        Returns:
            bool: True if the model was loaded successfully, False otherwise

        Raises:
            ImportError: If ONNX is not installed
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX support requires 'onnx' package. "
                "Install it with: pip install charboundary[onnx]"
            )

        try:
            self.onnx_model = load_onnx_model(file_path)

            # Handle case where model doesn't have onnx_optimization_level attribute
            # Default to level 1 if not present
            optimization_level = getattr(self, "onnx_optimization_level", 1)

            self.onnx_session = create_onnx_inference_session(
                self.onnx_model, optimization_level=optimization_level
            )
            self.use_onnx = True
            return True
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to load ONNX model: {str(e)}")
            self.use_onnx = False
            return False

    def enable_onnx(
        self, enable: bool = True, optimization_level: Optional[int] = None
    ) -> bool:
        """
        Enable or disable ONNX inference.

        Args:
            enable: Whether to enable ONNX inference
            optimization_level: ONNX optimization level (0-3) to use.
                              If provided, update the model's optimization level
                              and recreate the inference session.

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if enable and not ONNX_AVAILABLE:
            import warnings

            warnings.warn(
                "Cannot enable ONNX: ONNX support requires 'onnx' and 'skl2onnx' packages. "
                "Install them with: pip install charboundary[onnx]"
            )
            return False

        # Initialize onnx_optimization_level if it doesn't exist
        if not hasattr(self, "onnx_optimization_level"):
            self.onnx_optimization_level = 1

        # Update optimization level if provided
        if optimization_level is not None:
            if optimization_level not in [0, 1, 2, 3]:
                import warnings

                warnings.warn(
                    f"Invalid optimization level: {optimization_level}. Using current level: {self.onnx_optimization_level}"
                )
            else:
                self.onnx_optimization_level = optimization_level
                # Recreate the session with the new optimization level if we have a model
                if self.onnx_model is not None:
                    try:
                        self.onnx_session = create_onnx_inference_session(
                            self.onnx_model,
                            optimization_level=self.onnx_optimization_level,
                        )
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"Failed to create ONNX session with optimization level {self.onnx_optimization_level}: {str(e)}"
                        )
                        return False

        if enable and self.onnx_session is None:
            # Try to convert model to ONNX if not already done
            if self.onnx_model is None:
                if self.to_onnx() is None:
                    return False

        self.use_onnx = enable and ONNX_AVAILABLE and self.onnx_session is not None
        return True


# Factory function for creating models
class FeatureSelectedRandomForestModel(BinaryRandomForestModel):
    """
    A RandomForest text segmentation model with integrated feature selection.

    This model first trains a RandomForest, selects the most important features,
    and then retrains using only those selected features.

    The feature selection process works as follows:
    1. First, a full RandomForest model is trained with all available features
    2. Feature importance scores are calculated (using Gini importance from the RandomForest)
    3. Features with importance below a threshold are filtered out
    4. A new RandomForest model is trained using only the selected features

    This approach offers several benefits:
    - Reduced model complexity and size
    - Faster inference due to fewer features to evaluate
    - Often improved accuracy by focusing on the most discriminative features
    - Less overfitting by eliminating noisy or irrelevant features

    The feature selection threshold determines how aggressive the feature pruning will be:
    - Lower thresholds (e.g., 0.001) retain more features
    - Higher thresholds (e.g., 0.05) retain only the most important features

    The max_features parameter can be used to set an absolute limit on the number of
    features, regardless of their importance scores.

    This model supports conversion to ONNX format when the 'onnx' optional
    dependency is installed. ONNX models can be used for faster inference,
    especially when deployed in production environments.
    """

    def __init__(
        self,
        feature_selection_threshold: float = 0.01,
        max_features: int = None,
        threshold: float = 0.5,
        use_onnx: bool = False,
        **kwargs,
    ):
        """
        Initialize the FeatureSelectedRandomForestModel.

        Args:
            feature_selection_threshold (float, optional): Minimum importance threshold for
                                                          feature selection (0.0-1.0).
                                                          Features with importance below this
                                                          threshold will be filtered out.
                                                          Defaults to 0.01.
            max_features (int, optional): Maximum number of top features to keep.
                                         If None, keep all features above the threshold.
                                         Defaults to None.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        Defaults to 0.5.
            use_onnx (bool, optional): Whether to use ONNX for inference if available.
                                      Defaults to False.
            **kwargs: Parameters to pass to the underlying RandomForestClassifier
        """
        super().__init__(threshold=threshold, use_onnx=use_onnx, **kwargs)
        self.feature_selection_threshold = feature_selection_threshold
        self.max_features = max_features
        self.selected_feature_indices = None
        self.feature_importances = None

    def fit(self, X: List[List[int]], y: List[int]) -> None:
        """
        Fit the model with feature selection.

        Args:
            X (List[List[int]]): Feature vectors
            y (List[int]): Target labels (0 for non-boundary, 1 for boundary)
        """
        # Ensure binary labels
        y_binary = [1 if label > 0 else 0 for label in y]

        print("Initial training to determine feature importance...")
        # First train the model normally to determine feature importance
        self.model.fit(X=X, y=y_binary)

        # Get feature importances
        self.feature_importances = self.model.feature_importances_

        # Select important features
        selected_features = self._select_important_features(self.feature_importances)
        num_selected = len(selected_features)
        total_features = len(self.feature_importances)
        print(
            f"Selected {num_selected} features out of {total_features} ({num_selected / total_features:.1%})"
        )

        # Store selected feature indices for prediction
        self.selected_feature_indices = selected_features

        # Extract selected features only
        X_selected = [[x[i] for i in selected_features] for x in X]

        print("Retraining model with selected features...")
        # Reinitialize the model with the same parameters
        self.model = sklearn.ensemble.RandomForestClassifier(**self.model_params)

        # Retrain with selected features
        self.model.fit(X=X_selected, y=y_binary)

        # Save feature names and importance (if available)
        self.selected_feature_importance = self.model.feature_importances_

    def predict(
        self, X: List[List[int]], threshold: Optional[float] = None
    ) -> List[int]:
        """
        Predict segmentation labels for the given features.

        Args:
            X (List[List[int]]): Feature vectors
            threshold (float, optional): Custom probability threshold to use for this prediction.
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[int]: Predicted labels (0 for non-boundary, 1 for boundary)
        """
        # Use selected features if available
        if self.selected_feature_indices is not None:
            X = [[x[i] for i in self.selected_feature_indices] for x in X]

        # Use custom threshold if provided, otherwise use the model's default
        thresh = threshold if threshold is not None else self.threshold

        if thresh == 0.5:
            # Use the default scikit-learn prediction for the default threshold
            return self.model.predict(X)
        else:
            # Get class probabilities and apply custom threshold
            probas = self.model.predict_proba(X)
            # Class 1 (boundary) is typically the second column
            return [1 if proba[1] >= thresh else 0 for proba in probas]

    def predict_proba(self, X: List[List[int]]) -> List[List[float]]:
        """
        Predict class probabilities for the given features.

        Args:
            X (List[List[int]]): Feature vectors

        Returns:
            List[List[float]]: Predicted probabilities for each class
        """
        # Use selected features if available
        if self.selected_feature_indices is not None:
            X = [[x[i] for i in self.selected_feature_indices] for x in X]

        return self.model.predict_proba(X).tolist()

    def _select_important_features(self, importances):
        """
        Select important features based on feature importance scores.

        Args:
            importances (numpy.ndarray): Feature importance scores from the RandomForest.

        Returns:
            List[int]: Indices of selected features.
        """
        # Create a list of (index, importance) tuples
        indexed_importances = [(i, imp) for i, imp in enumerate(importances)]

        # Sort by importance in descending order
        sorted_importances = sorted(
            indexed_importances, key=lambda x: x[1], reverse=True
        )

        # Filter based on threshold
        thresholded_features = [
            idx
            for idx, imp in sorted_importances
            if imp >= self.feature_selection_threshold
        ]

        # Apply max_features limit if specified
        if (
            self.max_features is not None
            and len(thresholded_features) > self.max_features
        ):
            return thresholded_features[: self.max_features]

        return thresholded_features

    def get_feature_importances(self) -> Dict[str, Any]:
        """
        Get feature importances and selection information.

        Returns:
            Dict[str, Any]: Feature importance information
        """
        result = {
            "original_num_features": len(self.feature_importances)
            if self.feature_importances is not None
            else 0,
            "selected_num_features": len(self.selected_feature_indices)
            if self.selected_feature_indices is not None
            else 0,
            "selection_threshold": self.feature_selection_threshold,
            "selected_indices": self.selected_feature_indices
            if self.selected_feature_indices is not None
            else [],
        }

        if self.feature_importances is not None:
            result["original_importances"] = self.feature_importances.tolist()

        if self.selected_feature_importance is not None:
            result["selected_importances"] = self.selected_feature_importance.tolist()

        return result


def create_model(
    model_type: str = "random_forest",
    threshold: float = 0.5,
    use_onnx: bool = False,
    onnx_optimization_level: int = 1,
    **kwargs,
) -> TextSegmentationModel:
    """
    Create a text segmentation model.

    Args:
        model_type (str): Type of model to create
                        - "random_forest" or "binary_random_forest": Regular RandomForest model
                        - "feature_selected_rf": RandomForest with feature selection
        threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                   Values below 0.5 favor recall (fewer false negatives),
                                   values above 0.5 favor precision (fewer false positives).
                                   Defaults to 0.5.
        use_onnx (bool, optional): Whether to use ONNX for inference if available.
                                  Requires the 'onnx' optional dependency.
                                  Defaults to False.
        onnx_optimization_level (int, optional): ONNX optimization level (0-3).
                                               0: No optimization
                                               1: Basic optimizations (default)
                                               2: Extended optimizations
                                               3: All optimizations including extended memory reuse
                                               Defaults to 1.
        **kwargs: Parameters to pass to the model constructor

    Returns:
        TextSegmentationModel: A text segmentation model instance

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type.lower() in ["random_forest", "binary_random_forest"]:
        return BinaryRandomForestModel(
            threshold=threshold,
            use_onnx=use_onnx,
            onnx_optimization_level=onnx_optimization_level,
            **kwargs,
        )
    elif model_type.lower() in [
        "feature_selected_rf",
        "feature_selected_random_forest",
    ]:
        # Extract feature selection parameters
        feature_selection_threshold = kwargs.pop("feature_selection_threshold", 0.01)
        max_features = kwargs.pop("max_features", None)
        return FeatureSelectedRandomForestModel(
            feature_selection_threshold=feature_selection_threshold,
            max_features=max_features,
            threshold=threshold,
            use_onnx=use_onnx,
            onnx_optimization_level=onnx_optimization_level,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: 'random_forest', 'feature_selected_rf'"
        )
