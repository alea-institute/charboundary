"""
Model definitions and interfaces for the charboundary library.
"""

from typing import List, Dict, Any, Protocol, Optional
import sklearn.ensemble
import sklearn.metrics
from sklearn.base import BaseEstimator


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
    """
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        Initialize the BinaryRandomForestModel.
        
        Args:
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        Defaults to 0.5.
            **kwargs: Parameters to pass to the underlying RandomForestClassifier
        """
        self.threshold = threshold
        self.model_params = kwargs.copy() if kwargs else {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }
        
        # Set class weight to 'balanced' to handle imbalanced data
        if 'class_weight' not in self.model_params:
            self.model_params['class_weight'] = 'balanced'
            
        self.model = sklearn.ensemble.RandomForestClassifier(**self.model_params)
        
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
            
        self.model.fit(X=X, y=y_binary)
        
    def predict(self, X: List[List[int]], threshold: Optional[float] = None) -> List[int]:
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
            "binary_mode": True
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
            boundary_indices = [i for i, (t, p) in enumerate(zip(y_binary, predictions)) 
                              if t == 1 or p == 1]
            
            if boundary_indices:
                boundary_true = [y_binary[i] for i in boundary_indices]
                boundary_pred = [predictions[i] for i in boundary_indices]
                boundary_accuracy = sklearn.metrics.accuracy_score(boundary_true, boundary_pred)
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
                if k not in ["accuracy", "macro avg", "weighted avg", "Non-boundary", "Boundary"]:
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


# Factory function for creating models
def create_model(model_type: str = "random_forest", threshold: float = 0.5, **kwargs) -> TextSegmentationModel:
    """
    Create a text segmentation model.
    
    Args:
        model_type (str): Type of model to create (only "random_forest" is supported)
        threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                   Values below 0.5 favor recall (fewer false negatives),
                                   values above 0.5 favor precision (fewer false positives).
                                   Defaults to 0.5.
        **kwargs: Parameters to pass to the model constructor
        
    Returns:
        TextSegmentationModel: A text segmentation model instance
        
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type.lower() in ["random_forest", "binary_random_forest"]:
        return BinaryRandomForestModel(threshold=threshold, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Only 'random_forest' is supported.")