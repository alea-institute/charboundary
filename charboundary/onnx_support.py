"""
ONNX support for charboundary models.

This module provides functionality to convert charboundary models to ONNX format
and to run inference using ONNX runtime.
"""

import os
from typing import List, Any, Union
from pathlib import Path
import warnings

# Import error handling for optional dependencies
try:
    import skl2onnx
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def check_onnx_available() -> bool:
    """
    Check if ONNX and its dependencies are available.

    Returns:
        bool: True if ONNX is available, False otherwise
    """
    return ONNX_AVAILABLE


def convert_to_onnx(
    model: Any, feature_count: int, model_name: str = "charboundary_model"
) -> bytes:
    """
    Convert a scikit-learn model to ONNX format.

    Args:
        model: The scikit-learn model to convert
        feature_count: Number of features (input dimension)
        model_name: Name to give the ONNX model

    Returns:
        bytes: Serialized ONNX model

    Raises:
        ImportError: If ONNX dependencies are not installed
        ValueError: If the model cannot be converted
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX conversion requires 'onnx' and 'skl2onnx' packages. "
            "Install them with: pip install charboundary[onnx]"
        )

    # Define the input type
    initial_type = [("input", FloatTensorType([None, feature_count]))]

    try:
        # Convert the model to ONNX
        onnx_model = skl2onnx.convert_sklearn(
            model=model,
            name=model_name,
            initial_types=initial_type,
            options={
                id(model): {"zipmap": False}
            },  # Disable ZipMap to get raw probabilities
        )

        # Serialize and return
        return onnx_model.SerializeToString()
    except Exception as e:
        raise ValueError(f"Failed to convert model to ONNX format: {str(e)}")


def save_onnx_model(
    onnx_model: bytes, file_path: Union[str, Path], compress: bool = True
) -> None:
    """
    Save an ONNX model to disk, optionally with XZ compression.

    Args:
        onnx_model: Serialized ONNX model
        file_path: Path to save the model
        compress: Whether to compress the model with XZ (default: True)

    Raises:
        ImportError: If ONNX is not installed
        IOError: If the model cannot be saved
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support requires 'onnx' package. "
            "Install it with: pip install charboundary[onnx]"
        )

    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Check if we should compress
        if compress and not (
            str(file_path).endswith(".xz") or str(file_path).endswith(".lzma")
        ):
            import lzma

            # Compressed write to file
            with lzma.open(str(file_path) + ".xz", "wb") as f:
                f.write(onnx_model)
        else:
            # Standard write to file
            with open(file_path, "wb") as f:
                f.write(onnx_model)
    except Exception as e:
        raise IOError(f"Failed to save ONNX model to {file_path}: {str(e)}")


def load_onnx_model(file_path: Union[str, Path]) -> bytes:
    """
    Load an ONNX model from disk, handles compressed (.xz, .lzma) files automatically.

    Args:
        file_path: Path to the ONNX model file

    Returns:
        bytes: Serialized ONNX model

    Raises:
        ImportError: If ONNX is not installed
        FileNotFoundError: If the model file does not exist
        IOError: If the model cannot be loaded
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support requires 'onnx' package. "
            "Install it with: pip install charboundary[onnx]"
        )

    # Convert to string for easier handling
    path_str = str(file_path)

    # Check for XZ or LZMA compression
    is_compressed = path_str.endswith(".xz") or path_str.endswith(".lzma")

    # Try original path first
    if os.path.exists(path_str):
        try:
            if is_compressed:
                import lzma

                with lzma.open(path_str, "rb") as f:
                    return f.read()
            else:
                with open(path_str, "rb") as f:
                    return f.read()
        except Exception as e:
            raise IOError(f"Failed to load ONNX model from {path_str}: {str(e)}")

    # If not found, try adding compression extension
    if not is_compressed:
        compressed_path = path_str + ".xz"
        if os.path.exists(compressed_path):
            try:
                import lzma

                with lzma.open(compressed_path, "rb") as f:
                    return f.read()
            except Exception as e:
                raise IOError(
                    f"Failed to load compressed ONNX model from {compressed_path}: {str(e)}"
                )

    # If we reach here, file was not found
    raise FileNotFoundError(
        f"ONNX model file not found: {file_path} (also tried with .xz extension)"
    )


def create_onnx_inference_session(
    onnx_model: bytes, optimization_level: int = 1
) -> Any:
    """
    Create an ONNX inference session from a serialized ONNX model.

    Args:
        onnx_model: Serialized ONNX model
        optimization_level: ONNX optimization level (0-3):
            - 0: No optimization
            - 1: Basic optimizations (default)
            - 2: Extended optimizations
            - 3: All optimizations including extended memory reuse

    Returns:
        Any: ONNX inference session

    Raises:
        ImportError: If ONNX runtime is not installed
        RuntimeError: If the session cannot be created
        ValueError: If the optimization level is invalid
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise ValueError("Optimization level must be between 0 and 3")

    try:
        import onnxruntime as ort

        # Create session options with specified optimization level
        session_options = ort.SessionOptions()

        # Set optimization level using the enum
        if optimization_level == 0:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        elif optimization_level == 1:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == 2:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        elif optimization_level == 3:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        # Create inference session with options
        return ort.InferenceSession(onnx_model, session_options)
    except ImportError:
        raise ImportError(
            "ONNX inference requires 'onnxruntime' package. "
            "Install it with: pip install onnxruntime"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create ONNX inference session: {str(e)}")


def onnx_predict(session: Any, X: List[List[int]], threshold: float = 0.5) -> List[int]:
    """
    Run inference using an ONNX session.

    Args:
        session: ONNX inference session
        X: Feature vectors
        threshold: Probability threshold for binary classification

    Returns:
        List[int]: Predicted labels
    """
    try:
        import numpy as np

        # Convert input to float32 numpy array
        X_np = np.array(X, dtype=np.float32)

        # Get the name of the input
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: X_np})

        # For classification models, the output is typically class probabilities
        if len(outputs) > 1:  # Assuming the second output is class probabilities
            probas = outputs[1]  # Probabilities for each class
            # Apply threshold to probabilities of class 1
            return [1 if p[1] >= threshold else 0 for p in probas]
        else:
            # Direct class prediction
            return outputs[0].tolist()

    except ImportError:
        raise ImportError(
            "ONNX inference requires 'numpy' and 'onnxruntime' packages. "
            "Install them with: pip install numpy onnxruntime"
        )
    except Exception as e:
        raise RuntimeError(f"ONNX inference failed: {str(e)}")


def onnx_predict_proba(session: Any, X: List[List[int]]) -> List[List[float]]:
    """
    Get predicted probabilities using an ONNX session.

    Args:
        session: ONNX inference session
        X: Feature vectors

    Returns:
        List[List[float]]: Predicted probabilities for each class
    """
    try:
        import numpy as np

        # Convert input to float32 numpy array
        X_np = np.array(X, dtype=np.float32)

        # Get the name of the input
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: X_np})

        # For classification models, the second output is typically class probabilities
        if len(outputs) > 1:
            return outputs[1].tolist()  # Probabilities for each class
        else:
            # Handle case where probabilities are not directly available
            warnings.warn("ONNX model does not provide probabilities directly")
            # Try to convert raw scores to probabilities using softmax
            scores = outputs[0]
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probas.tolist()

    except ImportError:
        raise ImportError(
            "ONNX inference requires 'numpy' and 'onnxruntime' packages. "
            "Install them with: pip install numpy onnxruntime"
        )
    except Exception as e:
        raise RuntimeError(f"ONNX probability inference failed: {str(e)}")
