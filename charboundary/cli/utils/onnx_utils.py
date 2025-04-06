"""
ONNX management utilities for CharBoundary CLI.

This module provides functions for the management of ONNX models:
- Converting models to ONNX format with optimization levels
- Benchmarking ONNX models against scikit-learn
- Testing ONNX model compatibility and accuracy
"""

import os
import time
import random
import warnings
from typing import List, Dict, Any, Tuple, Optional

# Check if ONNX is available
try:
    from charboundary.onnx_support import check_onnx_available

    ONNX_AVAILABLE = check_onnx_available()
except ImportError:
    ONNX_AVAILABLE = False

from charboundary import (
    get_small_segmenter,
    get_default_segmenter,
    get_large_segmenter,
    get_small_onnx_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter,
    TextSegmenter,
)
from charboundary.models import (
    BinaryRandomForestModel,
    FeatureSelectedRandomForestModel,
)

# Test text for segmentation
TEST_TEXT = """
The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that 
racial segregation in public schools was unconstitutional. This landmark 
decision changed the course of U.S. history.

Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson 
at 2:30 p.m. They discussed Mr. Brown's case regarding Section 1.2.3 of 
the tax code.

"This is a direct quote," said the author. "It contains multiple sentences. 
And it spans multiple lines." The audience nodded in agreement.
"""

# Configuration for built-in models
BUILTIN_MODELS = {
    "small": {
        "getter": get_small_segmenter,
        "onnx_getter": get_small_onnx_segmenter,
        "feature_count": 19,
        "optimal_level": 2,
    },
    "medium": {
        "getter": get_default_segmenter,
        "onnx_getter": get_medium_onnx_segmenter,
        "feature_count": 21,
        "optimal_level": 2,
    },
    "large": {
        "getter": get_large_segmenter,
        "onnx_getter": get_large_onnx_segmenter,
        "feature_count": 27,
        "optimal_level": 3,
    },
}

#############################################################################
# Conversion functions
#############################################################################


def infer_feature_count(model) -> int:
    """Try to infer the feature count from the model."""
    # For feature-selected models, get the count of selected features
    if hasattr(model, "selected_feature_indices") and model.selected_feature_indices:
        return len(model.selected_feature_indices)

    # For standard RandomForest models, get the feature count from the first tree
    if hasattr(model, "model") and hasattr(model.model, "n_features_in_"):
        return model.model.n_features_in_

    # Default values for known models
    model_sizes = {
        "small": 19,  # Small model has 19 features
        "medium": 21,  # Medium model has 21 features
        "large": 27,  # Large model has 27 features
    }

    # If we can identify the model type from its name
    for model_type, feature_count in model_sizes.items():
        if model_type in str(model).lower():
            print(
                f"Using default feature count of {feature_count} for {model_type} model"
            )
            return feature_count

    # If we can't determine the feature count, return a default value and warn
    warnings.warn(
        "Could not determine feature count from model. Using default value of 32. "
        "If conversion fails, please specify the feature count manually."
    )
    return 32


def convert_segmenter_model(
    model_getter,
    model_name: str,
    output_path: Optional[str] = None,
    optimization_level: int = 2,
) -> bool:
    """
    Convert a segmenter model to ONNX format.

    Args:
        model_getter: Function that returns a segmenter
        model_name: Name of the model
        output_path: Path to save the ONNX model (if None, use default path)
        optimization_level: ONNX optimization level (0-3)

    Returns:
        bool: True if successful, False otherwise
    """
    if optimization_level not in [0, 1, 2, 3]:
        print(
            f"Error: Invalid optimization level {optimization_level}. Must be between 0-3."
        )
        return False

    print(f"Loading {model_name} model...")

    # Get the segmenter
    try:
        segmenter = model_getter()
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Get the model from the segmenter
    model = segmenter.model

    # Set the feature count - this is required for ONNX conversion
    feature_count = infer_feature_count(model)
    model.feature_count = feature_count
    print(f"Set feature count to {feature_count}")

    # Set optimization level
    try:
        model.onnx_optimization_level = optimization_level
    except Exception as e:
        print(f"Warning: Unable to set optimization level attribute: {e}")
        print("Will attempt to set during ONNX conversion")

    print(f"Using optimization level: {optimization_level}")

    # Determine output path if not provided
    if output_path is None:
        # Get package directory
        package_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_dir = os.path.join(package_dir, "resources", "onnx")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_model.onnx")

    # Convert to ONNX
    print(f"Converting {model_name} model to ONNX...")
    try:
        # Convert to ONNX
        onnx_model = model.to_onnx()

        if onnx_model is None:
            print(
                f"Error: ONNX conversion failed for {model_name} model. No model was generated."
            )
            return False

        # Save the ONNX model
        print(f"Saving ONNX model to {output_path}...")
        if model.save_onnx(output_path):
            # Check for both the file and the compressed version
            if os.path.exists(output_path):
                file_path = output_path
            elif os.path.exists(f"{output_path}.xz"):
                file_path = f"{output_path}.xz"
            else:
                print(
                    f"Warning: Model saved successfully but file not found at {output_path} or {output_path}.xz"
                )
                return True

            print(f"Successfully saved {model_name} ONNX model to {file_path}")
            print(f"Model size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"Error: Failed to save {model_name} ONNX model.")
            return False

    except Exception as e:
        print(f"Error converting {model_name} model to ONNX: {e}")
        return False


def convert_custom_model(
    input_path: str,
    output_path: str,
    feature_count: Optional[int] = None,
    optimization_level: int = 2,
) -> bool:
    """
    Convert a custom model file to ONNX format.

    Args:
        input_path: Path to the input model file
        output_path: Path to save the ONNX model
        feature_count: Number of features in the model (optional)
        optimization_level: ONNX optimization level (0-3)

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return False

    if optimization_level not in [0, 1, 2, 3]:
        print(
            f"Error: Invalid optimization level {optimization_level}. Must be between 0-3."
        )
        return False

    print(f"Loading model from {input_path}...")

    try:
        # Try to load as a segmenter (which contains the full model)
        segmenter = TextSegmenter.load(input_path, trust_model=True)
        model = segmenter.model
        print("Successfully loaded model from TextSegmenter.")
    except Exception as e:
        print(f"Error loading as TextSegmenter: {e}")
        print("Trying to load as raw model...")

        try:
            # Try to use pickle loading for standalone models
            import pickle
            import lzma

            if input_path.endswith(".xz"):
                with lzma.open(input_path, "rb") as f:
                    model = pickle.load(f)
            else:
                with open(input_path, "rb") as f:
                    model = pickle.load(f)

            print("Successfully loaded raw model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # Verify we have a proper model
    if not isinstance(
        model, (BinaryRandomForestModel, FeatureSelectedRandomForestModel)
    ):
        print(f"Error: Expected a charboundary model, but got {type(model)}.")
        return False

    # Infer feature count if not provided
    if feature_count is None:
        feature_count = infer_feature_count(model)
        print(f"Inferred feature count: {feature_count}")

    # Set the feature count on the model (needed for ONNX conversion)
    model.feature_count = feature_count

    # Set optimization level
    try:
        model.onnx_optimization_level = optimization_level
    except Exception as e:
        print(f"Warning: Unable to set optimization level attribute: {e}")
        print("Will attempt to set during ONNX conversion")

    print(f"Using optimization level: {optimization_level}")

    # Convert to ONNX
    print(f"Converting model to ONNX with {feature_count} features...")
    try:
        onnx_model = model.to_onnx()

        if onnx_model is None:
            print("Error: ONNX conversion failed. No model was generated.")
            return False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the ONNX model
        print(f"Saving ONNX model to {output_path}...")
        if model.save_onnx(output_path):
            # Check for both the file and the compressed version
            if os.path.exists(output_path):
                file_path = output_path
            elif os.path.exists(f"{output_path}.xz"):
                file_path = f"{output_path}.xz"
            else:
                print(
                    f"Warning: Model saved successfully but file not found at {output_path} or {output_path}.xz"
                )
                return True

            print(f"Successfully saved ONNX model to {file_path}")
            print(f"Model size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print("Error: Failed to save ONNX model.")
            return False

    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
        return False


def convert_all_models(optimization_levels: Optional[Dict[str, int]] = None) -> bool:
    """
    Convert all built-in models to ONNX format with specified optimization levels.

    Args:
        optimization_levels: Dictionary mapping model names to optimization levels
                            If None, use the optimal levels for each model

    Returns:
        bool: True if all conversions were successful, False otherwise
    """
    print("Converting all charboundary models to ONNX format...")

    # Use optimal levels if not specified
    if optimization_levels is None:
        optimization_levels = {
            name: info["optimal_level"] for name, info in BUILTIN_MODELS.items()
        }

    # Track success
    success = True

    # Convert each model
    for model_name, model_info in BUILTIN_MODELS.items():
        # Get optimization level (use optimal if not specified)
        opt_level = optimization_levels.get(model_name, model_info["optimal_level"])

        # Convert the model
        result = convert_segmenter_model(
            model_info["getter"], model_name, optimization_level=opt_level
        )

        # Track success
        success = success and result

    if success:
        print("\nAll models have been successfully converted to ONNX format.")
    else:
        print("\nSome models failed to convert. See above for details.")

    return success


#############################################################################
# Benchmark functions
#############################################################################


def generate_benchmark_data(feature_count: int, batch_size: int) -> List[List[int]]:
    """Generate random benchmark data for testing."""
    # Create random binary features (typical for charboundary)
    return [
        [random.randint(0, 1) for _ in range(feature_count)] for _ in range(batch_size)
    ]


def run_sklearn_benchmark(
    model, data: List[List[int]], runs: int
) -> Tuple[float, List[int]]:
    """Run sklearn inference benchmark."""
    # Ensure the model has a use_onnx attribute
    if not hasattr(model, "use_onnx"):
        setattr(model, "use_onnx", False)

    # Store original state and disable ONNX
    original_onnx_state = model.use_onnx
    model.use_onnx = False

    # Warmup run
    _ = model.predict(data)

    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = model.predict(data)
    end_time = time.time()

    # Restore original ONNX state
    model.use_onnx = original_onnx_state

    total_time = end_time - start_time
    return total_time, predictions


def run_onnx_benchmark(
    model, data: List[List[int]], runs: int, optimization_level: int
) -> Tuple[float, List[int]]:
    """Run ONNX inference benchmark with specified optimization level."""
    # Ensure model has all required attributes
    if not hasattr(model, "use_onnx"):
        setattr(model, "use_onnx", False)

    if not hasattr(model, "onnx_model"):
        setattr(model, "onnx_model", None)

    if not hasattr(model, "onnx_session"):
        setattr(model, "onnx_session", None)

    if not hasattr(model, "onnx_optimization_level"):
        setattr(model, "onnx_optimization_level", optimization_level)

    # Make sure the model has a feature count
    if not hasattr(model, "feature_count") or model.feature_count is None:
        # Try to infer feature count
        if (
            hasattr(model, "selected_feature_indices")
            and model.selected_feature_indices
        ):
            model.feature_count = len(model.selected_feature_indices)
        elif hasattr(model, "model") and hasattr(model.model, "n_features_in_"):
            model.feature_count = model.model.n_features_in_
        else:
            model.feature_count = len(data[0])  # Use input data as a guide

    # Convert to ONNX if needed
    if model.onnx_model is None:
        try:
            model.to_onnx()
        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            raise

    # Make sure ONNX is enabled with the specified optimization level
    try:
        model.enable_onnx(True, optimization_level=optimization_level)
    except Exception as e:
        print(f"Error enabling ONNX with optimization level {optimization_level}: {e}")
        raise

    # Warmup run
    _ = model.predict(data)

    # Timed runs
    start_time = time.time()
    predictions = None
    for _ in range(runs):
        predictions = model.predict(data)
    end_time = time.time()

    total_time = end_time - start_time
    return total_time, predictions


def benchmark_model_optimization_levels(
    model, feature_count: int, model_name: str, runs: int = 20, batch_size: int = 500
) -> Dict[str, Any]:
    """
    Benchmark a model with different ONNX optimization levels.

    Args:
        model: The model to benchmark
        feature_count: Number of features in the model
        model_name: Name of the model (for reporting)
        runs: Number of benchmark runs
        batch_size: Number of samples per batch

    Returns:
        Dict: Benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking {model_name.upper()} model")
    print(f"{'=' * 70}")

    print(f"Model: {model_name}")
    print(f"Feature count: {feature_count}")

    # Generate benchmark data
    print(f"Generating benchmark data (batch size: {batch_size})...")
    data = generate_benchmark_data(feature_count, batch_size)

    # Run sklearn benchmark
    print(f"Running sklearn benchmark ({runs} runs)...")
    sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, runs)
    sklearn_ops = (runs * batch_size) / sklearn_time

    # Dictionary to store results
    results = {
        "model_name": model_name,
        "sklearn_time": sklearn_time,
        "sklearn_ops_per_sec": sklearn_ops,
        "feature_count": feature_count,
        "batch_size": batch_size,
        "runs": runs,
        "onnx_results": {},
    }

    # Test all ONNX optimization levels
    print("\nBenchmarking ONNX optimization levels:")
    print(
        f"{'Level':10} {'Description':25} {'Time (s)':10} {'Ops/sec':15} {'Rel. Speedup':12}"
    )
    print("-" * 75)

    optimization_levels = [
        (0, "No optimization"),
        (1, "Basic optimizations"),
        (2, "Extended optimizations"),
        (3, "All optimizations"),
    ]

    for level, description in optimization_levels:
        print(f"Level {level}: Testing {description}...", end="", flush=True)

        try:
            # Run benchmark with this optimization level
            onnx_time, onnx_predictions = run_onnx_benchmark(model, data, runs, level)
            onnx_ops = (runs * batch_size) / onnx_time
            speedup = onnx_ops / sklearn_ops

            # Verify results match
            predictions_match = sklearn_predictions == onnx_predictions
            if isinstance(predictions_match, list):
                predictions_match = all(predictions_match)
            elif hasattr(predictions_match, "all"):  # For numpy arrays
                predictions_match = predictions_match.all()

            # Store results
            results["onnx_results"][level] = {
                "time": onnx_time,
                "ops_per_sec": onnx_ops,
                "speedup": speedup,
                "predictions_match": predictions_match,
            }

            # Print results
            accuracy_indicator = "✓" if predictions_match else "✗"
            print(
                f"\r{level:10} {description:25} {onnx_time:10.4f} {onnx_ops:15.2f} {speedup:11.2f}x {accuracy_indicator}"
            )

        except Exception as e:
            print(f"\nError benchmarking ONNX level {level}: {e}")
            results["onnx_results"][level] = {
                "error": str(e),
                "ops_per_sec": 0,
                "speedup": 0,
                "predictions_match": False,
            }

    # Find the best optimization level
    successful_levels = {
        k: v
        for k, v in results["onnx_results"].items()
        if v.get("error") is None and v.get("predictions_match", False)
    }

    if successful_levels:
        best_level = max(successful_levels.items(), key=lambda x: x[1]["ops_per_sec"])
        best_level_num, best_level_results = best_level

        print(f"\nBest performance with ONNX optimization level {best_level_num}.")
        print(
            f"ONNX with level {best_level_num} is {best_level_results['speedup']:.2f}x faster than scikit-learn."
        )
    else:
        print("\nNo successful ONNX benchmarks found.")

    return results


def benchmark_built_in_models(runs: int = 20, batch_size: int = 500) -> Dict[str, Any]:
    """
    Benchmark all built-in models with their recommended optimization levels.

    Args:
        runs: Number of benchmark runs
        batch_size: Number of samples per batch

    Returns:
        Dict: Benchmark results for all models
    """
    results = {}

    for model_name, model_info in BUILTIN_MODELS.items():
        # Get the segmenter with the model
        segmenter = model_info["getter"]()
        model = segmenter.model

        # Ensure feature count is set
        model.feature_count = model_info["feature_count"]

        # Benchmark sklearn vs ONNX with optimal level
        print(f"\n{'=' * 70}")
        print(
            f"Benchmarking {model_name.upper()} model with optimization level {model_info['optimal_level']}"
        )
        print(f"{'=' * 70}")

        # Generate benchmark data
        data = generate_benchmark_data(model_info["feature_count"], batch_size)

        try:
            # Benchmark sklearn
            sklearn_time, sklearn_predictions = run_sklearn_benchmark(model, data, runs)
            sklearn_ops = (runs * batch_size) / sklearn_time
            print(
                f"scikit-learn: {sklearn_ops:.2f} ops/sec, {sklearn_time:.4f} seconds"
            )

            # Benchmark ONNX with optimal level
            onnx_time, onnx_predictions = run_onnx_benchmark(
                model, data, runs, model_info["optimal_level"]
            )
            onnx_ops = (runs * batch_size) / onnx_time
            speedup = onnx_ops / sklearn_ops

            # Check if predictions match
            predictions_match = sklearn_predictions == onnx_predictions
            if isinstance(predictions_match, list):
                predictions_match = all(predictions_match)
            elif hasattr(predictions_match, "all"):  # For numpy arrays
                predictions_match = predictions_match.all()

            accuracy_indicator = "✓" if predictions_match else "✗"
            print(
                f"ONNX (level {model_info['optimal_level']}): {onnx_ops:.2f} ops/sec, {onnx_time:.4f} seconds"
            )
            print(f"Speedup: {speedup:.2f}x {accuracy_indicator}")

            # Save results
            results[model_name] = {
                "feature_count": model_info["feature_count"],
                "sklearn_ops": sklearn_ops,
                "onnx_ops": onnx_ops,
                "speedup": speedup,
                "opt_level": model_info["optimal_level"],
                "predictions_match": predictions_match,
            }

        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Print a summary table
    print("\n" + "=" * 80)
    print("ONNX PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':10} {'Opt. Level':15} {'sklearn (ops/s)':20} {'ONNX (ops/s)':20} {'Speedup':10}"
    )
    print("-" * 80)

    successful_results = {k: v for k, v in results.items() if "error" not in v}

    for model_name, model_results in successful_results.items():
        level_str = f"Level {model_results['opt_level']}"
        print(
            f"{model_name:10} {level_str:15} "
            f"{model_results['sklearn_ops']:20.2f} {model_results['onnx_ops']:20.2f} "
            f"{model_results['speedup']:10.2f}x"
        )

    return results


#############################################################################
# Testing functions
#############################################################################


def test_model(model_name: str, optimization_level: Optional[int] = None):
    """Test a specific ONNX model."""
    print(f"\nTesting {model_name} ONNX model:")

    # Get model information
    if model_name not in BUILTIN_MODELS:
        print(f"✗ Unknown model name: {model_name}")
        return False

    model_info = BUILTIN_MODELS[model_name]

    # Use the specified optimization level or the optimal one
    if optimization_level is None:
        optimization_level = model_info["optimal_level"]

    # Load the model
    try:
        # Load using ONNX getter
        segmenter = model_info["onnx_getter"]()
        print("✓ Model successfully loaded")

        # Check if the model has ONNX enabled
        onnx_enabled = hasattr(segmenter.model, "use_onnx") and segmenter.model.use_onnx
        print(f"✓ ONNX enabled: {onnx_enabled}")

        # Try to get current optimization level
        try:
            current_level = getattr(
                segmenter.model, "onnx_optimization_level", "unknown"
            )
            print(f"✓ Current optimization level: {current_level}")
        except Exception as e:
            print(f"✗ Error getting optimization level: {str(e)}")

        # Try to set optimization level
        try:
            if hasattr(segmenter.model, "onnx_optimization_level"):
                segmenter.model.onnx_optimization_level = optimization_level
                print(f"✓ Set optimization level attribute to {optimization_level}")
        except Exception as e:
            print(f"✗ Error setting optimization level attribute: {str(e)}")

        # Try to enable ONNX with optimization level
        try:
            if hasattr(segmenter.model, "enable_onnx"):
                segmenter.model.enable_onnx(True, optimization_level=optimization_level)
                print(
                    f"✓ Called enable_onnx with optimization level {optimization_level}"
                )
        except Exception as e:
            print(f"✗ Error calling enable_onnx: {str(e)}")

        # Manually load the specific ONNX file
        try:
            # Get the path to the ONNX model
            package_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            onnx_dir = os.path.join(package_dir, "resources", "onnx")
            onnx_file = os.path.join(onnx_dir, f"{model_name}_model.onnx")

            if os.path.exists(onnx_file):
                print(f"✓ Found ONNX model at {onnx_file}")

                try:
                    # Load the ONNX model
                    segmenter.load_onnx(onnx_file)
                    print("✓ Loaded ONNX model from file")
                except Exception as e:
                    print(f"✗ Error loading ONNX model from file: {str(e)}")
            else:
                print(f"✗ ONNX model file not found at {onnx_file}")
        except Exception as e:
            print(f"✗ Error accessing ONNX file: {str(e)}")

        # Test segmentation
        start_time = time.time()
        sentences = segmenter.segment_to_sentences(TEST_TEXT)
        end_time = time.time()

        # Print results
        print("✓ Segmentation successful")
        print(f"✓ Found {len(sentences)} sentences")
        print(f"✓ Processing time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"✓ Sample sentence: {sentences[0][:60]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_all_models(optimization_levels: Optional[Dict[str, int]] = None):
    """Test all built-in models with their optimal or specified optimization levels."""
    print("Testing ONNX models via charboundary package functions")
    print("=====================================================")

    # Use optimal levels if not specified
    if optimization_levels is None:
        optimization_levels = {
            name: info["optimal_level"] for name, info in BUILTIN_MODELS.items()
        }

    # Track success
    success = True

    # Test each model
    for model_name in BUILTIN_MODELS:
        opt_level = optimization_levels.get(model_name)
        result = test_model(model_name, opt_level)
        success = success and result

    print("\nAll tests completed!")
    return success


# CLI command functions will be imported by main.py
