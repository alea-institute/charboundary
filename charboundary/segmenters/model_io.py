"""
Model saving and loading functionality for the segmenters module.
"""

import os
import pickle
import tempfile
from typing import Type, TYPE_CHECKING

from skops.io import dump, load

from charboundary.constants import DEFAULT_ABBREVIATIONS
from charboundary.encoders import CharacterEncoder
from charboundary.features import FeatureExtractor

if TYPE_CHECKING:
    from charboundary.segmenters.base import TextSegmenter


class ModelIO:
    """
    Handles saving and loading segmentation models.
    """

    @staticmethod
    def save(
        segmenter: "TextSegmenter",
        path: str,
        format: str = "skops",
        compress: bool = True,
        compression_level: int = 9,
    ) -> None:
        """
        Save the model and configuration to a file.

        Args:
            segmenter: The segmenter to save
            path (str): Path to save the model
            format (str, optional): Serialization format to use ('skops' or 'pickle').
                                    Defaults to 'skops' for secure serialization.
            compress (bool, optional): Whether to use compression. Defaults to True.
            compression_level (int, optional): Compression level (0-9, where 9 is highest).
                                              Defaults to 9.
        """
        if not segmenter.is_trained:
            raise ValueError("Model has not been trained yet.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save all necessary information to recreate the model
        data = {
            "model": segmenter.model,
            "encoder_cache": segmenter.encoder.cache,
            "config": segmenter.config,
            "version": 5,  # Version for backward compatibility (5 = with compression)
            "compressed": compress,
        }

        # Determine if we need to add a compression extension
        compressed_path = None
        if compress and not (path.endswith(".xz") or path.endswith(".lzma")):
            compressed_path = path + ".xz"

        if format.lower() == "skops":
            # Use skops for secure serialization
            if compress:
                # Create a temporary buffer to hold the serialized data
                import io
                import lzma

                # Create a BytesIO buffer to hold the intermediate result
                buffer = io.BytesIO()

                # Serialize to the buffer using skops
                dump(data, buffer)

                # Get the serialized content
                buffer.seek(0)
                serialized_data = buffer.read()

                # Compress the serialized data using LZMA
                compressed_data = lzma.compress(
                    serialized_data, preset=compression_level
                )

                # Write the compressed data to disk - use compressed_path if specified
                save_path = compressed_path if compressed_path else path
                with open(save_path, "wb") as f:
                    f.write(compressed_data)

                # Remove the uncompressed file if both paths exist and are different
                if compressed_path and os.path.exists(path) and path != save_path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass  # Ignore errors when removing
            else:
                # Regular uncompressed saving
                dump(data, path)
        else:
            # Fallback to pickle format (less secure)
            if compress:
                import lzma

                # Use compressed_path if specified
                save_path = compressed_path if compressed_path else path
                with lzma.open(save_path, "wb", preset=compression_level) as f:
                    pickle.dump(data, f)

                # Remove the uncompressed file if it exists
                if compressed_path and os.path.exists(path) and path != save_path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass  # Ignore errors when removing
            else:
                with open(path, "wb") as f:
                    pickle.dump(data, f)

    @classmethod
    def load(
        cls,
        path: str,
        segmenter_class: Type["TextSegmenter"],
        use_skops: bool = True,
        trust_model: bool = False,
    ) -> "TextSegmenter":
        """
        Load a model and configuration from a file.

        Args:
            path (str): Path to load the model from
            segmenter_class: The TextSegmenter class to instantiate
            use_skops (bool, optional): Whether to use skops to load the model. Defaults to True.
            trust_model (bool, optional): Whether to trust all types in the model file.
                                         Set to True only if you trust the source of the model file.
                                         Defaults to False.

        Returns:
            TextSegmenter: Loaded TextSegmenter instance
        """
        # Check for compression extensions and try alternative paths if needed
        paths_to_try = [path]

        # If the path doesn't end with a compression extension, also try with extensions
        if not (path.endswith(".xz") or path.endswith(".lzma")):
            paths_to_try.append(path + ".xz")
            paths_to_try.append(path + ".lzma")

        # Initialize variables for loading
        data = None
        last_exception = None

        # Try each path until one works
        for try_path in paths_to_try:
            if not os.path.exists(try_path):
                continue

            try:
                # Detect if file is compressed (looking at first few bytes)
                is_compressed = False
                with open(try_path, "rb") as test_file:
                    # LZMA files start with 0xFD, '7', 'z', 'X', 'Z', 0x00
                    file_start = test_file.read(6)
                    if file_start.startswith(b"\xfd7zXZ\x00"):
                        is_compressed = True

                if use_skops:
                    try:
                        if is_compressed:
                            # Handle compressed skops file
                            import io
                            import lzma

                            # Read and decompress the file
                            with open(try_path, "rb") as f:
                                compressed_data = f.read()

                            # Decompress the data
                            decompressed_data = lzma.decompress(compressed_data)

                            # Create a BytesIO buffer with the decompressed data
                            buffer = io.BytesIO(decompressed_data)

                            # Load using skops
                            if trust_model:
                                # Trust all types in the file (use with caution)
                                from skops.io import get_untrusted_types

                                # Need a temporary file to get untrusted types
                                with tempfile.NamedTemporaryFile(
                                    delete=False
                                ) as temp_file:
                                    temp_file.write(decompressed_data)
                                    temp_file.flush()
                                    temp_path = temp_file.name

                                try:
                                    # Get untrusted types from the temp file
                                    untrusted_types = get_untrusted_types(
                                        file=temp_path
                                    )
                                    buffer.seek(0)  # Reset buffer position
                                    data = load(buffer, trusted=untrusted_types)
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                            else:
                                data = load(buffer)

                        else:
                            # Regular uncompressed skops file
                            if trust_model:
                                # Trust all types in the file (use with caution)
                                from skops.io import get_untrusted_types

                                untrusted_types = get_untrusted_types(file=try_path)
                                data = load(try_path, trusted=untrusted_types)
                            else:
                                # Only load trusted types
                                data = load(try_path)
                    except Exception as e:
                        if "UntrustedTypesFoundException" in str(e):
                            # Handle the specific case of untrusted types
                            print(
                                f"Warning: Untrusted types found in model file. "
                                f"Attempting to load with untrusted types: {e}"
                            )

                            # Try to load with untrusted types
                            from skops.io import get_untrusted_types

                            if is_compressed:
                                # Need a temporary file to get untrusted types for compressed file
                                # Get the untrusted types using a temporary file
                                with open(try_path, "rb") as f:
                                    compressed_data = f.read()
                                decompressed_data = lzma.decompress(compressed_data)

                                with tempfile.NamedTemporaryFile(
                                    delete=False
                                ) as temp_file:
                                    temp_file.write(decompressed_data)
                                    temp_file.flush()
                                    temp_path = temp_file.name

                                try:
                                    # Get untrusted types from the temp file
                                    untrusted_types = get_untrusted_types(
                                        file=temp_path
                                    )

                                    # Load with all types trusted (for default model)
                                    buffer = io.BytesIO(decompressed_data)
                                    data = load(buffer, trusted=untrusted_types)
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                            else:
                                untrusted_types = get_untrusted_types(file=try_path)

                                # Register our custom types if possible
                                try:
                                    from skops.io import register_trusted_types

                                    # Import the specific types we need
                                    from charboundary.models import (
                                        BinaryRandomForestModel,
                                    )
                                    from charboundary.segmenters.types import (
                                        SegmenterConfig,
                                    )

                                    register_trusted_types(BinaryRandomForestModel)
                                    register_trusted_types(SegmenterConfig)
                                except (ImportError, NameError):
                                    pass

                                # Load with all types trusted (for default model)
                                data = load(try_path, trusted=untrusted_types)
                        else:
                            # Re-raise other exceptions
                            raise
                else:
                    # Fallback to pickle (less secure)
                    if is_compressed:
                        import lzma

                        with lzma.open(try_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(try_path, "rb") as f:
                            data = pickle.load(f)

                # If we reach here, we successfully loaded the data
                break

            except Exception as e:
                last_exception = e
                continue

        # If we couldn't load from any path, raise the last exception
        if data is None:
            # If all paths fail, try pickle as fallback for backward compatibility
            print(
                f"Warning: Could not load model with specified method: {last_exception}"
            )
            print("Attempting to load with pickle as fallback...")

            for try_path in paths_to_try:
                if not os.path.exists(try_path):
                    continue

                try:
                    # Check if the file might be compressed
                    with open(try_path, "rb") as test_file:
                        file_start = test_file.read(6)

                    if file_start.startswith(b"\xfd7zXZ\x00"):
                        # LZMA compressed file
                        import lzma

                        with lzma.open(try_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        # Regular file
                        with open(try_path, "rb") as f:
                            data = pickle.load(f)
                    break
                except Exception as e:
                    last_exception = e
                    continue

            if data is None:
                raise ValueError(
                    f"Failed to load model from any of the candidate paths: {paths_to_try}. Last error: {last_exception}"
                )

        encoder = CharacterEncoder()
        encoder.cache = data.get("encoder_cache", {})

        # Handle different versions
        version = data.get("version", 1)

        if version >= 4:
            # Version 4+ uses the config dataclass
            config = data.get("config", None)
        else:
            # Older versions used individual parameters
            from charboundary.segmenters.types import SegmenterConfig

            config = SegmenterConfig(
                left_window=data.get("left_window", 5),
                right_window=data.get("right_window", 5),
                abbreviations=data.get("abbreviations", DEFAULT_ABBREVIATIONS.copy()),
            )

        # Create the feature extractor
        feature_extractor = FeatureExtractor(
            encoder=encoder,
            abbreviations=config.abbreviations,
            use_numpy=config.use_numpy,
            cache_size=config.cache_size,
        )

        segmenter = segmenter_class(
            model=data["model"],
            encoder=encoder,
            feature_extractor=feature_extractor,
            config=config,
        )

        return segmenter
