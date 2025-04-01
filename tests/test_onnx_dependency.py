"""Test for ONNX dependency detection."""

import pytest

def test_onnx_availability():
    """Test if ONNX support is correctly detected."""
    # Try to import ONNX support
    try:
        from charboundary.onnx_support import check_onnx_available
        is_available = check_onnx_available()
        
        # This test should pass regardless of whether ONNX is available
        # It just verifies that we can detect ONNX availability without errors
        assert isinstance(is_available, bool)
        
        # If ONNX is available, the function should return True
        if is_available:
            # Verify we can import the required ONNX modules
            import onnx
            import skl2onnx
            assert onnx is not None
            assert skl2onnx is not None
    
    except ImportError:
        # If the module can't be imported, mark the test as skipped
        pytest.skip("ONNX support not available")


def test_onnx_import_in_models():
    """Test ONNX import handling in the models module."""
    # This will work whether or not ONNX is installed
    from charboundary.models import ONNX_AVAILABLE
    
    # ONNX_AVAILABLE should be a boolean flag
    assert isinstance(ONNX_AVAILABLE, bool)
    
    # If ONNX is available, verify the functions are imported
    if ONNX_AVAILABLE:
        from charboundary.models import (
            convert_to_onnx,
            save_onnx_model,
            load_onnx_model,
            create_onnx_inference_session
        )
        assert callable(convert_to_onnx)
        assert callable(save_onnx_model)
        assert callable(load_onnx_model)
        assert callable(create_onnx_inference_session)