#!/usr/bin/env python
"""
Simple test script for using ONNX models via the main package functions.
This script specifically tests setting optimization levels for the models.
"""

import sys
import time
import os
from charboundary import (
    get_small_onnx_segmenter,
    get_medium_onnx_segmenter,
    get_large_onnx_segmenter
)

# Sample text for testing
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

def test_model(model_name):
    """Test a specific ONNX model."""
    print(f"\nTesting {model_name} ONNX model:")
    
    # Set optimization level based on model name
    if model_name == "small":
        opt_level = 2  # Small model should use level 2
        get_segmenter = get_small_onnx_segmenter
    elif model_name == "medium":
        opt_level = 2  # Medium model should use level 2
        get_segmenter = get_medium_onnx_segmenter
    elif model_name == "large":
        opt_level = 3  # Large model should use level 3
        get_segmenter = get_large_onnx_segmenter
    else:
        print(f"Unknown model name: {model_name}")
        return
    
    try:
        # Load the model
        segmenter = get_segmenter()
        print(f"✓ Model successfully loaded")
        
        # Check if the model has ONNX enabled
        onnx_enabled = hasattr(segmenter.model, "use_onnx") and segmenter.model.use_onnx
        print(f"✓ ONNX enabled: {onnx_enabled}")
        
        # Try to get current optimization level
        try:
            current_level = getattr(segmenter.model, "onnx_optimization_level", "unknown")
            print(f"✓ Current optimization level: {current_level}")
        except Exception as e:
            print(f"✗ Error getting optimization level: {str(e)}")
        
        # Try to set optimization level directly (this may fail due to attribute errors)
        try:
            segmenter.model.onnx_optimization_level = opt_level
            print(f"✓ Set optimization level attribute to {opt_level}")
        except Exception as e:
            print(f"✗ Error setting optimization level attribute: {str(e)}")
        
        # Try to enable ONNX with optimization level (this may fail due to attribute errors)
        try:
            segmenter.model.enable_onnx(True, optimization_level=opt_level)
            print(f"✓ Called enable_onnx with optimization level {opt_level}")
        except Exception as e:
            print(f"✗ Error calling enable_onnx: {str(e)}")
        
        # Manually load the specific ONNX file
        try:
            # Get the path to the ONNX model
            onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "charboundary", "resources", "onnx")
            onnx_file = os.path.join(onnx_dir, f"{model_name}_model.onnx")
            
            if os.path.exists(onnx_file):
                print(f"✓ Found ONNX model at {onnx_file}")
                
                try:
                    # Load the ONNX model
                    segmenter.load_onnx(onnx_file)
                    print(f"✓ Loaded ONNX model from file")
                except Exception as e:
                    print(f"✗ Error loading ONNX model from file: {str(e)}")
            else:
                print(f"✗ ONNX model file not found at {onnx_file}")
        except Exception as e:
            print(f"✗ Error accessing ONNX file: {str(e)}")
        
        # Benchmark segmentation
        start_time = time.time()
        sentences = segmenter.segment_to_sentences(TEST_TEXT)
        end_time = time.time()
        
        # Print results
        print(f"✓ Segmentation successful")
        print(f"✓ Found {len(sentences)} sentences")
        print(f"✓ Processing time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"✓ Sample sentence: {sentences[0][:60]}...")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Test all ONNX models."""
    print("Testing ONNX models via charboundary package functions")
    print("=====================================================")
    
    # Test each model
    test_model("small")
    test_model("medium")
    test_model("large")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()