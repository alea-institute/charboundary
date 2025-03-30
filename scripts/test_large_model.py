#!/usr/bin/env python3
"""
Script to test the large model for the CharBoundary library.
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple

from charboundary import TextSegmenter


# Test cases with expected results
TEST_CASES = [
    {
        "name": "Simple sentences",
        "text": "Hello world. This is a simple test.",
        "expected_sentences": 2
    },
    {
        "name": "Quotes mid-sentence",
        "text": 'Creditors may also typically invoke these laws to void "constructive" fraudulent transfers.',
        "expected_sentences": 1
    },
    {
        "name": "Quotes at the end of sentence",
        "text": 'The lawyer exclaimed, "This case is closed!" Then he left the courtroom.',
        "expected_sentences": 2
    },
    {
        "name": "Legal abbreviations",
        "text": 'The case Brown v. Board of Education, 347 U.S. 483 (1954), was a landmark decision.',
        "expected_sentences": 1
    },
    {
        "name": "Enumerated list",
        "text": 'The contract requires: (1) timely payment; (2) quality deliverables; and (3) written notice of termination.',
        "expected_sentences": 1
    },
    {
        "name": "Mixed quotes",
        "text": 'The witness said, "He told me \'Stop immediately\' before he left." This was recorded in the transcript.',
        "expected_sentences": 2
    },
    {
        "name": "Wrapped quotes across lines",
        "text": """The defense argued that "the plaintiff's claims
are without merit" in their motion.""",
        "expected_sentences": 1
    },
    {
        "name": "Multiple sentences with periods",
        "text": "First sentence. Second sentence. Third sentence.",
        "expected_sentences": 3
    },
    {
        "name": "Semicolons",
        "text": "First point; second point; third point.",
        "expected_sentences": 1
    }
]


def test_large_model() -> Tuple[int, int]:
    """
    Test the large model for the CharBoundary library.
    
    Returns:
        Tuple[int, int]: (number of successful tests, number of failed tests)
    """
    print("CharBoundary Large Model Test")
    print("=============================")
    
    # Check if the model exists (regular or compressed)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(package_dir, "charboundary", "resources")
    model_paths = [
        os.path.join(model_dir, "large_model.skops"),
        os.path.join(model_dir, "large_model.skops.xz"),
        os.path.join(model_dir, "large_model.skops.lzma")
    ]
    
    found_model = False
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            found_model = True
            model_path = path
            print(f"Found large model at {path} ({os.path.getsize(path) / (1024*1024):.1f} MB)")
            
            # If it's compressed, mention it
            if path.endswith('.xz') or path.endswith('.lzma'):
                print("The model is compressed for smaller size.")
            break
    
    if not found_model:
        print(f"Error: Large model not found at any of the expected locations:")
        for path in model_paths:
            print(f"  - {path}")
        print("Please run scripts/train_large_model.py first to generate the model.")
        return 0, 1
    
    # Load the model
    print("\nLoading the large model...", end="")
    t0 = time.time()
    try:
        segmenter = TextSegmenter.load(model_path, trust_model=True)
        load_time = time.time() - t0
        print(f" done in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"\nError loading the large model: {e}")
        return 0, 1
    
    # Run the tests
    print("\nRunning tests on various examples...\n")
    
    successes = 0
    failures = 0
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"Test {i+1}: {test_case['name']}")
        print(f"  Text: {test_case['text']}")
        
        # Segment the text
        t0 = time.time()
        sentences = segmenter.segment_to_sentences(test_case['text'])
        segment_time = time.time() - t0
        
        # Print results
        print(f"  Found {len(sentences)} sentences (in {segment_time*1000:.1f} ms):")
        for j, sentence in enumerate(sentences):
            print(f"    {j+1}. {sentence}")
        
        # Check if result matches expected
        success = len(sentences) == test_case['expected_sentences']
        if success:
            print(f"  ✓ PASS: Found expected {test_case['expected_sentences']} sentences.")
            successes += 1
        else:
            print(f"  ✗ FAIL: Expected {test_case['expected_sentences']} sentences, got {len(sentences)}.")
            failures += 1
        
        print()  # Empty line between tests
    
    # Print summary
    total = successes + failures
    print(f"Test summary: {successes}/{total} tests passed ({successes/total*100:.1f}%)")
    
    # Print model information
    print("\nModel information:")
    config = segmenter.config
    print(f"  Window sizes: left={config.left_window}, right={config.right_window}")
    print(f"  Model type: {config.model_type}")
    print(f"  Model parameters: {config.model_params}")
    print(f"  Abbreviations configured: {len(segmenter.get_abbreviations())}")
    
    # Test a simple segmentation example to demonstrate usage
    print("\nUsage example:")
    example_text = "This is a simple example. It demonstrates how to use the large model."
    print(f"  Input text: {example_text}")
    example_sentences = segmenter.segment_to_sentences(example_text)
    print("  Segmented sentences:")
    for i, sentence in enumerate(example_sentences):
        print(f"    {i+1}. {sentence}")
    
    return successes, failures


if __name__ == "__main__":
    successes, failures = test_large_model()
    sys.exit(1 if failures > 0 else 0)