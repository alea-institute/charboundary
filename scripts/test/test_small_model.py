#!/usr/bin/env python3
"""
Script to test the small model for the CharBoundary library.
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


def test_small_model() -> Tuple[int, int]:
    """
    Test the small model for the CharBoundary library.
    
    Returns:
        Tuple[int, int]: (number of successful tests, number of failed tests)
    """
    print("CharBoundary Small Model Test")
    print("=============================")
    
    # Check if the model exists (regular or compressed)
    # Fix path to find models correctly after reorganization
    script_path = os.path.abspath(__file__)
    # From scripts/test/ to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    model_dir = os.path.join(project_root, "charboundary", "resources")
    model_paths = [
        os.path.join(model_dir, "small_model.skops"),
        os.path.join(model_dir, "small_model.skops.xz"),
        os.path.join(model_dir, "small_model.skops.lzma")
    ]
    
    found_model = False
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            found_model = True
            model_path = path
            print(f"Found small model at {path} ({os.path.getsize(path) / (1024*1024):.1f} MB)")
            
            # If it's compressed, mention it
            if path.endswith('.xz') or path.endswith('.lzma'):
                print("The model is compressed for smaller size.")
            break
    
    if not found_model:
        print(f"Error: Small model not found at any of the expected locations:")
        for path in model_paths:
            print(f"  - {path}")
        print("Please run scripts/train_small_model.py first to generate the model.")
        return 0, 1
    
    # Load the model
    print("\nLoading the small model...", end="")
    t0 = time.time()
    try:
        segmenter = TextSegmenter.load(model_path, trust_model=True)
        load_time = time.time() - t0
        print(f" done in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"\nError loading the small model: {e}")
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
    example_text = "This is a simple example. It demonstrates how to use the small model."
    print(f"  Input text: {example_text}")
    example_sentences = segmenter.segment_to_sentences(example_text)
    print("  Segmented sentences:")
    for i, sentence in enumerate(example_sentences):
        print(f"    {i+1}. {sentence}")
    
    # Add performance benchmark
    print("\nBenchmarking processing speed...")
    
    # Generate a large text sample by repeating the README sample
    sample_text = """
    Employee also specifically and forever releases the Acme Inc. (Company) and the Company Parties (except where and 
    to the extent that such a release is expressly prohibited or made void by law) from any claims based on unlawful 
    employment discrimination or harassment, including, but not limited to, the Federal Age Discrimination in 
    Employment Act (29 U.S.C. § 621 et. seq.). This release does not include Employee's right to indemnification, 
    and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement, his right to equity awards,
    or continued exercise, pursuant to the terms of any specific equity award (or similar) agreement between 
    Employee and the Company nor to Employee's right to benefits under any Company plan or program in which
    Employee participated and is due a benefit in accordance with the terms of the plan or program as of the Effective
    Date and ending at 11:59 p.m. Eastern Time on Sep. 15, 2013.
    """
    
    # Repeat the sample to create a larger text
    large_text = sample_text * 1000
    total_chars = len(large_text)
    
    # Measure processing time
    print(f"Processing a text with {total_chars:,} characters...")
    start_time = time.time()
    sentences = segmenter.segment_to_sentences(large_text)
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate statistics
    chars_per_second = int(total_chars / processing_time)
    avg_sentence_length = total_chars / len(sentences) if sentences else 0
    
    # Print performance results
    print("\nPerformance Results:")
    print(f"  Documents processed:      1")
    print(f"  Total characters:         {total_chars:,}")
    print(f"  Total sentences found:    {len(sentences):,}")
    print(f"  Processing time:          {processing_time:.2f} seconds")
    print(f"  Processing speed:         {chars_per_second:,} characters/second")
    print(f"  Average sentence length:  {avg_sentence_length:.1f} characters")
    
    return successes, failures


if __name__ == "__main__":
    successes, failures = test_small_model()
    sys.exit(1 if failures > 0 else 0)