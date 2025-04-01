#!/usr/bin/env python
"""
Test script to compare prediction quality across different ONNX optimization levels.

This script tests whether different ONNX optimization levels maintain the same
prediction quality as the original scikit-learn model.
"""

import time
import numpy as np
from typing import List, Tuple, Dict

from charboundary import get_small_segmenter

# Sample complex text with various boundary challenges
TEST_TEXTS = [
    # Legal text with citations
    "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that "
    "racial segregation in public schools was unconstitutional. This landmark "
    "decision changed the course of U.S. history.",
    
    # Text with abbreviations
    "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson "
    "at 2:30 p.m. They discussed Mr. Brown's case regarding Section 1.2.3 of the tax code.",
    
    # Text with quotes
    "\"This is a direct quote,\" said the author. \"It contains multiple sentences. "
    "And it spans multiple lines.\" The audience nodded in agreement.",
    
    # Text with complex punctuation
    "Have you considered the following: (1) timing; (2) cost; and (3) feasibility? "
    "Each factor requires careful analysis. The report addresses all three points.",
    
    # Text with challenging sentence boundaries
    "Did he say \"Hello\"? I'm not sure. What about \"Goodbye\"? That I heard clearly.",
    
    # Text with numbers and decimals
    "The company grew by 5.6% in Q1 2023. Meanwhile, inflation rose to 3.2%, causing concerns."
]


def compare_predictions(optimizations: List[int]) -> Dict:
    """
    Compare predictions across different ONNX optimization levels.
    
    Args:
        optimizations: List of optimization levels to test (0-3)
        
    Returns:
        Dict: Comparison results
    """
    # Get the segmenter
    segmenter = get_small_segmenter()
    
    # Get scikit-learn predictions first (no ONNX)
    segmenter.config.use_onnx = False
    sklearn_predictions = []
    
    for text in TEST_TEXTS:
        sentences = segmenter.segment_to_sentences(text)
        sklearn_predictions.append(sentences)
    
    # Test each optimization level
    results = {'sklearn': sklearn_predictions}
    
    for level in optimizations:
        # Reset the segmenter (necessary to avoid cached predictions)
        segmenter = get_small_segmenter()
        
        # Configure ONNX with the specified optimization level
        segmenter.model.feature_count = 19  # Small model has 19 features
        segmenter.to_onnx()
        segmenter.model.enable_onnx(True, optimization_level=level)
        
        # Get predictions
        onnx_predictions = []
        for text in TEST_TEXTS:
            sentences = segmenter.segment_to_sentences(text)
            onnx_predictions.append(sentences)
        
        results[f'onnx_level_{level}'] = onnx_predictions
    
    return results


def analyze_results(results: Dict) -> Tuple[Dict, Dict]:
    """
    Analyze and compare predictions across optimization levels.
    
    Args:
        results: Dictionary with prediction results from different models
        
    Returns:
        Tuple[Dict, Dict]: Match status and detailed differences
    """
    # Check if predictions match for each text
    match_status = {}
    differences = {}
    
    # Use scikit-learn as reference
    reference = results['sklearn']
    
    for level_key in results:
        if level_key == 'sklearn':
            continue
        
        preds = results[level_key]
        matches = []
        level_differences = []
        
        for i, (ref_sentences, test_sentences) in enumerate(zip(reference, preds)):
            if ref_sentences == test_sentences:
                matches.append(True)
            else:
                matches.append(False)
                level_differences.append({
                    'text_index': i,
                    'text': TEST_TEXTS[i],
                    'sklearn': ref_sentences,
                    level_key: test_sentences
                })
        
        match_status[level_key] = {
            'all_match': all(matches),
            'match_count': sum(matches),
            'total_count': len(matches),
            'match_percentage': sum(matches) / len(matches) * 100 if matches else 0
        }
        
        differences[level_key] = level_differences
    
    return match_status, differences


def main():
    """Run comparison and display results."""
    print("ONNX Optimization Levels Quality Test\n")
    
    # Compare predictions with different optimization levels
    results = compare_predictions([0, 1, 2, 3])
    
    # Analyze results
    match_status, differences = analyze_results(results)
    
    # Print results
    print("Prediction Quality Results:\n")
    for level_key, status in match_status.items():
        print(f"{level_key}:")
        print(f"  All predictions match scikit-learn: {status['all_match']}")
        print(f"  Match count: {status['match_count']}/{status['total_count']} ({status['match_percentage']:.1f}%)")
        
        # Print differences if any
        level_diffs = differences[level_key]
        if level_diffs:
            print(f"  Differences found in {len(level_diffs)} test cases:")
            for diff in level_diffs:
                print(f"    Text {diff['text_index'] + 1}: ")
                print(f"      sklearn: {diff['sklearn']}")
                print(f"      {level_key}: {diff[level_key]}")
        print()


if __name__ == "__main__":
    main()