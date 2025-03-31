#!/usr/bin/env python3
"""
Script to test and benchmark the optimized text segmentation functions.
"""

import time
import numpy as np
from charboundary.segmenters import TextSegmenter
from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG

def test_inference_performance():
    """Test and benchmark the inference performance of the optimized segmenter."""
    print("Creating a segmenter with minimal training data...")
    
    # Simple training data with a mix of sentence and paragraph boundaries
    train_data = [
        "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
        "Here is a paragraph.<|sentence|> It contains multiple sentences.<|sentence|> And another one.<|sentence|><|paragraph|>",
        "Dr. Smith visited Washington D.C. last week.<|sentence|> He met with Prof. Johnson.<|sentence|><|paragraph|>",
        "The cat sat on the mat.<|sentence|> The dog barked loudly.<|sentence|> Both animals were happy.<|sentence|><|paragraph|>"
    ] * 10
    
    # Create and train a segmenter
    segmenter = TextSegmenter()
    print("Training the segmenter...")
    metrics = segmenter.train(
        data=train_data,
        model_params={"n_estimators": 32, "max_depth": 8},
        sample_rate=1.0,
        left_window=5,
        right_window=5
    )
    
    # Print training metrics
    print("Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")
    
    # Example texts to segment
    examples = [
        "Hello world. This is a simple test.",
        "The cat sat on the mat. The dog barked loudly.",
        "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson."
    ]
    
    # Test segmentation
    print("\nTesting segmentation on examples:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example}")
        
        # Time the segment_text method
        t0 = time.time()
        segmented = segmenter.segment_text(example)
        t1 = time.time()
        
        print(f"segment_text time: {(t1-t0)*1000:.2f}ms")
        print(f"Segmented text: {segmented}")
        
        # Time the segment_to_sentences method
        t0 = time.time()
        sentences = segmenter.segment_to_sentences(example)
        t1 = time.time()
        
        print(f"segment_to_sentences time: {(t1-t0)*1000:.2f}ms")
        print(f"Sentences ({len(sentences)}):")
        for j, sentence in enumerate(sentences):
            print(f"  {j+1}. {sentence}")
        
        # Time the segment_to_paragraphs method
        t0 = time.time()
        paragraphs = segmenter.segment_to_paragraphs(example)
        t1 = time.time()
        
        print(f"segment_to_paragraphs time: {(t1-t0)*1000:.2f}ms")
        print(f"Paragraphs ({len(paragraphs)}):")
        for j, paragraph in enumerate(paragraphs):
            print(f"  {j+1}. {paragraph}")
    
    print("\nBenchmarking inference speed...")
    
    # Benchmark with the largest example repeated multiple times
    text = examples[2] * 20  # Repeat the text to make it larger
    
    # Time the optimized version
    iterations = 10
    total_time = 0
    
    print(f"Running {iterations} iterations on a text of length {len(text)}...")
    
    for _ in range(iterations):
        t0 = time.time()
        sentences = segmenter.segment_to_sentences(text)
        paragraphs = segmenter.segment_to_paragraphs(text)
        t1 = time.time()
        total_time += (t1 - t0)
    
    avg_time = total_time / iterations
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Characters per second: {len(text)/avg_time:.2f}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_inference_performance()