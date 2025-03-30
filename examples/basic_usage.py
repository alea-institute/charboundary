#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import gzip
import json
import time

from charboundary import TextSegmenter


def demonstrate_basic_usage():
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Sample annotated text for training

    training_data = []
    with gzip.open("data/train.jsonl.gz", "rt", encoding="utf-8") as input_file:
        for i, line in enumerate(input_file):
            training_data.append(json.loads(line).get("text"))
            if i > 10000:
                break

    # Train the segmenter
    print("Training segmenter...")
    t0 = time.time()
    metrics = segmenter.train(
        data=training_data,
        model_params={"n_estimators": 512, "max_depth": 64},
        sample_rate=0.001,  # Increase sample rate to get better class balance
        left_window=9,  # Specify window sizes during training
        right_window=9
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - t0))
    
    # Display training metrics
    print(f"Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")
    
    # Example texts to segment
    examples = [
        "Hello world. This is a simple test.",
        
        "The cat sat on the mat. The dog barked loudly. Both animals were happy.",
        
        "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson at 2:30 p.m.",
        
        "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial segregation in public schools was unconstitutional. This landmark decision changed the course of U.S. history. The ruling was unanimous and delivered by Chief Justice Earl Warren."
    ]
    
    # Segment each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Original: {example}")
        
        # Get sentences
        sentences = segmenter.segment_to_sentences(example)
        print("\nSentences:")
        for j, sentence in enumerate(sentences):
            print(f"  {j+1}. {sentence}")
        
        # Get paragraphs
        paragraphs = segmenter.segment_to_paragraphs(example)
        print("\nParagraphs:")
        for j, paragraph in enumerate(paragraphs):
            print(f"  {j+1}. {paragraph}")
        
        print("-" * 50)


def main():
    """Run the example script."""
    print("CharBoundary Library Example\n")
    demonstrate_basic_usage()


if __name__ == "__main__":
    main()
