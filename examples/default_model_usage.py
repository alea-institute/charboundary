#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library with the default model.
"""

from charboundary import get_default_segmenter


def demonstrate_default_model_usage():
    """Demonstrate usage of the default pre-trained model."""
    
    # Get the default pre-trained segmenter
    segmenter = get_default_segmenter()
    
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
    print("CharBoundary Default Model Example\n")
    print("Using the pre-trained default model for text segmentation")
    demonstrate_default_model_usage()


if __name__ == "__main__":
    main()