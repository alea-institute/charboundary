"""
Example demonstrating how to train, save, and load models using skops serialization.

This example shows how to:
1. Create a TextSegmenter
2. Train the model on sample data
3. Save the model using skops serialization
4. Load the model back
5. Use the loaded model for text segmentation
"""

import os
import tempfile

from charboundary.segmenters import TextSegmenter, SegmenterConfig


def main():
    # Import the sentence and paragraph tags from constants
    from charboundary.constants import SENTENCE_TAG, PARAGRAPH_TAG
    
    # Sample training data with sentence and paragraph tags
    # Note: The correct format is to place tags AFTER the punctuation
    training_data = [
        f"This is a sentence.{SENTENCE_TAG} This is another sentence.{SENTENCE_TAG}{PARAGRAPH_TAG}",
        f"This is a new paragraph.{SENTENCE_TAG} It contains multiple sentences.{SENTENCE_TAG} Three to be exact.{SENTENCE_TAG}{PARAGRAPH_TAG}",
        f"Short sentences work too.{SENTENCE_TAG} Even very short ones.{SENTENCE_TAG}{PARAGRAPH_TAG}",
        f"Sentences with question marks?{SENTENCE_TAG} And exclamation marks!{SENTENCE_TAG} Work as expected.{SENTENCE_TAG}{PARAGRAPH_TAG}"
    ]

    # Create a segmenter with a small model for demonstration purposes
    config = SegmenterConfig(
        left_window=3,
        right_window=3,
        model_type="random_forest",
        model_params={
            "n_estimators": 20,  # Small model for faster execution
            "max_depth": 10
        }
    )
    
    segmenter = TextSegmenter(config=config)
    
    # Train the model
    print("Training model...")
    metrics = segmenter.train(data=training_data)
    
    # Print training metrics
    print(f"Training metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    # Create a temporary file for saving the model
    with tempfile.NamedTemporaryFile(suffix='.skops', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        # Save the model using skops
        print(f"Saving model to {model_path}...")
        segmenter.save(model_path, format="skops")
        print("Model saved successfully")
        
        # Load the model back with skops
        print("Loading model...")
        # Note: Using trust_model=True is necessary for our custom types
        # In production, only use trust_model=True if you trust the source of the model file
        loaded_segmenter = TextSegmenter.load(model_path, use_skops=True, trust_model=True)
        print("Model loaded successfully")
        
        # Test the loaded model
        test_text = "Can the loaded model segment this text? Yes, it can. Here's another sentence."
        print("\nOriginal text:")
        print(test_text)
        
        # Segment the text
        segmented_text = loaded_segmenter.segment_text(test_text)
        print("\nSegmented text with annotations:")
        print(segmented_text)
        
        # Extract sentences
        sentences = loaded_segmenter.segment_to_sentences(test_text)
        print("\nExtracted sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"{i}. {sentence}")
            
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"\nTemporary model file removed: {model_path}")


if __name__ == "__main__":
    main()