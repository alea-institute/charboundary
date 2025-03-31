#!/usr/bin/env python3
"""
Example script to demonstrate feature selection in the CharBoundary library.
"""

import time
from charboundary import TextSegmenter

def demonstrate_feature_selection():
    """Demonstrate feature selection in CharBoundary."""
    print("CharBoundary Feature Selection Example\n")
    
    # Sample annotated text for training
    training_data = [
        "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
        "Here is a paragraph.<|sentence|> It has multiple sentences.<|sentence|> The third sentence ends here.<|sentence|><|paragraph|>",
        "Dr. Smith visited Washington D.C. last week.<|sentence|> He met with Prof. Johnson.<|sentence|><|paragraph|>",
        "The cat sat on the mat.<|sentence|> The dog barked loudly.<|sentence|> Both animals were happy.<|sentence|><|paragraph|>"
    ] * 5  # Duplicate for more samples
    
    # Train a model with feature selection
    print("Training model with feature selection...")
    segmenter_with_fs = TextSegmenter()
    t0 = time.time()
    metrics_with_fs = segmenter_with_fs.train(
        data=training_data,
        model_params={"n_estimators": 50, "max_depth": 10},
        sample_rate=1.0,  # Use all samples for this small dataset
        left_window=5,
        right_window=5,
        use_feature_selection=True,
        feature_selection_threshold=0.01,
        max_features=40  # Limit to top 40 features
    )
    training_time_with_fs = time.time() - t0
    
    # Train a standard model without feature selection
    print("\nTraining standard model without feature selection...")
    segmenter_standard = TextSegmenter()
    t0 = time.time()
    metrics_standard = segmenter_standard.train(
        data=training_data,
        model_params={"n_estimators": 50, "max_depth": 10},
        sample_rate=1.0,
        left_window=5,
        right_window=5
    )
    training_time_standard = time.time() - t0
    
    # Print model training metrics comparison
    print("\nTraining Metrics Comparison:")
    print(f"{'Metric':<20} | {'With Feature Selection':<20} | {'Standard Model':<20}")
    print(f"{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        with_fs = metrics_with_fs.get(metric, 0)
        standard = metrics_standard.get(metric, 0)
        print(f"{metric:<20} | {with_fs:<20.4f} | {standard:<20.4f}")
    
    print(f"\nTraining time (seconds):")
    print(f"With Feature Selection: {training_time_with_fs:.2f}")
    print(f"Standard Model: {training_time_standard:.2f}")
    
    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    
    # Example texts to segment
    examples = [
        "Hello world. This is a simple test.",
        "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson at 2:30 p.m.",
        "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial segregation in public schools was unconstitutional. This landmark decision changed the course of U.S. history."
    ]
    
    # Time the feature-selected model
    iterations = 50
    t0 = time.time()
    for _ in range(iterations):
        for example in examples:
            sentences = segmenter_with_fs.segment_to_sentences(example)
    fs_inference_time = (time.time() - t0) / iterations
    
    # Time the standard model
    t0 = time.time()
    for _ in range(iterations):
        for example in examples:
            sentences = segmenter_standard.segment_to_sentences(example)
    standard_inference_time = (time.time() - t0) / iterations
    
    # Print inference speed comparison
    print(f"\nInference time per batch (ms):")
    print(f"With Feature Selection: {fs_inference_time*1000:.2f}")
    print(f"Standard Model: {standard_inference_time*1000:.2f}")
    
    speedup = standard_inference_time / fs_inference_time if fs_inference_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    # Show segmentation results to verify equivalent output
    print("\nSegmentation Examples to Verify Results:")
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example}")
        
        fs_sentences = segmenter_with_fs.segment_to_sentences(example)
        print("\nFeature-Selected Model Sentences:")
        for j, sentence in enumerate(fs_sentences, 1):
            print(f"  {j}. {sentence}")
        
        std_sentences = segmenter_standard.segment_to_sentences(example)
        print("\nStandard Model Sentences:")
        for j, sentence in enumerate(std_sentences, 1):
            print(f"  {j}. {sentence}")


def main():
    """Run the example script."""
    demonstrate_feature_selection()


if __name__ == "__main__":
    main()