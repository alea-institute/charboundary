#!/usr/bin/env python3
"""
Profiling script for the CharBoundary library.
Uses cProfile to identify the top 25 methods that take the most time.
"""

import cProfile
import gzip
import json
import os
import pstats
import io
import time
import argparse
from pathlib import Path

from charboundary import TextSegmenter


def profile_training(num_samples=1000, sample_rate=0.001, n_estimators=100, max_depth=16):
    """Profile the training process."""
    print(f"Profiling training with {num_samples} samples...")
    
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Load sample data for training
    training_data = []
    data_path = Path("data/train.jsonl.gz")
    
    # Check if the data file exists
    if not data_path.exists():
        print(f"Data file {data_path} not found. Creating synthetic training data...")
        # Create synthetic training data for demonstration
        training_data = [
            "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
            "Here's a new paragraph.<|sentence|> It has multiple sentences.<|sentence|> And another one.<|sentence|><|paragraph|>",
            "Dr. Smith visited Washington D.C. last week.<|sentence|> He met with Prof. Johnson.<|sentence|><|paragraph|>",
            "The cat sat on the mat.<|sentence|> The dog barked loudly.<|sentence|> Both animals were happy.<|sentence|><|paragraph|>"
        ] * 50  # Duplicate to get more samples
        num_samples = min(num_samples, len(training_data))
    else:
        # Load from the actual data file
        with gzip.open(data_path, "rt", encoding="utf-8") as input_file:
            for i, line in enumerate(input_file):
                training_data.append(json.loads(line).get("text"))
                if i >= num_samples:
                    break
    
    # Profile the training process
    pr = cProfile.Profile()
    pr.enable()
    
    # Train the segmenter
    t0 = time.time()
    metrics = segmenter.train(
        data=training_data,
        model_params={"n_estimators": n_estimators, "max_depth": max_depth},
        sample_rate=sample_rate,
        left_window=9,
        right_window=9
    )
    
    pr.disable()
    elapsed = time.time() - t0
    
    # Print training metrics
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")
    
    return pr


def profile_inference(segmenter, num_iterations=100):
    """Profile the inference process."""
    print(f"Profiling inference with {num_iterations} iterations...")
    
    # Example texts to segment
    examples = [
        "Hello world. This is a simple test.",
        
        "The cat sat on the mat. The dog barked loudly. Both animals were happy.",
        
        "Dr. Smith visited Washington D.C. last week. He met with Prof. Johnson at 2:30 p.m.",
        
        "The court held in Brown v. Board of Education, 347 U.S. 483 (1954) that racial segregation in public schools was unconstitutional. This landmark decision changed the course of U.S. history. The ruling was unanimous and delivered by Chief Justice Earl Warren."
    ]
    
    # Test the segmenter on one example first to verify it works correctly
    try:
        print("Testing segmentation on one example first...")
        test_sentences = segmenter.segment_to_sentences(examples[0])
        test_paragraphs = segmenter.segment_to_paragraphs(examples[0])
        print(f"  Successfully segmented into {len(test_sentences)} sentences and {len(test_paragraphs)} paragraphs")
    except Exception as e:
        print(f"Error during test segmentation: {e}")
        # Create a fixed segmenter for profiling
        print("Creating a simplified version for profiling")
        
        # Create simpler examples for profiling if the real ones are causing errors
        examples = [
            "Hello world. This is a simple test.",
            "The cat sat on the mat. The dog barked loudly."
        ]
    
    # Profile the inference process
    pr = cProfile.Profile()
    pr.enable()
    
    t0 = time.time()
    try:
        for _ in range(num_iterations):
            for example in examples:
                # Get sentences
                sentences = segmenter.segment_to_sentences(example)
                
                # Get paragraphs
                paragraphs = segmenter.segment_to_paragraphs(example)
    except Exception as e:
        print(f"Error during profiling: {e}")
    
    pr.disable()
    elapsed = time.time() - t0
    
    print(f"Inference completed in {elapsed:.2f} seconds.")
    
    return pr


def profile_load_model(model_path):
    """Profile the model loading process."""
    print(f"Profiling model loading from {model_path}...")
    
    # Profile the model loading process
    pr = cProfile.Profile()
    pr.enable()
    
    t0 = time.time()
    try:
        # Try to load with trust_model=True to handle untrusted types
        segmenter = TextSegmenter.load(model_path, trust_model=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new segmenter for demonstration purposes.")
        segmenter = TextSegmenter()
        # Train minimal model for testing
        segmenter.train(
            data=["Hello world. This is a test."],
            model_params={"n_estimators": 10, "max_depth": 4},
            sample_rate=1.0
        )
    
    pr.disable()
    elapsed = time.time() - t0
    
    print(f"Model loading/creation completed in {elapsed:.2f} seconds.")
    
    return pr, segmenter


def print_stats(pr, title, sort_key='cumtime', limit=25):
    """Print profiling statistics."""
    print(f"\n{'-' * 40}")
    print(f"{title} (sorted by {sort_key})")
    print(f"{'-' * 40}")
    
    # Redirect output to a string
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
    ps.print_stats(limit)
    
    # Print the top methods
    stats_str = s.getvalue()
    print(stats_str)
    
    return stats_str


def main():
    """Run the profiling script."""
    parser = argparse.ArgumentParser(description='Profile the CharBoundary library')
    parser.add_argument('--mode', choices=['all', 'train', 'inference', 'load'], 
                        default='all', help='Profiling mode')
    parser.add_argument('--samples', type=int, default=1000, 
                        help='Number of samples for training')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations for inference')
    parser.add_argument('--model', type=str, default='charboundary/resources/small_model.skops.xz', 
                        help='Path to the model file for loading')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save profile results')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    results = {}
    
    # Profile model loading
    if args.mode in ['all', 'load']:
        load_pr, segmenter = profile_load_model(args.model)
        load_stats = print_stats(load_pr, "Model Loading Profile", limit=25)
        results['load'] = load_stats
    else:
        # Create segmenter directly if not loading
        segmenter = TextSegmenter()
    
    # Profile training
    if args.mode in ['all', 'train']:
        train_pr = profile_training(num_samples=args.samples)
        train_stats = print_stats(train_pr, "Training Profile", limit=25)
        results['train'] = train_stats
    
    # Profile inference
    if args.mode in ['all', 'inference']:
        try:
            # Make sure we have a trained model for inference
            if not hasattr(segmenter, 'is_trained') or not segmenter.is_trained:
                print("Training a minimal model for inference profiling...")
                # Create synthetic training data for demonstration with all character classes
                # Use more examples to ensure all required classes are represented
                train_data = [
                    "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
                    "Here's a paragraph.<|sentence|> It has sentences.<|sentence|><|paragraph|>",
                    "Dr. Smith visited Washington D.C. last week.<|sentence|> He met Prof. Johnson.<|sentence|><|paragraph|>",
                    "The cat sat on the mat.<|sentence|> The dog barked loudly.<|sentence|> Both animals were happy.<|sentence|><|paragraph|>",
                    "Hello world! This is a test.<|sentence|> This is another test.<|sentence|><|paragraph|>",
                    "This has a list: (1) First item;<|sentence|> (2) Second item;<|sentence|> and (3) Third item.<|sentence|><|paragraph|>"
                ] * 5  # Duplicate data to ensure sufficient samples
                
                segmenter = TextSegmenter()
                segmenter.train(
                    data=train_data,
                    model_params={"n_estimators": 32, "max_depth": 6},
                    sample_rate=1.0,
                    left_window=5,
                    right_window=5
                )
            
            inference_pr = profile_inference(segmenter, num_iterations=args.iterations)
            inference_stats = print_stats(inference_pr, "Inference Profile", limit=25)
            results['inference'] = inference_stats
        except Exception as e:
            print(f"Error during inference profiling: {e}")
            print("Skipping inference profiling.")
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for mode, stats in results.items():
                f.write(f"{'=' * 50}\n")
                f.write(f"{mode.upper()} PROFILE\n")
                f.write(f"{'=' * 50}\n")
                f.write(stats)
                f.write("\n\n")
        
        print(f"Profile results saved to {args.output}")


if __name__ == "__main__":
    main()