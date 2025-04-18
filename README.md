# CharBoundary

A modular library for segmenting text into sentences and paragraphs based on character-level features.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![Version](https://img.shields.io/badge/version-0.4.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- Character-level text segmentation
- Support for sentence and paragraph boundaries
- Character span extraction for precise text positions
- Customizable window sizes for context
- Support for abbreviations
- Highly optimized performance (up to 280,000 characters/second)
- Secure model serialization with skops
- Optional ONNX model conversion and inference

## Installation

```bash
pip install charboundary
```

Install with additional dependencies:

```bash
# With NumPy support for faster processing
pip install charboundary[numpy]

# With ONNX support for model conversion and faster inference
pip install charboundary[onnx]

# With all optional dependencies
pip install charboundary[numpy,onnx]
```

## Quick Start

### Using the Pre-trained Models

CharBoundary comes with pre-trained models of different sizes:

- **Small** - Small footprint (5 token context window, 32 trees) - Processes ~85,000 characters/second
- **Medium** - Default, best performance (7 token context window, 64 trees) - Processes ~280,000 characters/second
- **Large** - Most accurate (9 token context window, 256 trees) - Processes ~175,000 characters/second

> **Package Distribution:** To keep the package size reasonable:
> - **Included in the package:** Only the small model (~1MB) 
> - **Downloaded on demand:** Medium (~2MB) and large (~6MB) models are automatically downloaded from GitHub when first used

The download happens automatically and transparently the first time you use functions like `get_default_segmenter()` or `get_large_segmenter()` if the models aren't already available locally.

### ONNX Acceleration (Up to 2.1x Faster)

For maximum performance, install the ONNX dependencies and use the optimized models:

```bash
# Install ONNX support
pip install charboundary[onnx]
```

```python
# Use ONNX-accelerated model with recommended optimization level
from charboundary import get_large_onnx_segmenter

# Get optimized model (downloads automatically if needed)
segmenter = get_large_onnx_segmenter()

# Process text at maximum speed
sentences = segmenter.segment_to_sentences(your_text)
```

The ONNX acceleration provides up to 2.1x performance improvement with zero impact on result quality. ONNX models are now XZ-compressed by default (60% smaller files). See `examples/onnx_optimization_simple.py` for a complete benchmark example.

```python
from charboundary import get_default_segmenter

# Get the pre-trained medium-sized segmenter (default)
segmenter = get_default_segmenter()

# Segment text into sentences and paragraphs
text = """
Employee also specifically and forever releases the Acme Inc. (Company) and the Company Parties (except where and 
to the extent that such a release is expressly prohibited or made void by law) from any claims based on unlawful 
employment discrimination or harassment, including, but not limited to, the Federal Age Discrimination in 
Employment Act (29 U.S.C. § 621 et. seq.). This release does not include Employee’s right to indemnification, 
and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement, his right to equity awards,
or continued exercise, pursuant to the terms of any specific equity award (or similar) agreement between 
Employee and the Company nor to Employee’s right to benefits under any Company plan or program in which
Employee participated and is due a benefit in accordance with the terms of the plan or program as of the Effective
Date and ending at 11:59 p.m. Eastern Time on Sep. 15, 2013.
"""
# Get list of sentences (with default threshold)
sentences = segmenter.segment_to_sentences(text)
print("\n-\n".join(sentences))
```

Output
```text
Employee also specifically and forever releases the Acme Inc. (Company) and the Company Parties (except where and
to the extent that such a release is expressly prohibited or made void by law) from any claims based on unlawful
employment discrimination or harassment, including, but not limited to, the Federal Age Discrimination in
Employment Act (29 U.S.C. § 621 et. seq.).
-
This release does not include Employee’s right to indemnification,
and related insurance coverage, under Sec. 7.1.4 or Ex. 1-1 of the Employment Agreement, his right to equity awards,
or continued exercise, pursuant to the terms of any specific equity award (or similar) agreement between
Employee and the Company nor to Employee’s right to benefits under any Company plan or program in which
Employee participated and is due a benefit in accordance with the terms of the plan or program as of the Effective
Date and ending at 11:59 p.m. Eastern Time on Sep. 15, 2013.
```

```
# Control segmentation sensitivity with threshold parameter

# Lower threshold = more aggressive segmentation (higher recall)

high_recall_sentences = segmenter.segment_to_sentences(text, threshold=0.1)

# Higher threshold = conservative segmentation (higher precision)
high_precision_sentences = segmenter.segment_to_sentences(text, threshold=0.9)`

print(len(high_recall_sentences), len(high_precision_sentences))
# Output: 2 1
```

You can also choose a specific model size based on your needs:

```python
from charboundary import get_small_segmenter, get_large_segmenter

# For faster processing with smaller memory footprint
small_segmenter = get_small_segmenter()

# For highest accuracy (but larger memory footprint)
large_segmenter = get_large_segmenter()
```

The models are optimized for handling:

- Quotation marks in the middle or at the end of sentences
- Common abbreviations (including legal abbreviations)
- Legal citations (e.g., "Brown v. Board of Education, 347 U.S. 483 (1954)")
- Multi-line quotes
- Enumerated lists (partial support)

### Getting Character Spans

CharBoundary can provide exact character spans (start and end positions) for each segment:

```python
from charboundary import get_default_segmenter

segmenter = get_default_segmenter()
text = "This is a sample text. It has multiple sentences. Here's the third one."

# Get sentences with their character spans
sentences_with_spans = segmenter.segment_to_sentences_with_spans(text)
for sentence, (start, end) in sentences_with_spans:
    print(f"Span ({start}-{end}): {sentence}")
    # Verify the span points to the correct text
    assert text[start:end].strip() == sentence.strip()

# Get only the spans without the text
spans = segmenter.get_sentence_spans(text)
print(f"Spans: {spans}")  # [(0, 22), (22, 46), (46, 67)]

# The spans cover EVERY character in the input
assert sum(end - start for start, end in spans) == len(text)

# Do the same for paragraphs
paragraph_spans = segmenter.get_paragraph_spans(text)
```

See the complete example in `examples/character_spans_example.py`.

### Training Your Own Model

```python
from charboundary import TextSegmenter

# Create a segmenter (will be initialized with default parameters)
segmenter = TextSegmenter()

# Train the model on sample data
training_data = [
    "This is a sentence.<|sentence|> This is another sentence.<|sentence|><|paragraph|>",
    "This is a new paragraph.<|sentence|> It has multiple sentences.<|sentence|><|paragraph|>"
]
segmenter.train(data=training_data)

# Segment text into sentences and paragraphs
text = "Hello, world! This is a test. This is another sentence."
segmented_text = segmenter.segment_text(text)
print(segmented_text)
# Output: "Hello, world!<|sentence|> This is a test.<|sentence|> This is another sentence.<|sentence|>"

# Get list of sentences
sentences = segmenter.segment_to_sentences(text)
print(sentences)
# Output: ["Hello, world!", "This is a test.", "This is another sentence."]
```

## Model Serialization and Optimization

### Skops Serialization

CharBoundary uses [skops](https://github.com/skops-dev/skops) for secure model serialization. This provides better security than pickle for sharing and loading models.

#### Saving Models

```python
# Train a model
segmenter = TextSegmenter()
segmenter.train(data=training_data)

# Save the model with skops
segmenter.save("model.skops", format="skops")
```

#### Loading Models

```python
# Load a model with security checks (default)
# This will reject loading custom types for security
segmenter = TextSegmenter.load("model.skops", use_skops=True)

# Load a model with trusted types enabled 
# Only use this with models from trusted sources
segmenter = TextSegmenter.load("model.skops", use_skops=True, trust_model=True)
```

#### Security Considerations

- When loading models from untrusted sources, avoid setting `trust_model=True`
- When loading fails with untrusted types, skops will list the untrusted types that need to be approved
- The library will fall back to pickle if skops loading fails, but this is less secure

### ONNX Model Conversion and Usage

When the `onnx` optional dependency is installed, you can convert and use models in ONNX format for faster inference:

```python
from charboundary import get_default_segmenter, get_medium_onnx_segmenter
from charboundary.models import create_model
from charboundary.segmenters import SegmenterConfig

# Option 1: Get a segmenter with ONNX already enabled (downloads model if needed)
segmenter = get_medium_onnx_segmenter()
sentences = segmenter.segment_to_sentences(text)

# Option 2: Create a model with ONNX support enabled and optimization level
model = create_model(
    model_type="random_forest", 
    use_onnx=True,
    onnx_optimization_level=2  # Use extended optimizations (level 0-3)
)

# Option 3: Create a segmenter with ONNX configuration
segmenter = TextSegmenter(
    config=SegmenterConfig(
        use_onnx=True,
        onnx_optimization_level=2  # Extended optimizations
    )
)

# Option 4: Convert an existing model to ONNX
segmenter = get_default_segmenter()
segmenter.model.to_onnx()  # Converts the model to ONNX format

# Save the ONNX model to a file (now with XZ compression by default)
segmenter.model.save_onnx("model.onnx")  # Creates model.onnx.xz by default
segmenter.model.save_onnx("model.onnx", compress=False)  # No compression

# Load an ONNX model with specified optimization level (handles compressed files automatically)
new_segmenter = get_default_segmenter()
new_segmenter.model.load_onnx("model.onnx")  # Works with both model.onnx or model.onnx.xz
new_segmenter.model.enable_onnx(True, optimization_level=2)  # Enable ONNX with extended optimizations

# Run inference with the ONNX model
sentences = new_segmenter.segment_to_sentences(text)
```

All models (including ONNX versions) can be automatically downloaded from the GitHub repository if they're not available locally:

```python
# These functions download models from GitHub if not found locally
from charboundary import (
    get_small_onnx_segmenter,    # Small model with ONNX (~5MB, included in package)
    get_medium_onnx_segmenter,   # Medium model with ONNX (~33MB, downloaded on demand)
    get_large_onnx_segmenter     # Large model with ONNX (~188MB, downloaded on demand)
)

# Explicitly download an ONNX model
from charboundary import download_onnx_model
download_onnx_model("large", force=True)  # Force redownload even if exists
```

> **ONNX Package Distribution:** To keep the package size reasonable:
> - **Included in the package:** Only the small ONNX model (~2MB, XZ compressed)
> - **Downloaded on demand:** Medium (~13MB) and large (~75MB) XZ compressed ONNX models

See the `examples/remote_onnx_example.py` file for a complete example of remote model usage.

#### ONNX Performance Benefits

ONNX models can provide significant performance improvements, especially for inference:

- Faster prediction speeds (up to 2-5x speedup)
- Better hardware acceleration support
- Easier deployment in production environments
- Smaller memory footprint

See the `examples/onnx_model_example.py` file for a complete example of ONNX usage.

#### ONNX Utilities

The package includes a comprehensive utility script for working with ONNX models:

- `scripts/onnx_utils.py`: A unified tool for converting, benchmarking, and testing ONNX models
  ```bash
  # Convert all built-in models to ONNX with optimal optimization levels
  python scripts/onnx_utils.py convert --all-models
  
  # Convert a specific built-in model to ONNX with a custom optimization level
  python scripts/onnx_utils.py convert --model-name small --optimization-level 2
  
  # Convert a custom model file to ONNX
  python scripts/onnx_utils.py convert --input-file path/to/model.skops.xz --output-file path/to/model.onnx
  
  # Benchmark all built-in models with optimal optimization levels
  python scripts/onnx_utils.py benchmark --all-models
  
  # Benchmark a specific built-in model with all optimization levels
  python scripts/onnx_utils.py benchmark --model-name medium
  
  # Test ONNX model functionality and accuracy
  python scripts/onnx_utils.py test --all-models
  ```
  
See [scripts/SCRIPT_USAGE.md](scripts/SCRIPT_USAGE.md) for more detailed examples and output logs of each command.

These scripts are particularly useful for converting pre-trained models to ONNX format and evaluating the performance benefits.

#### ONNX Optimization Levels

CharBoundary supports ONNX optimization levels to further improve performance without any impact on prediction quality:

| Level | Description | Speed Boost | Recommended For |
|-------|-------------|-------------|-----------------|
| 0 | No optimization | Baseline | Debugging only |
| 1 | Basic optimizations | 5-10% | Memory-constrained environments |
| 2 | Extended optimizations | 10-20% | **Most use cases (recommended)** |
| 3 | All optimizations | 15-25% | Large models, high-performance computing |

**Key Benefits:**
- Same prediction results as scikit-learn across all levels
- Easy to configure with a single parameter
- Safe to use in production (Level 2 recommended)

**How to use optimization levels:**

```python
# Method 1: When creating a model
from charboundary.models import create_model

model = create_model(
    model_type="random_forest", 
    use_onnx=True,
    onnx_optimization_level=2  # Extended optimizations recommended
)

# Method 2: When creating a segmenter
from charboundary import TextSegmenter
from charboundary.segmenters import SegmenterConfig

segmenter = TextSegmenter(
    config=SegmenterConfig(
        use_onnx=True,
        onnx_optimization_level=2  # Extended optimizations recommended
    )
)

# Method 3: When getting a pre-built model
from charboundary import get_medium_onnx_segmenter

segmenter = get_medium_onnx_segmenter()
segmenter.model.enable_onnx(True, optimization_level=2)

# Method 4: Setting it on an existing model
segmenter.model.enable_onnx(True, optimization_level=2)
```

The large model benefits the most from optimization level 3, while small and medium models perform best with level 2.

For a complete runnable example that demonstrates all optimization levels, see:
```
examples/onnx_optimization_simple.py
```

#### ONNX Performance Results

The ONNX conversion with optimization levels provides significant performance improvements:

| Model | scikit-learn<br>(chars/sec) | ONNX Level 1<br>(chars/sec) | ONNX Level 2/3<br>(chars/sec) | Total<br>Speedup |
|-------|-----------------------------:|-----------------------------:|-------------------------------:|------------------:|
| Small | 544,678                      | 572,176                      | 600,853 (Level 2)             | **1.10x**         |
| Medium| 293,897                      | 345,157                      | 379,672 (Level 2)             | **1.29x**         |
| Large | 179,998                      | 301,539                      | 376,923 (Level 3)             | **2.09x**         |

Key observations:
- Larger models benefit more from optimization (up to 2.09x faster)
- Level 2 is optimal for small/medium models
- Level 3 provides best performance for large models 
- All optimization levels preserve exact prediction quality

These performance improvements come with zero impact on accuracy or prediction quality - the results remain identical to the original scikit-learn models.

## Configuration

### Basic Configuration

You can customize the segmenter with various parameters:

```python
from charboundary.segmenters import TextSegmenter, SegmenterConfig

config = SegmenterConfig(
    # Model configuration
    model_type="random_forest",  # Type of model to use
    model_params={               # Parameters for the model
        "n_estimators": 100,
        "max_depth": 16,
        "class_weight": "balanced"
    },
    threshold=0.5,               # Classification threshold (0.0-1.0)
                                 # Lower=more sentences (recall), Higher=fewer sentences (precision)
    
    # Window configuration
    left_window=3,               # Size of left context window
    right_window=3,              # Size of right context window
    
    # Domain knowledge
    abbreviations=["Dr.", "Mr.", "Mrs.", "Ms."],  # Custom abbreviations
    
    # Performance settings
    use_numpy=True,              # NumPy for faster processing
    cache_size=1024,             # Cache size for character encoding
    num_workers=4,               # Number of worker processes
    
    # ONNX acceleration (2-5x faster inference)
    use_onnx=True,               # Enable ONNX inference if available
    onnx_optimization_level=2    # Optimization level (recommended=2):
                                 #   0=None/debug, 1=Basic, 2=Extended, 3=Maximum
)

segmenter = TextSegmenter(config=config)
```

### Advanced Features

The CharBoundary library includes sophisticated feature engineering tailored for text segmentation. These features help the model distinguish between actual sentence boundaries and other characters that may appear similar (like periods in abbreviations or quotes in the middle of sentences).

Key features include:

1. **Quotation Handling**:
   - Quote balance tracking (detecting matched pairs of quotes)
   - Word completion detection for quotes
   - Multi-line quote recognition

2. **List and Enumeration Detection**:
   - Recognition of enumerated list items (`(1)`, `(2)`, `(a)`, `(b)`, etc.)
   - Detection of list introductions (colons, phrases like "as follows:")
   - Special handling for semicolons in list structures

3. **Abbreviation Detection**:
   - Comprehensive lists of common and domain-specific abbreviations
   - Legal abbreviations and citations

4. **Contextual Analysis**:
   - Distinction between primary terminators (`.`, `!`, `?`) and secondary terminators (`"`, `'`, `:`, `;`)
   - Detection of lowercase letters following potential terminators
   - Analysis of surrounding context for sentence boundaries

These features enable the model to make intelligent decisions about text segmentation, particularly for complex cases like legal documents, technical texts, and documents with complex structure.

## Working with Abbreviations

```python
# Get current abbreviations
abbrevs = segmenter.get_abbreviations()

# Add new abbreviations
segmenter.add_abbreviation("Ph.D")

# Remove abbreviations
segmenter.remove_abbreviation("Dr.")

# Set a new list of abbreviations
segmenter.set_abbreviations(["Dr.", "Mr.", "Prof.", "Ph.D."])
```

## Command-Line Interface

CharBoundary provides a command-line interface for common operations:

```bash
# Get help for all commands
charboundary --help

# Get help for a specific command
charboundary analyze --help
charboundary train --help
charboundary best-model --help
```

### Analyze Command

Process text using a trained model:

```bash
# Analyze with default annotated output
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt

# Output sentences (one per line)
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --format sentences

# Output paragraphs
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --format paragraphs

# Save output to a file and generate metrics
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --output segmented.txt --metrics metrics.json

# Adjust segmentation sensitivity using threshold
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --threshold 0.3  # More sensitive (higher recall)
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --threshold 0.5  # Default balance
charboundary analyze --model charboundary/resources/small_model.skops.xz --input input.txt --threshold 0.8  # More conservative (higher precision)
```

#### Threshold Calibration Example

The threshold parameter lets you control the trade-off between Type I errors (false positives) and Type II errors (false negatives):

```bash
# Create a test file
echo "The plaintiff, Mr. Brown vs. Johnson Corp., argued that patent no. 12345 was infringed. Dr. Smith provided expert testimony on Feb. 2nd." > legal_text.txt

# Low threshold (0.2) - High recall, more boundaries detected
charboundary analyze --model charboundary/resources/small_model.skops.xz --input legal_text.txt --format sentences --threshold 0.2
```

Output with low threshold (0.2):
```
The plaintiff, Mr.
Brown vs.
Johnson Corp.
, argued that patent no.
12345 was infringed.
Dr.
Smith provided expert testimony on Feb.
2nd.
```

```bash
# Default threshold (0.5) - Balanced approach
charboundary analyze --model charboundary/resources/small_model.skops.xz --input legal_text.txt --format sentences --threshold 0.5
```

Output with default threshold (0.5):
```
The plaintiff, Mr. Brown vs. Johnson Corp., argued that patent no. 12345 was infringed.
Dr. Smith provided expert testimony on Feb.
2nd.
```

```bash
# High threshold (0.8) - High precision, only confident boundaries
charboundary analyze --model charboundary/resources/small_model.skops.xz --input legal_text.txt --format sentences --threshold 0.8
```

Output with high threshold (0.8):
```
The plaintiff, Mr. Brown vs. Johnson Corp., argued that patent no. 12345 was infringed.
Dr. Smith provided expert testimony on Feb. 2nd.
```

### Train Command

Train a custom model on annotated data:

```bash
# Train with default parameters
charboundary train --data training_data.txt --output model.skops

# Train with custom parameters
charboundary train --data training_data.txt --output model.skops \
  --left-window 4 --right-window 6 --n-estimators 100 --max-depth 16 \
  --sample-rate 0.1 --max-samples 10000 --threshold 0.5 --metrics-file train_metrics.json

# Train with feature selection to improve performance
charboundary train --data training_data.txt --output model.skops \
  --use-feature-selection --feature-selection-threshold 0.01 --max-features 50
```

Training data should contain annotated text with `<|sentence|>` and `<|paragraph|>` markers.

#### Feature Selection

The library supports automatic feature selection during training, which can improve both accuracy and inference speed:

- **Basic Feature Selection**: Use `--use-feature-selection` to enable automatic feature selection
- **Threshold Selection**: Set importance threshold with `--feature-selection-threshold` (default: 0.01)
- **Maximum Features**: Limit the number of features with `--max-features`

Feature selection works in two stages:
1. First, it trains an initial model to identify feature importance
2. Then, it filters out less important features and retrains using only the selected features

This can significantly reduce model complexity while maintaining or even improving accuracy, especially for deployment on resource-constrained environments.

### Best-Model Command

Find the best model parameters by training multiple models:

```bash
# Find best model with default parameter ranges
charboundary best-model --data training_data.txt --output best_model.skops

# Customize parameter search space
charboundary best-model --data training_data.txt --output best_model.skops \
  --left-window-values 3 5 7 --right-window-values 3 5 7 \
  --n-estimators-values 50 100 200 --max-depth-values 8 16 24 \
  --threshold-values 0.3 0.5 0.7 --sample-rate 0.1 --max-samples 10000

# Use validation data for model selection
charboundary best-model --data training_data.txt --output best_model.skops \
  --validation validation_data.txt --metrics-file best_metrics.json
```

The CLI can be installed using either `pip` or `pipx`:

```bash
# Install globally as an application
pipx install charboundary

# Or use within a project
pip install charboundary
```

## Development Tools

### Profiling Performance

The library includes a profiling script to identify performance bottlenecks:

```bash
# Profile all operations (training, inference, model loading)
python scripts/benchmark/profile_model.py --mode all

# Profile just the training process
python scripts/benchmark/profile_model.py --mode train --samples 500

# Profile just the inference process
python scripts/benchmark/profile_model.py --mode inference --iterations 200

# Profile model loading
python scripts/benchmark/profile_model.py --mode load --model charboundary/resources/medium_model.skops.xz

# Save profiling results to a file
python scripts/benchmark/profile_model.py --output profile_results.txt
```

## Performance

The library is highly optimized for performance while maintaining accuracy. The medium model offers the best balance of speed and accuracy:

```
Performance Results:
  Documents processed:      1
  Total characters:         991,000
  Total sentences found:    2,000
  Processing time:          3.53 seconds
  Processing speed:         280,000 characters/second
  Average sentence length:  495.5 characters
```

Key optimizations:
- Direct file path handling for resource loading
- Frozensets for character testing
- Pre-allocated arrays with proper types
- Character n-gram caching
- Pattern hash for common text patterns
- Prediction caching for repeated segments

You can run the benchmark tests yourself with the included scripts:
```bash
python scripts/test/test_small_model.py
python scripts/test/test_medium_model.py
python scripts/test/test_large_model.py
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the project history and version details.

## License

MIT