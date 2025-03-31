# CharBoundary

A modular library for segmenting text into sentences and paragraphs based on character-level features.

## Features

- Character-level text segmentation
- Support for sentence and paragraph boundaries
- Customizable window sizes for context
- Support for abbreviations
- Optimized for both accuracy and performance
- Secure model serialization with skops

## Installation

```bash
pip install charboundary
```

Or install with NumPy support for faster processing:

```bash
pip install charboundary[numpy]
```

## Quick Start

### Using the Pre-trained Models

CharBoundary comes with pre-trained models of different sizes:

- **Small** - Fast with a small footprint (5 token context window, 64 trees)
- **Medium** - Default, balanced performance (7 token context window, 128 trees) 
- **Large** - Most accurate but larger and slower (9 token context window, 512 trees)

> **Note:** The small and medium models are included in the package distribution. The large model is not included by default to keep the package size reasonable, but can be downloaded separately from the project repository.

```python
from charboundary import get_default_segmenter

# Get the pre-trained medium-sized segmenter (default)
segmenter = get_default_segmenter()

# Segment text into sentences and paragraphs
text = "Hello, world! This is a test. This is another sentence."

# Get list of sentences (with default threshold)
sentences = segmenter.segment_to_sentences(text)
print(sentences)
# Output: ["Hello, world!", "This is a test.", "This is another sentence."]

# Control segmentation sensitivity with threshold parameter
# Lower threshold = more aggressive segmentation (higher recall)
high_recall_sentences = segmenter.segment_to_sentences(text, threshold=0.3)
# Default threshold = balanced approach
balanced_sentences = segmenter.segment_to_sentences(text, threshold=0.5)
# Higher threshold = conservative segmentation (higher precision)
high_precision_sentences = segmenter.segment_to_sentences(text, threshold=0.8)

# Get list of paragraphs
paragraphs = segmenter.segment_to_paragraphs(text)
print(paragraphs)
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

For example, the models correctly handle quotes in the middle of sentences:

```python
text = 'Creditors may also typically invoke these laws to void "constructive" fraudulent transfers.'
sentences = segmenter.segment_to_sentences(text)
# Output: ['Creditors may also typically invoke these laws to void "constructive" fraudulent transfers.']
```

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

## Model Serialization with skops

CharBoundary uses [skops](https://github.com/skops-dev/skops) for secure model serialization. This provides better security than pickle for sharing and loading models.

### Saving Models

```python
# Train a model
segmenter = TextSegmenter()
segmenter.train(data=training_data)

# Save the model with skops
segmenter.save("model.skops", format="skops")
```

### Loading Models

```python
# Load a model with security checks (default)
# This will reject loading custom types for security
segmenter = TextSegmenter.load("model.skops", use_skops=True)

# Load a model with trusted types enabled 
# Only use this with models from trusted sources
segmenter = TextSegmenter.load("model.skops", use_skops=True, trust_model=True)
```

### Security Considerations

- When loading models from untrusted sources, avoid setting `trust_model=True`
- When loading fails with untrusted types, skops will list the untrusted types that need to be approved
- The library will fall back to pickle if skops loading fails, but this is less secure

## Configuration

### Basic Configuration

You can customize the segmenter with various parameters:

```python
from charboundary.segmenters import TextSegmenter, SegmenterConfig

config = SegmenterConfig(
    left_window=3,             # Size of left context window
    right_window=3,            # Size of right context window
    abbreviations=["Dr.", "Mr.", "Mrs.", "Ms."],  # Custom abbreviations
    model_type="random_forest",  # Type of model to use
    model_params={             # Parameters for the model
        "n_estimators": 100,
        "max_depth": 16,
        "class_weight": "balanced"
    },
    threshold=0.5,             # Probability threshold for classification (0.0-1.0)
                               # Lower values favor recall, higher values favor precision
    use_numpy=True,            # Use NumPy for faster processing
    cache_size=1024,           # Cache size for character encoding
    num_workers=4              # Number of worker processes
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
python scripts/profile_model.py --mode all

# Profile just the training process
python scripts/profile_model.py --mode train --samples 500

# Profile just the inference process
python scripts/profile_model.py --mode inference --iterations 200

# Profile model loading
python scripts/profile_model.py --mode load --model charboundary/resources/medium_model.skops.xz

# Save profiling results to a file
python scripts/profile_model.py --output profile_results.txt
```

## License

MIT