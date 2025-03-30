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

# Get list of sentences
sentences = segmenter.segment_to_sentences(text)
print(sentences)
# Output: ["Hello, world!", "This is a test.", "This is another sentence."]

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
# Get help
charboundary --help

# Analyze text using the default model
charboundary analyze --text "This is a test. This is another sentence." --output sentences

# Train a custom model
charboundary train --input training_data.jsonl --output model.skops

# Find the best model parameters
charboundary best-model --input training_data.jsonl --test test_data.jsonl
```

The CLI can be installed using either `pip` or `pipx`:

```bash
# Install globally as an application
pipx install charboundary

# Or use within a project
pip install charboundary
```

## License

MIT