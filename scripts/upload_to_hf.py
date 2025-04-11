#!/usr/bin/env python3
"""
Simple script to upload charboundary models to Hugging Face Hub.

This script uploads models (skops or ONNX) to Hugging Face Hub
with appropriate README files.

Requirements:
- huggingface_hub package: pip install huggingface_hub
- A Hugging Face Hub account with API token

Usage:
    python upload_to_hf.py --model [small|medium|large] --format [skops|onnx] 
                          --token PATH_TO_TOKEN --org ORG_NAME
"""

import argparse
import json
import lzma
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import HfApi, create_repo, upload_file
from charboundary import __version__ as charboundary_version
from charboundary.remote_models import get_resource_dir, get_onnx_dir


# README template for skops models
SKOPS_README_TEMPLATE = """---
language:
  - en
tags:
  - charboundary
  - sentence-boundary-detection
  - paragraph-detection
  - legal-text
  - legal-nlp
  - text-segmentation
  - cpu
  - document-processing
  - rag
license: mit
library_name: charboundary
pipeline_tag: text-classification
datasets:
  - alea-institute/alea-legal-benchmark-sentence-paragraph-boundaries
  - alea-institute/kl3m-data-snapshot-20250324
metrics:
  - accuracy
  - f1
  - precision
  - recall
  - throughput
papers:
  - https://arxiv.org/abs/2504.04131
---

# CharBoundary {size} Model

This is the {size} model for the [CharBoundary](https://github.com/alea-institute/charboundary) library (v{version}),
a fast character-based sentence and paragraph boundary detection system optimized for legal text.

## Model Details

- **Size**: {size}
- **Model Size**: {model_size_mb} MB (SKOPS compressed)
- **Memory Usage**: {memory_usage} MB at runtime
- **Training Data**: Legal text with {num_samples} samples from [KL3M dataset](https://huggingface.co/datasets/alea-institute/kl3m-data-snapshot-20250324)
- **Model Type**: Random Forest ({n_estimators} trees, max depth {max_depth})
- **Format**: scikit-learn model (serialized with skops)
- **Task**: Character-level boundary detection for text segmentation
- **License**: MIT
- **Throughput**: ~{throughput} characters/second

## Usage

> **Important:** When loading models from Hugging Face Hub, you must set `trust_model=True` to allow loading custom class types.
> 
> **Security Note:** The ONNX model variants are recommended in security-sensitive environments as they don't require bypassing skops security measures with `trust_model=True`. See the [ONNX versions](https://huggingface.co/alea-institute/charboundary-{size}-onnx) for a safer alternative.

```python
# pip install charboundary
from huggingface_hub import hf_hub_download
from charboundary import TextSegmenter

# Download the model
model_path = hf_hub_download(repo_id="alea-institute/charboundary-{size}", filename="model.pkl")

# Load the model (trust_model=True is required when loading from external sources)
segmenter = TextSegmenter.load(model_path, trust_model=True)

# Use the model
text = "This is a test sentence. Here's another one!"
sentences = segmenter.segment_to_sentences(text)
print(sentences)
# Output: ['This is a test sentence.', " Here's another one!"]

# Segment to spans
sentence_spans = segmenter.get_sentence_spans(text)
print(sentence_spans)
# Output: [(0, 24), (24, 44)]
```

## Performance

The model uses a character-based random forest classifier with the following configuration:
- Window Size: {left_window} characters before, {right_window} characters after potential boundary
- Accuracy: {accuracy:.4f}
- F1 Score: {f1_score:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}

### Dataset-specific Performance

| Dataset | Precision | F1 | Recall |
|---------|-----------|-------|--------|
| ALEA SBD Benchmark | {precision_alea:.3f} | {f1_alea:.3f} | {recall_alea:.3f} |
| SCOTUS | {precision_scotus:.3f} | {f1_scotus:.3f} | {recall_scotus:.3f} |
| Cyber Crime | {precision_cyber:.3f} | {f1_cyber:.3f} | {recall_cyber:.3f} |
| BVA | {precision_bva:.3f} | {f1_bva:.3f} | {recall_bva:.3f} |
| Intellectual Property | {precision_ip:.3f} | {f1_ip:.3f} | {recall_ip:.3f} |

## Available Models

CharBoundary comes in three sizes, balancing accuracy and efficiency:

| Model | Format | Size (MB) | Memory (MB) | Throughput (chars/sec) | F1 Score |
|-------|--------|-----------|-------------|------------------------|----------|
| Small | [SKOPS](https://huggingface.co/alea-institute/charboundary-small) / [ONNX](https://huggingface.co/alea-institute/charboundary-small-onnx) | 3.0 / 0.5 | 1,026 | ~748K | 0.773 |
| Medium | [SKOPS](https://huggingface.co/alea-institute/charboundary-medium) / [ONNX](https://huggingface.co/alea-institute/charboundary-medium-onnx) | 13.0 / 2.6 | 1,897 | ~587K | 0.779 |
| Large | [SKOPS](https://huggingface.co/alea-institute/charboundary-large) / [ONNX](https://huggingface.co/alea-institute/charboundary-large-onnx) | 60.0 / 13.0 | 5,734 | ~518K | 0.782 |

## Paper and Citation

This model is part of the research presented in the following paper:

```
@article{{bommarito2025precise,
  title={{Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary}},
  author={{Bommarito, Michael J and Katz, Daniel Martin and Bommarito, Jillian}},
  journal={{arXiv preprint arXiv:2504.04131}},
  year={{2025}}
}}
```

For more details on the model architecture, training, and evaluation, please see:
- [Paper: "Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary"](https://arxiv.org/abs/2504.04131)
- [CharBoundary GitHub repository](https://github.com/alea-institute/charboundary)
- [Annotated training data](https://huggingface.co/datasets/alea-institute/alea-legal-benchmark-sentence-paragraph-boundaries)

## Contact

This model is developed and maintained by the [ALEA Institute](https://aleainstitute.ai). 

For technical support, collaboration opportunities, or general inquiries:
 
- GitHub: https://github.com/alea-institute/kl3m-model-research
- Email: hello@aleainstitute.ai
- Website: https://aleainstitute.ai

For any questions, please contact [ALEA Institute](https://aleainstitute.ai) at [hello@aleainstitute.ai](mailto:hello@aleainstitute.ai) or
create an issue on this repository or [GitHub](https://github.com/alea-institute/kl3m-model-research).

![https://aleainstitute.ai](https://aleainstitute.ai/images/alea-logo-ascii-1x1.png)
"""

# README template for ONNX models
ONNX_README_TEMPLATE = """---
language:
  - en
tags:
  - charboundary
  - sentence-boundary-detection
  - paragraph-detection
  - legal-text
  - legal-nlp
  - text-segmentation
  - onnx
  - cpu
  - document-processing
  - rag
  - optimized-inference
license: mit
library_name: charboundary
pipeline_tag: text-classification
datasets:
  - alea-institute/alea-legal-benchmark-sentence-paragraph-boundaries
  - alea-institute/kl3m-data-snapshot-20250324
metrics:
  - accuracy
  - f1
  - precision
  - recall
  - throughput
papers:
  - https://arxiv.org/abs/2504.04131
---

# CharBoundary {size} ONNX Model

This is the {size} ONNX model for the [CharBoundary](https://github.com/alea-institute/charboundary) library (v{version}),
a fast character-based sentence and paragraph boundary detection system optimized for legal text.

## Model Details

- **Size**: {size}
- **Model Size**: {model_size_mb} MB (ONNX compressed)
- **Memory Usage**: {memory_usage} MB at runtime (non-ONNX version)
- **Training Data**: Legal text with {num_samples} samples from [KL3M dataset](https://huggingface.co/datasets/alea-institute/kl3m-data-snapshot-20250324)
- **Model Type**: Random Forest ({n_estimators} trees, max depth {max_depth}) converted to ONNX
- **Format**: ONNX optimized for inference
- **Task**: Character-level boundary detection for text segmentation
- **License**: MIT
- **Throughput**: ~{throughput} characters/second (base model; ONNX is typically 2-4x faster)

## Usage

> **Security Advantage:** This ONNX model format provides enhanced security compared to SKOPS models, as it doesn't require bypassing security measures with `trust_model=True`. ONNX models are the recommended option for security-sensitive environments.

```python
# Make sure to install with the onnx extra to get ONNX runtime support
# pip install charboundary[onnx]
from charboundary import get_{size}_onnx_segmenter

# First load can be slow
segmenter = get_{size}_onnx_segmenter()

# Use the model
text = "This is a test sentence. Here's another one!"
sentences = segmenter.segment_to_sentences(text)
print(sentences)
# Output: ['This is a test sentence.', " Here's another one!"]

# Segment to spans
sentence_spans = segmenter.get_sentence_spans(text)
print(sentence_spans)
# Output: [(0, 24), (24, 44)]
```

## Performance

ONNX models provide significantly faster inference compared to the standard scikit-learn models
while maintaining the same accuracy metrics. The performance differences between model sizes are shown below.

### Base Model Performance 

| Dataset | Precision | F1 | Recall |
|---------|-----------|-------|--------|
| ALEA SBD Benchmark | {precision_alea:.3f} | {f1_alea:.3f} | {recall_alea:.3f} |
| SCOTUS | {precision_scotus:.3f} | {f1_scotus:.3f} | {recall_scotus:.3f} |
| Cyber Crime | {precision_cyber:.3f} | {f1_cyber:.3f} | {recall_cyber:.3f} |
| BVA | {precision_bva:.3f} | {f1_bva:.3f} | {recall_bva:.3f} |
| Intellectual Property | {precision_ip:.3f} | {f1_ip:.3f} | {recall_ip:.3f} |

### Size and Speed Comparison

| Model | Format | Size (MB) | Memory Usage | Throughput (chars/sec) | F1 Score |
|-------|--------|-----------|--------------|------------------------|----------|
| Small | [SKOPS](https://huggingface.co/alea-institute/charboundary-small) / [ONNX](https://huggingface.co/alea-institute/charboundary-small-onnx) | 3.0 / 0.5 | 1,026 MB | ~748K | 0.773 |
| Medium | [SKOPS](https://huggingface.co/alea-institute/charboundary-medium) / [ONNX](https://huggingface.co/alea-institute/charboundary-medium-onnx) | 13.0 / 2.6 | 1,897 MB | ~587K | 0.779 |
| Large | [SKOPS](https://huggingface.co/alea-institute/charboundary-large) / [ONNX](https://huggingface.co/alea-institute/charboundary-large-onnx) | 60.0 / 13.0 | 5,734 MB | ~518K | 0.782 |

## Paper and Citation

This model is part of the research presented in the following paper:

```
@article{{bommarito2025precise,
  title={{Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary}},
  author={{Bommarito, Michael J and Katz, Daniel Martin and Bommarito, Jillian}},
  journal={{arXiv preprint arXiv:2504.04131}},
  year={{2025}}
}}
```

For more details on the model architecture, training, and evaluation, please see:
- [Paper: "Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary"](https://arxiv.org/abs/2504.04131)
- [CharBoundary GitHub repository](https://github.com/alea-institute/charboundary)
- [Annotated training data](https://huggingface.co/datasets/alea-institute/alea-legal-benchmark-sentence-paragraph-boundaries)


## Contact

This model is developed and maintained by the [ALEA Institute](https://aleainstitute.ai). 

For technical support, collaboration opportunities, or general inquiries:
 
- GitHub: https://github.com/alea-institute/kl3m-model-research
- Email: hello@aleainstitute.ai
- Website: https://aleainstitute.ai

For any questions, please contact [ALEA Institute](https://aleainstitute.ai) at [hello@aleainstitute.ai](mailto:hello@aleainstitute.ai) or
create an issue on this repository or [GitHub](https://github.com/alea-institute/kl3m-model-research).

![https://aleainstitute.ai](https://aleainstitute.ai/images/alea-logo-ascii-1x1.png)
"""


# Model metadata based on training scripts and paper results
MODEL_METADATA = {
    "small": {
        "n_estimators": 32,
        "max_depth": 16,
        "left_window": 5,
        "right_window": 3,
        "num_samples": "~50,000",
        "accuracy": 0.997,
        "f1_score": 0.773,
        "precision": 0.746,
        "recall": 0.987,
        "sample_rate": 0.001,
        "model_size_mb": 3.0,
        "memory_usage": 1026,
        "throughput": "748K",
        # Dataset-specific performance metrics from the paper
        "precision_alea": 0.624,
        "f1_alea": 0.718,
        "recall_alea": 0.845,
        "precision_scotus": 0.926,
        "f1_scotus": 0.773,
        "recall_scotus": 0.664,
        "precision_cyber": 0.939,
        "f1_cyber": 0.837,
        "recall_cyber": 0.755,
        "precision_bva": 0.937,
        "f1_bva": 0.870,
        "recall_bva": 0.812,
        "precision_ip": 0.927,
        "f1_ip": 0.883,
        "recall_ip": 0.843
    },
    "medium": {
        "n_estimators": 64,
        "max_depth": 20,
        "left_window": 5,
        "right_window": 3,
        "num_samples": "~500,000",
        "accuracy": 0.998,
        "f1_score": 0.779,
        "precision": 0.757,
        "recall": 0.991,
        "sample_rate": 0.01,
        "model_size_mb": 13.0,
        "memory_usage": 1897,
        "throughput": "587K",
        # Dataset-specific performance metrics from the paper
        "precision_alea": 0.631,
        "f1_alea": 0.722,
        "recall_alea": 0.842,
        "precision_scotus": 0.938,
        "f1_scotus": 0.775,
        "recall_scotus": 0.661,
        "precision_cyber": 0.961,
        "f1_cyber": 0.853,
        "recall_cyber": 0.767,
        "precision_bva": 0.957,
        "f1_bva": 0.875,
        "recall_bva": 0.806,
        "precision_ip": 0.948,
        "f1_ip": 0.889,
        "recall_ip": 0.837
    },
    "large": {
        "n_estimators": 100,
        "max_depth": 24,
        "left_window": 5,
        "right_window": 3,
        "num_samples": "~5,000,000",
        "accuracy": 0.999,
        "f1_score": 0.782,
        "precision": 0.763,
        "recall": 0.993,
        "sample_rate": 0.1,
        "model_size_mb": 60.0,
        "memory_usage": 5734,
        "throughput": "518K",
        # Dataset-specific performance metrics from the paper
        "precision_alea": 0.637,
        "f1_alea": 0.727,
        "recall_alea": 0.847,
        "precision_scotus": 0.950,
        "f1_scotus": 0.778,
        "recall_scotus": 0.658,
        "precision_cyber": 0.968,
        "f1_cyber": 0.853,
        "recall_cyber": 0.762,
        "precision_bva": 0.963,
        "f1_bva": 0.881,
        "recall_bva": 0.813,
        "precision_ip": 0.954,
        "f1_ip": 0.890,
        "recall_ip": 0.834
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload charboundary models to Hugging Face Hub")
    parser.add_argument("--model", type=str, choices=["small", "medium", "large"], required=True,
                       help="Model size to upload")
    parser.add_argument("--format", type=str, choices=["skops", "onnx", "both"], default="both",
                       help="Model format to upload (default: both)")
    parser.add_argument("--token", type=str, help="Path to file containing HF token")
    parser.add_argument("--org", type=str, default="alea-institute", 
                       help="Organization name on Hugging Face (default: alea-institute)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Prepare files but don't actually upload")
    return parser.parse_args()


def get_token(token_path: Optional[str] = None) -> Optional[str]:
    """Get Hugging Face token from file if specified."""
    if token_path and os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    
    return None  # Let HfApi handle token discovery itself


def get_model_paths(model_size: str) -> Dict[str, str]:
    """Get paths to model files."""
    resource_dir = get_resource_dir()
    onnx_dir = get_onnx_dir()
    
    return {
        "skops": os.path.join(resource_dir, f"{model_size}_model.skops.xz"),
        "onnx": os.path.join(onnx_dir, f"{model_size}_model.onnx.xz")
    }


def decompress_model(compressed_path: str, output_path: str) -> bool:
    """Decompress an XZ compressed model file."""
    try:
        with lzma.open(compressed_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"Error decompressing {compressed_path}: {e}")
        return False


def upload_skops_model(
    model_size: str, 
    model_path: str, 
    hf_token: Optional[str], 
    org_name: str,
    dry_run: bool = False
) -> bool:
    """Upload a skops model to Hugging Face Hub."""
    repo_id = f"{org_name}/charboundary-{model_size}"
    
    # Create a temporary directory for working files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Decompress the model
        model_filename = "model.pkl"
        output_path = os.path.join(temp_dir, model_filename)
        
        if not decompress_model(model_path, output_path):
            print(f"Failed to decompress {model_path}")
            return False
        
        # Create README
        readme_path = os.path.join(temp_dir, "README.md")
        model_info = MODEL_METADATA[model_size]
        
        # Include model metadata
        model_metadata = {
            "small": "small",
            "medium": "medium (default)",
            "large": "large"
        }
        
        readme_content = SKOPS_README_TEMPLATE.format(
            size=model_metadata.get(model_size, model_size),
            version=charboundary_version,
            n_estimators=model_info["n_estimators"],
            max_depth=model_info["max_depth"],
            left_window=model_info["left_window"],
            right_window=model_info["right_window"],
            num_samples=model_info["num_samples"],
            accuracy=model_info["accuracy"],
            f1_score=model_info["f1_score"],
            precision=model_info["precision"],
            recall=model_info["recall"],
            model_size_mb=model_info["model_size_mb"],
            memory_usage=model_info["memory_usage"],
            throughput=model_info["throughput"],
            # Dataset-specific metrics
            precision_alea=model_info["precision_alea"],
            f1_alea=model_info["f1_alea"],
            recall_alea=model_info["recall_alea"],
            precision_scotus=model_info["precision_scotus"],
            f1_scotus=model_info["f1_scotus"],
            recall_scotus=model_info["recall_scotus"],
            precision_cyber=model_info["precision_cyber"],
            f1_cyber=model_info["f1_cyber"],
            recall_cyber=model_info["recall_cyber"],
            precision_bva=model_info["precision_bva"],
            f1_bva=model_info["f1_bva"],
            recall_bva=model_info["recall_bva"],
            precision_ip=model_info["precision_ip"],
            f1_ip=model_info["f1_ip"],
            recall_ip=model_info["recall_ip"]
        )
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        if dry_run:
            print(f"DRY RUN: Would upload {model_size} skops model to {repo_id}")
            print(f"Files prepared at: {temp_dir}")
            print(f"  - {model_filename} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
            print(f"  - README.md")
            return True
        
        try:
            # Create or get the repository
            api = HfApi(token=hf_token)
            create_repo(repo_id, exist_ok=True)
            
            # Upload files
            print(f"Uploading {model_size} skops model to {repo_id}...")
            
            # Upload model
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=model_filename,
                repo_id=repo_id,
                commit_message=f"Upload {model_size} model v{charboundary_version}"
            )
            
            # Upload README
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=f"Update README for {model_size} model"
            )
            
            print(f"Successfully uploaded {model_size} skops model to {repo_id}")
            return True
            
        except Exception as e:
            print(f"Error uploading to {repo_id}: {e}")
            return False


def upload_onnx_model(
    model_size: str, 
    model_path: str, 
    hf_token: Optional[str], 
    org_name: str,
    dry_run: bool = False
) -> bool:
    """Upload an ONNX model to Hugging Face Hub."""
    repo_id = f"{org_name}/charboundary-{model_size}-onnx"
    
    # Create a temporary directory for working files
    with tempfile.TemporaryDirectory() as temp_dir:
        # For ONNX models, preserve the .xz extension as these need to remain compressed
        model_filename = "model.onnx.xz"
        output_path = os.path.join(temp_dir, model_filename)
        
        # Just copy the compressed file directly - no need to decompress 
        try:
            import shutil
            shutil.copy(model_path, output_path)
        except Exception as e:
            print(f"Failed to copy {model_path} to {output_path}: {e}")
            return False
        
        # Create README
        readme_path = os.path.join(temp_dir, "README.md")
        
        # Include model metadata
        model_metadata = {
            "small": "small",
            "medium": "medium (default)",
            "large": "large"
        }
        
        model_info = MODEL_METADATA[model_size]
        
        readme_content = ONNX_README_TEMPLATE.format(
            size=model_metadata.get(model_size, model_size),
            version=charboundary_version,
            n_estimators=model_info["n_estimators"],
            max_depth=model_info["max_depth"],
            left_window=model_info["left_window"],
            right_window=model_info["right_window"],
            num_samples=model_info["num_samples"],
            accuracy=model_info["accuracy"],
            f1_score=model_info["f1_score"],
            precision=model_info["precision"],
            recall=model_info["recall"],
            model_size_mb=model_info["model_size_mb"] / 5,  # ONNX is ~1/5 the size
            memory_usage=model_info["memory_usage"],
            throughput=model_info["throughput"],
            # Dataset-specific metrics
            precision_alea=model_info["precision_alea"],
            f1_alea=model_info["f1_alea"],
            recall_alea=model_info["recall_alea"],
            precision_scotus=model_info["precision_scotus"],
            f1_scotus=model_info["f1_scotus"],
            recall_scotus=model_info["recall_scotus"],
            precision_cyber=model_info["precision_cyber"],
            f1_cyber=model_info["f1_cyber"],
            recall_cyber=model_info["recall_cyber"],
            precision_bva=model_info["precision_bva"],
            f1_bva=model_info["f1_bva"],
            recall_bva=model_info["recall_bva"],
            precision_ip=model_info["precision_ip"],
            f1_ip=model_info["f1_ip"],
            recall_ip=model_info["recall_ip"]
        )
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        if dry_run:
            print(f"DRY RUN: Would upload {model_size} ONNX model to {repo_id}")
            print(f"Files prepared at: {temp_dir}")
            print(f"  - {model_filename} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
            print(f"  - README.md")
            return True
        
        try:
            # Create or get the repository
            api = HfApi(token=hf_token)
            create_repo(repo_id, exist_ok=True)
            
            # Upload files
            print(f"Uploading {model_size} ONNX model to {repo_id}...")
            
            # Upload model
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=model_filename,
                repo_id=repo_id,
                commit_message=f"Upload {model_size} ONNX model v{charboundary_version}"
            )
            
            # Upload README
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=f"Update README for {model_size} ONNX model"
            )
            
            print(f"Successfully uploaded {model_size} ONNX model to {repo_id}")
            return True
            
        except Exception as e:
            print(f"Error uploading to {repo_id}: {e}")
            return False


def main():
    """Main function."""
    args = parse_args()
    
    # Get Hugging Face token if provided explicitly
    hf_token = None
    if not args.dry_run and args.token:
        hf_token = get_token(args.token)
        
    if args.dry_run:
        hf_token = "dummy_token_for_dry_run"
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("Error: huggingface_hub package is not installed.")
        print("Please install it with: pip install huggingface_hub")
        return 1
    
    # Get model paths
    model_paths = get_model_paths(args.model)
    
    success = []
    
    # Upload skops model if requested
    if args.format in ["skops", "both"]:
        if os.path.exists(model_paths["skops"]):
            print(f"Processing {args.model} skops model...")
            if upload_skops_model(args.model, model_paths["skops"], hf_token, args.org, args.dry_run):
                success.append(f"{args.model} (skops)")
        else:
            print(f"Error: {args.model} skops model not found at {model_paths['skops']}")
    
    # Upload ONNX model if requested
    if args.format in ["onnx", "both"]:
        if os.path.exists(model_paths["onnx"]):
            print(f"Processing {args.model} ONNX model...")
            if upload_onnx_model(args.model, model_paths["onnx"], hf_token, args.org, args.dry_run):
                success.append(f"{args.model} (ONNX)")
        else:
            print(f"Error: {args.model} ONNX model not found at {model_paths['onnx']}")
    
    # Print summary
    print("\nSummary:")
    if success:
        if args.dry_run:
            print(f"Would successfully upload {len(success)} models:")
        else:
            print(f"Successfully uploaded {len(success)} models:")
        for model in success:
            prefix = "alea-institute/charboundary-"
            if "(ONNX)" in model:
                model_name = model.replace(" (ONNX)", "")
                print(f"  - {prefix}{model_name}-onnx")
            else:
                model_name = model.replace(" (skops)", "")
                print(f"  - {prefix}{model_name}")
    else:
        print("No models were processed successfully.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
