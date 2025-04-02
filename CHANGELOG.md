# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - 2025-04-02

### Changed
- Disabled feature selection for medium model to improve inference speed
- Fixed ONNX conversion utility to properly handle XZ compressed files
- Added quality comparison between scikit-learn and ONNX models
- Improved benchmarking tools to evaluate both performance and accuracy

## [0.4.0] - 2025-04-01

### Added
- ONNX model conversion and inference support
- ONNX optimization levels (0-3) for performance tuning
- Remote model retrieval for both standard and ONNX models
- ONNX acceleration with up to 2.1x performance boost
- Automatic model downloads from GitHub
- New example scripts demonstrating ONNX usage and optimization
- Performance benchmarks for ONNX vs. sklearn models
- Comprehensive documentation for ONNX features
- Consolidated benchmark scripts into a single comprehensive tool

### Changed
- Reduced package size by including only the small model in the distribution
- Medium and large models are now downloaded from GitHub on first use
- Improved documentation explaining automatic model downloads
- Made model loading more robust with better error handling

## [0.3.0] - 2025-03-31

### Added
- Automatic download of large model from GitHub when not available locally
- Performance benchmarks for all model sizes
- Comprehensive legal abbreviation support
- Pattern hash for common text patterns
- Character n-gram cache for feature computation
- Prediction caching for repeated segments

### Changed
- Optimized text segmentation with frozensets and direct array operations
- Fixed model loading to prevent resource leaks
- Updated model feature extraction for better accuracy
- Improved handling of legal abbreviations and citations
- Special handling for quotations and multi-part abbreviations
- Changed NumPy array type from int16 to int32 to handle larger character encodings

### Performance
- Small model: ~85,000 characters/second
- Medium model: ~280,000 characters/second (best performance)
- Large model: ~175,000 characters/second

## [0.2.0] - 2025-03-29

### Added
- Feature selection capability to improve model performance
- Optimized model sizes after grid search
- Threshold parameter to control segmentation sensitivity
- Documentation on threshold calibration

## [0.1.0] - 2025-03-27

### Added
- Initial release with basic sentence and paragraph boundary detection
- Small, medium, and large pre-trained models
- Command-line interface for text analysis and training
- Support for abbreviation handling
- Basic documentation and examples