# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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