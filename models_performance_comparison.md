# CharBoundary Models Performance Comparison

The following table compares the performance of the three included models in charboundary, showing both scikit-learn and ONNX inference speeds:

| Model  | scikit-learn (chars/sec) | ONNX (chars/sec) | Speedup |
|--------|-------------------------:|------------------:|--------:|
| Small  | 544,678                  | 572,176          | 1.05x   |
| Medium | 293,897                  | 345,157          | 1.17x   |
| Large  | 179,998                  | 301,539          | 1.68x   |

## Key Observations:

1. **Small Model**: 
   - Fastest overall processing with scikit-learn (544,678 chars/sec)
   - Modest ONNX speedup of 1.05x
   - Best choice for speed-critical applications with lower accuracy requirements

2. **Medium Model**:
   - Balanced performance with scikit-learn (293,897 chars/sec)
   - 17% performance improvement with ONNX
   - Recommended default choice for most applications

3. **Large Model**:
   - Slowest with scikit-learn (179,998 chars/sec)
   - Greatest ONNX improvement (1.68x)
   - With ONNX, performance increases to 301,539 chars/sec
   - Best choice for accuracy-critical applications

## Analysis

- ONNX optimization benefits larger models more substantially
- The complex structure of larger models gets optimized better with ONNX compilation
- For production deployment, the large model with ONNX provides excellent accuracy while maintaining competitive speed
- ONNX conversion particularly makes the large model more viable for performance-sensitive applications

## Testing Environment

These benchmarks were performed on a standard development machine with the following specs:
- Python 3.11
- ONNX runtime 1.21.0
- Batch size: 10,000 samples
- Test runs: 5-10 iterations

Actual performance may vary based on hardware, batch size, and specific text characteristics.