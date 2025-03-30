#!/bin/bash
# Script to run all tests with coverage and benchmarking

# Run regular tests with coverage
echo "Running tests with coverage..."
pytest --cov=charboundary --cov-report=term --cov-report=html

# Run benchmarks if explicitly requested
if [ "$1" == "--benchmark" ]; then
    echo "Running benchmarks..."
    pytest tests/test_benchmarks.py -xvs --benchmark-only
fi

echo "Tests completed. HTML coverage report available in htmlcov/index.html"