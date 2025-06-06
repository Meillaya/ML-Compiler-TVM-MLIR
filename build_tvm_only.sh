#!/bin/bash
# Script to build TVM only (assumes it's already configured)

set -e

NUM_CORES=$(nproc)

if [ ! -d "third_party/tvm/build" ]; then
    echo "ERROR: TVM build directory not found. Run test_tvm_build.sh first."
    exit 1
fi

cd third_party/tvm/build

# Calculate parallel jobs (reduce to avoid memory issues)
TVM_PARALLEL_JOBS=$((NUM_CORES / 2))
if [ $TVM_PARALLEL_JOBS -lt 1 ]; then
    TVM_PARALLEL_JOBS=1
fi

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
echo "Available memory: ${AVAILABLE_MEM}MB"
echo "CPU cores: $NUM_CORES"
echo "Planned parallel jobs: $TVM_PARALLEL_JOBS"

if [ $AVAILABLE_MEM -lt 4000 ]; then
    echo "WARNING: Low memory detected (${AVAILABLE_MEM}MB). TVM build may fail."
    echo "Forcing single-threaded build due to low memory."
    TVM_PARALLEL_JOBS=1
fi

echo "--- Building TVM with $TVM_PARALLEL_JOBS parallel jobs ---"

# Try building with reduced parallelism first
cmake --build . --parallel ${TVM_PARALLEL_JOBS} || {
    echo "Build failed with $TVM_PARALLEL_JOBS jobs, trying with single job..."
    cmake --build . --parallel 1 || {
        echo "ERROR: TVM build failed even with single job"
        echo "This might be due to insufficient memory or missing dependencies"
        echo ""
        echo "Troubleshooting tips:"
        echo "1. Check if you have enough RAM (TVM needs 8GB+ for parallel builds)"
        echo "2. Close other applications to free memory"
        echo "3. Try building with SWAP enabled"
        echo "4. Check for missing system dependencies"
        exit 1
    }
}

echo "--- TVM build completed successfully! ---" 