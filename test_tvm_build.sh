#!/bin/bash
# Test script to validate TVM build configuration

set -e

TVM_VERSION="v0.16.0"
NUM_CORES=$(nproc)

echo "--- Testing TVM build configuration ---"

# Check if LLVM is already built
if [ ! -d "third_party/llvm-project/build" ]; then
    echo "ERROR: LLVM build not found. Please run the full setup script first to build LLVM."
    exit 1
fi

# Check if TVM source exists
if [ ! -d "third_party/tvm" ]; then
    echo "--- Cloning TVM project ---"
    git clone --recursive https://github.com/apache/tvm.git third_party/tvm
fi

cd third_party/tvm
git checkout ${TVM_VERSION}
git submodule update --init --recursive

echo "--- Configuring TVM build ---"
rm -rf build
mkdir -p build
cp cmake/config.cmake build/

# Get absolute path to our custom LLVM build
LLVM_DIR=$(realpath ../llvm-project/build)
export PATH=${LLVM_DIR}/bin:$PATH

# Apply the corrected TVM configuration based on official docs
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> build/config.cmake
echo "set(USE_LLVM \"${LLVM_DIR}/bin/llvm-config --ignore-libllvm --link-static\")" >> build/config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> build/config.cmake

# GPU SDKs - turn off for basic build
echo "set(USE_CUDA   OFF)" >> build/config.cmake
echo "set(USE_METAL  OFF)" >> build/config.cmake
echo "set(USE_VULKAN OFF)" >> build/config.cmake
echo "set(USE_OPENCL OFF)" >> build/config.cmake

# cuBLAS, cuDNN, cutlass support - turn off for basic build
echo "set(USE_CUBLAS OFF)" >> build/config.cmake
echo "set(USE_CUDNN  OFF)" >> build/config.cmake
echo "set(USE_CUTLASS OFF)" >> build/config.cmake

# MLIR support configuration
echo "set(USE_MLIR ON)" >> build/config.cmake
echo "set(USE_MLIR_CMAKE_PATH ${LLVM_DIR}/lib/cmake/mlir)" >> build/config.cmake

# Use ccache if available
echo "set(USE_CCACHE AUTO)" >> build/config.cmake

# Additional stability settings for the build
echo "set(CMAKE_CXX_STANDARD 17)" >> build/config.cmake
echo "set(LLVM_ENABLE_RTTI OFF)" >> build/config.cmake
echo "set(LLVM_ENABLE_EH OFF)" >> build/config.cmake

cd build

# Validate that llvm-config exists and works
echo "--- Validating LLVM installation ---"
if ! ${LLVM_DIR}/bin/llvm-config --version; then
    echo "ERROR: LLVM installation appears to be broken"
    exit 1
fi

echo "LLVM version: $(${LLVM_DIR}/bin/llvm-config --version)"
echo "LLVM flags: $(${LLVM_DIR}/bin/llvm-config --cxxflags)"

echo "--- Generated config.cmake content ---"
cat config.cmake

echo "--- Running TVM CMake configuration ---"
cmake -G Ninja .. || {
    echo "ERROR: TVM CMake configuration failed"
    echo "Check the config.cmake file above for issues"
    exit 1
}

echo "--- TVM CMake configuration successful! ---"

# Calculate recommended parallel jobs
TVM_PARALLEL_JOBS=$((NUM_CORES / 2))
if [ $TVM_PARALLEL_JOBS -lt 1 ]; then
    TVM_PARALLEL_JOBS=1
fi

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
echo "Available memory: ${AVAILABLE_MEM}MB"
if [ $AVAILABLE_MEM -lt 4000 ]; then
    echo "WARNING: Low memory detected. Recommend single-threaded build."
    TVM_PARALLEL_JOBS=1
fi

echo "To build TVM, run: cmake --build . --parallel ${TVM_PARALLEL_JOBS}"
echo "Or for a safe single-threaded build: cmake --build . --parallel 1" 