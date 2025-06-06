#!/bin/bash
# Script to set up the development environment for TVM and MLIR on Arch Linux.
#
# Key fixes based on official TVM documentation (https://tvm.apache.org/docs/install/from_source.html):
# 1. Changed CMAKE_BUILD_TYPE from Release to RelWithDebInfo (recommended)
# 2. Added HIDE_PRIVATE_SYMBOLS=ON to prevent symbol conflicts with PyTorch
# 3. Fixed USE_LLVM format to include --ignore-libllvm --link-static flags
# 4. Added explicit GPU SDK settings (all OFF for basic build)
# 5. Added validation and error checking for build steps
# 6. Added ccache configuration through TVM's USE_CCACHE option

set -e

# --- Configuration ---
# For LLVM, we use a specific LLVM version that is known to be compatible with TVM v0.16.0.
# For TVM, we stick to a release for better stability.
TVM_VERSION="v0.16.0" # A recent stable release
LLVM_VERSION="llvmorg-17.0.6" # LLVM 17.0.6 release

# Get the number of CPU cores to use for parallel builds
# You can also set this manually, e.g., NUM_CORES=8
NUM_CORES=$(nproc)

echo "--- Using ${NUM_CORES} cores for building ---"
echo "--- LLVM version: ${LLVM_VERSION} ---"
echo "--- TVM version: ${TVM_VERSION} ---"

# --- 1. Install Dependencies ---
echo "--- Installing system dependencies ---"
sudo pacman -Syu --needed base-devel cmake git ninja python uv ccache llvm clang lld

# --- 2. Build LLVM/MLIR from source ---
echo "--- Setting up LLVM/MLIR ---"
# As per TVM docs, we need a compatible LLVM version.
if [ ! -d "third_party/llvm-project" ]; then
    echo "--- Cloning LLVM project (${LLVM_VERSION}) ---"
    git clone --branch ${LLVM_VERSION} --depth 1 https://github.com/llvm/llvm-project.git third_party/llvm-project
fi

cd third_party/llvm-project

# Add git config to filter user and revert branches, as per docs.
echo "--- Configuring git remote for LLVM project ---"
git config --add remote.origin.fetch '^refs/heads/users/*'
git config --add remote.origin.fetch '^refs/heads/revert-*'

echo "--- Configuring and building LLVM/MLIR ---"
rm -rf build
cmake -G Ninja -S llvm -B build \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PWD}/build" \
    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native"

cmake --build build -j${NUM_CORES}

echo "--- Installing LLVM/MLIR headers ---"
cmake --build build --target install
cd ../.. # Back to project root

echo "--- LLVM/MLIR build complete ---"


# --- 3. Build TVM from source ---
echo "--- Setting up TVM ---"
if [ ! -d "third_party/tvm" ]; then
    echo "--- Cloning TVM project ---"
    git clone --recursive https://github.com/apache/tvm.git third_party/tvm
fi

cd third_party/tvm
git checkout ${TVM_VERSION}
git submodule update --init --recursive

echo "--- Configuring and building TVM ---"
rm -rf build
mkdir -p build
cp cmake/config.cmake build/

# Get absolute path to our custom LLVM build
LLVM_DIR=$(realpath ../llvm-project/build)
export PATH=${LLVM_DIR}/bin:$PATH # Add our new LLVM tools to the path

# Following official TVM docs configuration
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> build/config.cmake
echo "set(USE_LLVM \"${LLVM_DIR}/bin/llvm-config --ignore-libllvm --link-static\")" >> build/config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> build/config.cmake

# GPU SDKs - turn off for now
echo "set(USE_CUDA   OFF)" >> build/config.cmake
echo "set(USE_METAL  OFF)" >> build/config.cmake
echo "set(USE_VULKAN OFF)" >> build/config.cmake
echo "set(USE_OPENCL OFF)" >> build/config.cmake

# cuBLAS, cuDNN, cutlass support - turn off for now
echo "set(USE_CUBLAS OFF)" >> build/config.cmake
echo "set(USE_CUDNN  OFF)" >> build/config.cmake
echo "set(USE_CUTLASS OFF)" >> build/config.cmake

# MLIR support configuration
echo "set(USE_MLIR ON)" >> build/config.cmake
echo "set(USE_MLIR_CMAKE_PATH ${LLVM_DIR}/lib/cmake/mlir)" >> build/config.cmake
echo "set(MLIR_DIR ${LLVM_DIR}/lib/cmake/mlir)" >> build/config.cmake

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

echo "--- Running TVM CMake configuration ---"
cmake -G Ninja .. || {
    echo "ERROR: TVM CMake configuration failed"
    echo "Check the config.cmake file in the build directory for issues"
    exit 1
}

echo "--- Building TVM ---"
# For TVM, reduce parallelism to avoid memory issues and use ninja for better performance
# TVM is a very memory-intensive build, so we limit parallel jobs
TVM_PARALLEL_JOBS=$((NUM_CORES / 2))
if [ $TVM_PARALLEL_JOBS -lt 1 ]; then
    TVM_PARALLEL_JOBS=1
fi
echo "Using $TVM_PARALLEL_JOBS parallel jobs for TVM build (reduced from $NUM_CORES to avoid memory issues)"

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
echo "Available memory: ${AVAILABLE_MEM}MB"
if [ $AVAILABLE_MEM -lt 4000 ]; then
    echo "WARNING: Low memory detected (${AVAILABLE_MEM}MB). TVM build may fail."
    echo "Consider using fewer parallel jobs or freeing up memory."
    TVM_PARALLEL_JOBS=1
    echo "Forcing single-threaded build due to low memory."
fi

# Try building with reduced parallelism first
cmake --build . --parallel ${TVM_PARALLEL_JOBS} || {
    echo "Build failed with $TVM_PARALLEL_JOBS jobs, trying with single job..."
    cmake --build . --parallel 1 || {
        echo "ERROR: TVM build failed even with single job"
        echo "This might be due to insufficient memory or missing dependencies"
        exit 1
    }
}

cd ../../.. # Back to project root

echo "--- TVM build complete ---"

# --- 4. Set up Python environment with uv ---
echo "--- Setting up Python virtual environment with uv ---"
uv venv # Creates a .venv by default

echo "--- Installing Python dependencies into .venv ---"
# Install dependencies from pyproject.toml
uv pip install -e .
# Install the editable TVM python package
uv pip install -e "third_party/tvm/python"


echo "--- Development environment setup is complete! ---"
echo "A Python virtual environment has been created in the '.venv' directory."
echo "To activate it in your shell, run: source .venv/bin/activate"
echo "'uv' commands (like 'uv run') will automatically use this environment." 