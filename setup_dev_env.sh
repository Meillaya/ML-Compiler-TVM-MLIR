#!/bin/bash
# Script to set up the development environment for TVM and MLIR on Arch Linux.

set -e

# --- Configuration ---
# For LLVM, we will use the latest main branch as per official documentation.
# For TVM, we stick to a release for better stability.
TVM_VERSION="v0.16.0" # A recent stable release

# Get the number of CPU cores to use for parallel builds
# You can also set this manually, e.g., NUM_CORES=8
NUM_CORES=$(nproc)

echo "--- Using ${NUM_CORES} cores for building ---"
echo "--- LLVM version: latest main branch ---"
echo "--- TVM version: ${TVM_VERSION} ---"

# --- 1. Install Dependencies ---
echo "--- Installing system dependencies ---"
sudo pacman -Syu --needed base-devel cmake git ninja python uv

# --- 2. Build LLVM/MLIR from source ---
echo "--- Setting up LLVM/MLIR ---"
# As per https://llvm.org/docs/GettingStarted.html#sources, we use a shallow clone of the main branch.
if [ ! -d "third_party/llvm-project" ]; then
    echo "--- Cloning LLVM project (shallow clone) ---"
    git clone --depth 1 https://github.com/llvm/llvm-project.git third_party/llvm-project
fi

cd third_party/llvm-project

# Add git config to filter user and revert branches, as per docs.
echo "--- Configuring git remote for LLVM project ---"
git config --add remote.origin.fetch '^refs/heads/users/*'
git config --add remote.origin.fetch '^refs/heads/revert-*'

echo "--- Configuring and building LLVM/MLIR ---"
cmake -G Ninja -S llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD="Native"

cmake --build build -j${NUM_CORES}
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

echo "--- Configuring and building TVM ---"
mkdir -p build
cp cmake/config.cmake build/

# Get absolute path to our custom LLVM build
LLVM_DIR=$(realpath ../llvm-project/build)

echo "set(USE_LLVM ${LLVM_DIR}/bin/llvm-config)" >> build/config.cmake
echo "set(USE_MLIR ON)" >> build/config.cmake

cd build
cmake ..
make -j${NUM_CORES}
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