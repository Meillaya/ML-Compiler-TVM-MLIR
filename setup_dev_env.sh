#!/bin/bash
# Simple script to set up TVM development environment with LLVM backend on Arch Linux

set -e

# Configuration
TVM_VERSION="v0.20.0"

echo "--- Setting up TVM development environment ---"
echo "--- TVM version: ${TVM_VERSION} ---"

# Install dependencies
echo "--- Installing dependencies ---"
sudo pacman -Syu --needed base-devel cmake git ninja python uv ccache
sudo pacman -S --needed llvm llvm-libs clang lld

# Set up Python environment first (needed for TVM build)
echo "--- Setting up Python environment ---"
uv venv
uv pip install cython numpy pybind11

# Clone TVM
echo "--- Setting up TVM ---"
if [ ! -d "third_party/tvm" ]; then
    git clone --recursive https://github.com/apache/tvm.git third_party/tvm
fi

cd third_party/tvm
git checkout ${TVM_VERSION}
git submodule update --init --recursive

# Configure TVM build
echo "--- Configuring TVM build ---"
rm -rf build
mkdir -p build

# Get the Python path from uv environment
PYTHON_PATH=$(cd ../../.. && uv run which python)

cat > build/config.cmake << EOF
set(USE_LLVM "/usr/bin/llvm-config")
set(USE_LLVM_IGNORE_LIBLLVM ON)
set(USE_LLVM_LINK_STATIC ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_MLIR ON)
set(USE_CCACHE AUTO)
set(USE_CUDA OFF)
set(USE_ROCM OFF)
set(USE_CUBLAS OFF)
set(USE_CUDNN  OFF)
set(USE_CUTLASS OFF)
set(USE_PROFILER ON)
set(USE_GRAPH_EXECUTOR ON)
set(HIDE_PRIVATE_SYMBOLS ON)
set(PYTHON_EXECUTABLE "${PYTHON_PATH}")
EOF

echo  >> build/config.cmake
# Build TVM
echo "--- Building TVM ---"
cd build
cmake -G Ninja ..
ninja -j$(nproc)

cd ../../.. # Back to project root

# Install project and TVM Python packages
echo "--- Installing Python packages ---"
uv pip install -e .
uv pip install -e "third_party/tvm/python"

echo "--- Setup complete! ---"
echo "To activate the environment: source .venv/bin/activate"
echo "Or use 'uv run' to run commands in the environment" 