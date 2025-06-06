#!/bin/bash
# Script to rebuild LLVM/MLIR with proper header installation

set -e

LLVM_VERSION="llvmorg-17.0.6"
NUM_CORES=$(nproc)

echo "--- Rebuilding LLVM/MLIR with proper installation ---"

if [ ! -d "third_party/llvm-project" ]; then
    echo "ERROR: LLVM source not found. Run the main setup script first."
    exit 1
fi

cd third_party/llvm-project

echo "--- Configuring LLVM/MLIR with installation ---"
rm -rf build
cmake -G Ninja -S llvm -B build \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PWD}/build" \
    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native"

echo "--- Building LLVM/MLIR ---"
cmake --build build -j${NUM_CORES}

echo "--- Installing LLVM/MLIR headers and libraries ---"
cmake --build build --target install

echo "--- Verifying MLIR headers are installed ---"
if [ -f "build/include/mlir/Analysis/Presburger/IntegerRelation.h" ]; then
    echo "✓ MLIR headers installed successfully"
else
    echo "✗ MLIR headers not found after installation"
    exit 1
fi

cd ../..

echo "--- LLVM/MLIR rebuild with installation complete ---"
echo "Now you can retry building TVM" 