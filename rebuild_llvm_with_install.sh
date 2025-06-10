#!/bin/bash
# Script to install pre-built LLVM/MLIR from Arch Linux packages instead of building.

set -e

# Using LLVM from Arch Linux packages.
# To use the latest llvm from pacman, leave LLVM_VERSION empty.
# For a specific version, use e.g., "17".
LLVM_VERSION="" # Using latest pre-built version from Arch repos.

if [ -z "${LLVM_VERSION}" ]; then
    echo "--- Installing latest pre-built LLVM/MLIR from Arch Linux packages ---"
else
    echo "--- Installing pre-built LLVM/MLIR version ${LLVM_VERSION} from Arch Linux packages ---"
fi

# This script is intended for Arch Linux, so we check for pacman.
if ! command -v pacman &> /dev/null; then
    echo "ERROR: 'pacman' command not found. This script is designed for Arch Linux."
    exit 1
fi

# Install required LLVM packages using pacman.
if [ -z "${LLVM_VERSION}" ]; then
    echo "--- Installing latest LLVM with MLIR support ---"
    sudo pacman -Syu --needed llvm llvm-libs clang lld
    LLVM_CONFIG_PATH="/usr/bin/llvm-config"
else
    echo "--- Installing LLVM ${LLVM_VERSION} with MLIR support ---"
    sudo pacman -Syu --needed \
        llvm${LLVM_VERSION} \
        llvm${LLVM_VERSION}-libs \
        clang${LLVM_VERSION} \
        lld${LLVM_VERSION}
    LLVM_CONFIG_PATH="/usr/lib/llvm${LLVM_VERSION}/bin/llvm-config"
fi

# Verify the installation.
echo "--- Verifying LLVM installation ---"

if ! ${LLVM_CONFIG_PATH} --version; then
    echo "ERROR: LLVM installation failed or llvm-config is not available at ${LLVM_CONFIG_PATH}"
    exit 1
fi

LLVM_PREFIX=$(${LLVM_CONFIG_PATH} --prefix)
echo "LLVM version: $(${LLVM_CONFIG_PATH} --version)"
echo "LLVM install prefix: ${LLVM_PREFIX}"

echo "--- Verifying MLIR headers are installed ---"
MLIR_INCLUDE_DIR=$(${LLVM_CONFIG_PATH} --includedir)
MLIR_HEADER="mlir/Analysis/Presburger/IntegerRelation.h"

# The header path for versioned LLVM on Arch is typically under /usr/lib/llvm<version>/include
# but llvm-config should give the correct path.
if [ -f "${MLIR_INCLUDE_DIR}/${MLIR_HEADER}" ]; then
    echo "✓ MLIR headers installed successfully at ${MLIR_INCLUDE_DIR}/${MLIR_HEADER}"
else
    echo "✗ MLIR header not found at the expected path: ${MLIR_INCLUDE_DIR}/${MLIR_HEADER}"
    echo "Please verify the contents of the llvm${LLVM_VERSION} package."
    exit 1
fi

echo "--- LLVM/MLIR installation from pre-built packages is complete. ---"
echo "You can now proceed with building TVM." 