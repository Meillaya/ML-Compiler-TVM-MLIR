#!/bin/bash
# Script to set up TVM and LLVM development environment using Arch Linux packages
# This approach uses system packages instead of building from source

set -e

echo "--- Setting up TVM and LLVM development environment using Arch packages ---"

# Install required dependencies
echo "--- Installing system dependencies ---"
sudo pacman -Syu --needed base-devel cmake git ninja python uv

# According to the TVM community discussion, LLVM 17 and 20 are well tested
# Let's use LLVM 19 as a good middle ground that's modern but stable
LLVM_VERSION="19"

echo "--- Installing LLVM ${LLVM_VERSION} with MLIR support ---"
sudo pacman -S --needed \
    llvm${LLVM_VERSION} \
    llvm${LLVM_VERSION}-libs \
    clang${LLVM_VERSION} \
    lld${LLVM_VERSION}

# Verify LLVM installation
echo "--- Verifying LLVM installation ---"
LLVM_CONFIG_PATH="/usr/lib/llvm${LLVM_VERSION}/bin/llvm-config"
if ! ${LLVM_CONFIG_PATH} --version; then
    echo "ERROR: LLVM ${LLVM_VERSION} installation failed or llvm-config not available"
    exit 1
fi

echo "LLVM ${LLVM_VERSION} version: $(${LLVM_CONFIG_PATH} --version)"
echo "LLVM ${LLVM_VERSION} prefix: $(${LLVM_CONFIG_PATH} --prefix)"

# Install TVM from AUR
echo "--- Installing TVM from AUR ---"
if ! command -v yay &> /dev/null; then
    echo "ERROR: yay (AUR helper) is not installed. Please install yay first:"
    echo "  git clone https://aur.archlinux.org/yay.git"
    echo "  cd yay && makepkg -si"
    exit 1
fi

# Note: The AUR TVM package might not be the latest version
# If you need the latest version, we'll build from source but use system LLVM
echo "Available TVM package in AUR:"
yay -Si tvm

echo "--- Do you want to install TVM from AUR (y) or build from source with system LLVM (n)? ---"
read -p "Choice [y/n]: " choice

if [[ $choice == [Yy] ]]; then
    echo "--- Installing TVM from AUR ---"
    yay -S --needed tvm
else
    echo "--- Building TVM from source using system LLVM ${LLVM_VERSION} ---"
    
    # Create or update TVM directory
    if [ ! -d "third_party/tvm" ]; then
        echo "--- Cloning TVM project ---"
        mkdir -p third_party
        git clone --recursive https://github.com/apache/tvm.git third_party/tvm
    fi
    
    cd third_party/tvm
    
    # Use latest TVM version
    git checkout main
    git submodule update --init --recursive
    
    echo "--- Configuring TVM build to use system LLVM ${LLVM_VERSION} ---"
    rm -rf build
    mkdir -p build
    
    # Get LLVM configuration from system installation
    LLVM_CONFIG_PATH="/usr/lib/llvm${LLVM_VERSION}/bin/llvm-config"
    LLVM_PREFIX=$(${LLVM_CONFIG_PATH} --prefix)
    
    # Create config.cmake for TVM build
    cat > build/config.cmake << EOF
# TVM Build Configuration using system LLVM ${LLVM_VERSION}
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "${LLVM_CONFIG_PATH} --ignore-libllvm --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)

# GPU SDKs - turn off for now  
set(USE_CUDA   OFF)
set(USE_METAL  OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)

# cuBLAS, cuDNN, cutlass support - turn off for now
set(USE_CUBLAS OFF)
set(USE_CUDNN  OFF)
set(USE_CUTLASS OFF)

# MLIR support configuration (using system LLVM)
set(USE_MLIR ON)
set(USE_MLIR_CMAKE_PATH ${LLVM_PREFIX}/lib/cmake/mlir)
set(MLIR_DIR ${LLVM_PREFIX}/lib/cmake/mlir)

# Use ccache if available
set(USE_CCACHE AUTO)

# Additional stability settings for the build
set(CMAKE_CXX_STANDARD 17)
set(LLVM_ENABLE_RTTI OFF)
set(LLVM_ENABLE_EH OFF)
EOF
    
    echo "--- Building TVM ---"
    cd build
    
    # Use environment variable for build path
    export TVM_BUILD_PATH="$(pwd)"
    
    # Configure with CMake
    cmake -G Ninja .. || {
        echo "ERROR: TVM CMake configuration failed"
        exit 1
    }
    
    # Build with appropriate parallelism
    NUM_CORES=$(nproc)
    TVM_PARALLEL_JOBS=$((NUM_CORES / 2))
    if [ $TVM_PARALLEL_JOBS -lt 1 ]; then
        TVM_PARALLEL_JOBS=1
    fi
    
    echo "Using $TVM_PARALLEL_JOBS parallel jobs for TVM build"
    
    cmake --build . --parallel ${TVM_PARALLEL_JOBS} || {
        echo "Build failed with $TVM_PARALLEL_JOBS jobs, trying with single job..."
        cmake --build . --parallel 1 || {
            echo "ERROR: TVM build failed even with single job"
            exit 1
        }
    }
    
    cd ../../.. # Back to project root
fi

echo "--- Setting up Python virtual environment with uv ---"
uv venv # Creates a .venv by default

echo "--- Installing Python dependencies ---"
# Install dependencies from pyproject.toml
uv pip install -e .

# Install TVM Python package
if [[ $choice == [Yy] ]]; then
    echo "--- TVM installed from AUR should be available system-wide ---"
    # Test if TVM is available
    uv run python -c "import tvm; print(f'TVM version: {tvm.__version__}')" || {
        echo "WARNING: TVM from AUR might not be in Python path. You may need to install python-tvm separately."
    }
else
    echo "--- Installing TVM Python package from source build ---"
    uv pip install -e "third_party/tvm/python"
fi

echo "--- Development environment setup complete! ---"
echo ""
echo "LLVM ${LLVM_VERSION} is installed at: $(llvm-config-${LLVM_VERSION} --prefix)"
echo "Python virtual environment created in: .venv"
echo ""
echo "To activate the environment: source .venv/bin/activate"
echo "Or use 'uv run' to run commands in the environment automatically"
echo ""
echo "--- Testing installation ---"
echo "LLVM version: $(${LLVM_CONFIG_PATH} --version)"
echo "Testing TVM import..."
uv run python -c "
import tvm
print(f'TVM version: {tvm.__version__}')
print(f'TVM build info:')
for key, value in tvm.support.libinfo().items():
    if 'LLVM' in key or 'MLIR' in key:
        print(f'  {key}: {value}')
" 