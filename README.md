# ML Compiler Development with TVM and MLIR

This is an educational framework for developing compiler passes that optimize machine learning workloads, with a particular focus on targeting novel architectures like Cerebras-like wafer-scale processors. This project aims to demonstrates the theoretical foundations and practical implementation of ML compiler optimization techniques including operator fusion, memory layout optimization, auto-tuning, and custom accelerator targeting.

## Overview

This project serves as an exercise in understanding and implementing ML compiler optimization strategies, drawing inspiration from modern compiler infrastructures like TVM and MLIR. The framework provides hands-on experience with the mathematical foundations of compiler optimization, dataflow analysis, and hardware-specific code generation. A central focus of this work is exploring compilation strategies for massively parallel architectures similar to Cerebras wafer-scale processors, which represent a paradigm shift in ML accelerator design with their 400,000+ processing elements and distributed memory hierarchy.


## Quick Start

Begin by installing the latest LLVM toolchain from the Arch User Repository, which provides the foundation for both TVM and MLIR compilation.

```bash
yay -S llvm-git clang-git mlir
```

Clone the repository and navigate to the project directory. The setup script will automatically configure the development environment, build TVM from source with MLIR support, and install all necessary Python dependencies using uv package manager.

```bash
git clone git@github.com:Meillaya/ML-Compiler-TVM-MLIR.git
cd ml-compiler-dev-tvm-mlir
./setup_dev_env.sh
```

Activate the virtual environment.

```bash
source .venv/bin/activate
```

## Running Examples

Run the PyTorch integration example to observe the complete compilation pipeline in action. This example demonstrates graph extraction from PyTorch models, application of optimization passes, and targeting of different hardware backends including the Cerebras-like wafer-scale processor simulation.

```bash
uv run examples/pytorch_integration_example.py
```

The example will load a convolutional neural network, extract its computation graph, apply operator fusion and memory optimization passes, and compare performance across different backend targets. The output provides detailed statistics on optimization effectiveness and execution characteristics for each target architecture.

For a more comprehensive demonstration of the optimization pipeline, run the basic usage example which showcases the full range of compiler passes and backend targeting capabilities.

```bash
uv run examples/basic_usage.py
```

## Cerebras-Like Architecture Targeting

This project implements a sophisticated simulation of Cerebras-like wafer-scale processor architectures, providing insight into the unique compilation challenges presented by massively parallel spatial computing systems. The Cerebras WSE represents a fundamental departure from traditional GPU architectures, featuring hundreds of thousands of processing elements with distributed memory and dataflow execution models.

The compiler framework addresses the specific characteristics of wafer-scale architectures including non-uniform memory access patterns, complex communication topologies, and the need for sophisticated load balancing across tens of thousands of processing elements. The implementation explores how traditional compiler optimizations must be adapted for architectures where communication costs dominate computation costs and where spatial mapping of operations becomes critical for performance.



## References
- [Cerebras Inference Documentation](https://inference-docs.cerebras.ai/quickstart)
- [TVM Documentation](https://tvm.apache.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM Documentation](https://llvm.org/docs/) 