# ML Compiler Development with TVM and MLIR

A comprehensive educational project for building compiler passes to optimize machine learning workloads using TVM and MLIR frameworks.

## Project Overview

This project aims to demonstrates the development of a simple yet comprehensive ML compiler that includes:

- **Operator Fusion**: Combine multiple operations to reduce memory bandwidth and improve performance
- **Memory Layout Optimization**: Transform tensor layouts for better cache performance
- **Auto-tuning**: Optimize kernels for specific hardware targets
- **Custom Accelerator Support**: Target novel architectures like Cerebras-like wafer-scale processors
- **Framework Integration**: Work with PyTorch and TensorFlow models



## Getting Started

### Prerequisites

- Python 3.8+
- CMake 3.18+
- LLVM/Clang (for building TVM and MLIR)
- Git

### Environment Setup

1. **Clone the repository**
2. **Run the setup script** (builds TVM and MLIR):
   ```bash
   ./setup_dev_env.sh
   ```
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Quick Start

Run the basic usage example:

```bash
python examples/basic_usage.py
```

This demonstrates:
- Operator fusion passes
- Memory layout optimization
- MLIR compilation pipeline
- Custom accelerator targeting
- End-to-end optimization

### PyTorch Integration Example

```bash
python examples/pytorch_integration_example.py
```

## Key Features

### 1. Operator Fusion

```python
from passes.fusion import OperatorFusionPass, ElementwiseFusionPass

# Fuse common patterns like Conv2D + BatchNorm + ReLU
fusion_pass = OperatorFusionPass(fusion_patterns=["conv2d_bn_relu", "matmul_add"])
fused_graph = fusion_pass.apply(computation_graph)
```

### 2. Memory Layout Optimization

```python
from passes.memory import MemoryLayoutOptimizer, MemoryLayout

# Optimize layout (e.g., NCHW -> NHWC for better fusion)
memory_optimizer = MemoryLayoutOptimizer(target_layout=MemoryLayout.NHWC)
optimized_graph = memory_optimizer.apply(graph)
```

### 3. Custom Accelerator Support

```python
from backends.custom import CerebrasLikeBackend, SystolicArrayBackend

# Target Cerebras-like architecture with 400k cores
cerebras_backend = CerebrasLikeBackend(cores=400000, memory_per_core_kb=48)
compiled_graph = cerebras_backend.compile(optimized_graph)
results = cerebras_backend.execute(compiled_graph, inputs)
```

## Advanced Extensions

### Custom Accelerator Targeting

- **Cerebras-like Wafer-Scale Processors**: Massive parallelism with distributed memory
- **TPU-like Systolic Arrays**: Optimized for matrix operations
- **Dataflow Accelerators**: Reconfigurable compute with custom dataflow patterns

### Performance Comparison

Compare performance across different backends and against hand-tuned kernels.

## Educational Goals

This project teaches:

1. **Compiler Design Principles**: How modern ML compilers work
2. **Optimization Techniques**: Common optimization strategies for ML workloads
3. **Hardware Targeting**: How to generate efficient code for different architectures
4. **Performance Analysis**: How to measure and improve compiler effectiveness
5. **Practical Implementation**: Real-world compiler development skills

## Dependencies

- **TVM**: Tensor compiler stack for optimizing ML workloads
- **MLIR**: Multi-level intermediate representation for compiler infrastructure
- **LLVM**: Compiler infrastructure for code generation
- **PyTorch/TensorFlow**: ML framework integration

## References

- [TVM Documentation](https://tvm.apache.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM Documentation](https://llvm.org/docs/) 