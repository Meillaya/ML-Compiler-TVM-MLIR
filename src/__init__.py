"""
ML Compiler Development with TVM and MLIR

This package provides tools and utilities for building compiler passes 
to optimize ML workloads including operator fusion, memory layout optimization,
auto-tuning for hardware backends, and integration with PyTorch/TensorFlow.
"""

__version__ = "0.1.0"
__author__ = "ML Compiler Dev Team"

from . import passes
from . import optimization
from . import backends
from . import integration
from . import utils

__all__ = [
    "passes",
    "optimization", 
    "backends",
    "integration",
    "utils",
] 