"""
Hardware backend modules for ML compiler.

This module provides support for different hardware backends including:
- CPU backends (LLVM, x86, ARM)
- GPU backends (CUDA, OpenCL, ROCm)
- Custom accelerators (TPU-like, Cerebras-like)
- Auto-tuning for hardware-specific optimizations
"""

from .cpu import CPUBackend, LLVMBackend
from .gpu import CUDABackend, OpenCLBackend 
from .custom import CustomAcceleratorBackend, CerebrasLikeBackend
from .auto_tuning import AutoTuner, HardwareProfiler

__all__ = [
    "CPUBackend",
    "LLVMBackend", 
    "CUDABackend",
    "OpenCLBackend",
    "CustomAcceleratorBackend",
    "CerebrasLikeBackend",
    "AutoTuner",
    "HardwareProfiler",
] 