"""
PyTorch Integration Example for ML Compiler Development.

This example demonstrates:
1. Converting PyTorch models to TVM/MLIR
2. Applying custom optimization passes
3. Comparing optimized vs unoptimized performance
4. Targeting different hardware backends
"""

import logging
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. This example will use simulation.")
    TORCH_AVAILABLE = False

from passes.fusion import OperatorFusionPass, ElementwiseFusionPass
from passes.memory import MemoryLayoutOptimizer, MemoryLayout
from backends.custom import CerebrasLikeBackend, SystolicArrayBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleConvNet(nn.Module):
    """Simple CNN for demonstration purposes."""
    
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_mock_pytorch_graph():
    """Create a mock PyTorch computation graph for simulation."""
    return {
        'operations': [
            {'type': 'conv2d', 'name': 'conv1', 'inputs': ['input'], 'outputs': ['conv1_out']},
            {'type': 'batch_norm', 'name': 'bn1', 'inputs': ['conv1_out'], 'outputs': ['bn1_out']},
            {'type': 'relu', 'name': 'relu1', 'inputs': ['bn1_out'], 'outputs': ['relu1_out']},
            {'type': 'max_pool2d', 'name': 'pool1', 'inputs': ['relu1_out'], 'outputs': ['pool1_out']},
            {'type': 'conv2d', 'name': 'conv2', 'inputs': ['pool1_out'], 'outputs': ['conv2_out']},
            {'type': 'batch_norm', 'name': 'bn2', 'inputs': ['conv2_out'], 'outputs': ['bn2_out']},
            {'type': 'relu', 'name': 'relu2', 'inputs': ['bn2_out'], 'outputs': ['relu2_out']},
            {'type': 'max_pool2d', 'name': 'pool2', 'inputs': ['relu2_out'], 'outputs': ['pool2_out']},
            {'type': 'flatten', 'name': 'flatten', 'inputs': ['pool2_out'], 'outputs': ['flatten_out']},
            {'type': 'linear', 'name': 'fc', 'inputs': ['flatten_out'], 'outputs': ['output']}
        ],
        'tensors': {
            'input': {'shape': [1, 3, 32, 32], 'dtype': 'float32'},
            'conv1_out': {'shape': [1, 32, 32, 32], 'dtype': 'float32'},
            'output': {'shape': [1, 10], 'dtype': 'float32'}
        },
        'framework': 'pytorch'
    }


def pytorch_to_compiler_graph(model, input_shape):
    """
    Convert PyTorch model to our compiler's internal graph representation.
    
    In a real implementation, this would use torch.jit.trace or torch.fx
    to extract the computation graph.
    """
    logger.info("Converting PyTorch model to compiler graph...")
    
    if TORCH_AVAILABLE:
        # In real implementation, we would trace the model
        # traced_model = torch.jit.trace(model, torch.randn(input_shape))
        # graph = extract_graph_from_trace(traced_model)
        pass
    
    # For demonstration, use mock graph
    compiler_graph = create_mock_pytorch_graph()
    
    logger.info(f"Extracted graph with {len(compiler_graph['operations'])} operations")
    return compiler_graph


def apply_optimization_pipeline(graph):
    """Apply a comprehensive optimization pipeline."""
    logger.info("Applying optimization pipeline...")
    
    optimized_graph = graph
    optimization_stats = {}
    
    # Step 1: Operator Fusion
    fusion_pass = OperatorFusionPass(fusion_patterns=["conv2d_bn_relu", "elementwise_chain"])
    optimized_graph = fusion_pass.apply(optimized_graph)
    optimization_stats['fusion_count'] = fusion_pass.fusion_count
    
    # Step 2: Elementwise Fusion
    elementwise_pass = ElementwiseFusionPass(max_fusion_depth=8)
    optimized_graph = elementwise_pass.apply(optimized_graph)
    optimization_stats['elementwise_fusion_count'] = elementwise_pass.fusion_count
    
    # Step 3: Memory Layout Optimization
    memory_optimizer = MemoryLayoutOptimizer(target_layout=MemoryLayout.NHWC)
    optimized_graph = memory_optimizer.apply(optimized_graph)
    optimization_stats['memory_optimizations'] = memory_optimizer.optimization_count
    
    logger.info(f"Optimization completed: {optimization_stats}")
    return optimized_graph, optimization_stats


def benchmark_model_performance(model, input_shape, num_runs=10):
    """Benchmark PyTorch model performance."""
    if not TORCH_AVAILABLE:
        # Simulate performance for demonstration
        return {
            'avg_time_ms': 15.5,
            'std_time_ms': 1.2,
            'throughput_fps': 64.5
        }
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1000.0 / avg_time  # FPS
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'throughput_fps': throughput
    }


def demonstrate_hardware_targeting(optimized_graph):
    """Demonstrate targeting different hardware backends."""
    logger.info("Demonstrating hardware backend targeting...")
    
    # Create different backend targets
    backends = [
        CerebrasLikeBackend(cores=50000, memory_per_core_kb=48),
        SystolicArrayBackend(array_size=(128, 128)),
    ]
    
    backend_results = {}
    
    for backend in backends:
        logger.info(f"Compiling for {backend.name}...")
        
        # Compile graph for backend
        compiled_graph = backend.compile(optimized_graph)
        
        # Simulate execution
        dummy_inputs = {'input': np.random.randn(1, 3, 32, 32)}
        execution_results = backend.execute(compiled_graph, dummy_inputs)
        
        backend_results[backend.name] = {
            'compilation_time_ms': 500,  # Simulated
            'execution_stats': execution_results['execution_stats']
        }
        
        logger.info(f"{backend.name} - Execution time: "
                   f"{execution_results['execution_stats']['execution_time_ms']:.2f}ms")
    
    return backend_results


def main():
    """Main demonstration function."""
    print("PyTorch Integration Example")
    print("=" * 50)
    
    # Create model
    input_shape = (1, 3, 32, 32)
    
    if TORCH_AVAILABLE:
        model = SimpleConvNet(num_classes=10)
        print(f"Created PyTorch model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Benchmark original model
        original_perf = benchmark_model_performance(model, input_shape)
        print(f"Original model performance: {original_perf['avg_time_ms']:.2f}ms")
    else:
        print("Using simulated PyTorch model (PyTorch not available)")
        original_perf = {'avg_time_ms': 20.0, 'throughput_fps': 50.0}
    
    # Convert to compiler graph
    if TORCH_AVAILABLE:
        compiler_graph = pytorch_to_compiler_graph(model, input_shape)
    else:
        compiler_graph = create_mock_pytorch_graph()
    
    print(f"Extracted graph with {len(compiler_graph['operations'])} operations")
    
    # Apply optimizations
    optimized_graph, opt_stats = apply_optimization_pipeline(compiler_graph)
    
    print("\nOptimization Results:")
    print(f"  - Operator fusions: {opt_stats['fusion_count']}")
    print(f"  - Elementwise fusions: {opt_stats['elementwise_fusion_count']}")
    print(f"  - Memory optimizations: {opt_stats['memory_optimizations']}")
    
    # Demonstrate hardware targeting
    backend_results = demonstrate_hardware_targeting(optimized_graph)
    
    print("\nHardware Backend Performance:")
    for backend_name, results in backend_results.items():
        exec_stats = results['execution_stats']
        print(f"  {backend_name}:")
        print(f"    - Execution time: {exec_stats['execution_time_ms']:.2f}ms")
        if 'throughput_achieved_tops' in exec_stats:
            print(f"    - Throughput: {exec_stats['throughput_achieved_tops']:.1f} TOPS")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print(f"  Original PyTorch: {original_perf['avg_time_ms']:.2f}ms")
    
    best_backend = min(backend_results.items(), 
                      key=lambda x: x[1]['execution_stats']['execution_time_ms'])
    best_time = best_backend[1]['execution_stats']['execution_time_ms']
    speedup = original_perf['avg_time_ms'] / best_time
    
    print(f"  Best optimized ({best_backend[0]}): {best_time:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    print("\nPyTorch integration demonstration completed!")


if __name__ == "__main__":
    main() 