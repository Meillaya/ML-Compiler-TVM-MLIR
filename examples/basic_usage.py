"""
Basic usage example for ML Compiler Development with TVM and MLIR.

This example demonstrates:
1. Creating and applying compiler passes
2. Memory layout optimization
3. Operator fusion
4. Backend targeting for different hardware
5. Performance comparison
"""

import logging
import sys
import os

# Add src to path for importing our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from passes.fusion import OperatorFusionPass, ElementwiseFusionPass
from passes.memory import MemoryLayoutOptimizer, TensorReorderPass
from passes.tvm_passes import create_tvm_optimization_pipeline
from passes.mlir_passes import create_mlir_optimization_pipeline, simulate_mlir_compilation
from backends.custom import CerebrasLikeBackend, SystolicArrayBackend, compare_accelerator_backends

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_fusion_passes():
    """Demonstrate operator fusion capabilities."""
    print("\n" + "="*60)
    print("OPERATOR FUSION DEMONSTRATION")
    print("="*60)
    
    # Create a mock computation graph
    mock_graph = {
        'nodes': ['conv2d', 'batch_norm', 'relu', 'add', 'mul'],
        'edges': [('conv2d', 'batch_norm'), ('batch_norm', 'relu'), ('relu', 'add'), ('add', 'mul')],
        'metadata': {'framework': 'pytorch'}
    }
    
    # Apply operator fusion pass
    fusion_pass = OperatorFusionPass(fusion_patterns=["conv2d_bn_relu", "elementwise_chain"])
    fused_graph = fusion_pass.apply(mock_graph)
    
    print(f"Original graph: {len(mock_graph['nodes'])} nodes")
    print(f"Fusions applied: {fusion_pass.fusion_count}")
    
    # Apply elementwise fusion
    elementwise_pass = ElementwiseFusionPass(max_fusion_depth=6)
    final_graph = elementwise_pass.apply(fused_graph)
    
    print(f"Elementwise fusions: {elementwise_pass.fusion_count}")
    
    return final_graph


def demonstrate_memory_optimization():
    """Demonstrate memory layout optimization."""
    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create mock tensor metadata
    mock_graph = {
        'tensors': {
            'input_tensor': {'shape': [1, 224, 224, 3], 'layout': 'NHWC'},
            'weight_tensor': {'shape': [64, 3, 7, 7], 'layout': 'NCHW'},
            'output_tensor': {'shape': [1, 112, 112, 64], 'layout': 'NHWC'}
        },
        'operations': ['conv2d', 'relu', 'maxpool']
    }
    
    # Apply memory layout optimization
    from passes.memory import MemoryLayout
    memory_optimizer = MemoryLayoutOptimizer(target_layout=MemoryLayout.NHWC)
    optimized_graph = memory_optimizer.apply(mock_graph)
    
    print(f"Memory optimizations applied: {memory_optimizer.optimization_count}")
    
    # Apply tensor reordering
    reorder_pass = TensorReorderPass(cache_line_size=64)
    final_graph = reorder_pass.apply(optimized_graph)
    
    print(f"Tensor reordering optimizations: {reorder_pass.optimization_count}")
    
    return final_graph


def demonstrate_mlir_compilation():
    """Demonstrate MLIR compilation pipeline."""
    print("\n" + "="*60)
    print("MLIR COMPILATION DEMONSTRATION")
    print("="*60)
    
    # Example MLIR IR (simplified)
    sample_mlir_ir = """
    func.func @matmul_add(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, 
                         %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
      %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) 
                         outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
      %1 = linalg.add ins(%0, %C : tensor<128x128xf32>, tensor<128x128xf32>) 
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
      return %1 : tensor<128x128xf32>
    }
    """
    
    # Simulate MLIR compilation
    compilation_results = simulate_mlir_compilation(sample_mlir_ir, target="llvm")
    
    print(f"Target: {compilation_results['target']}")
    print(f"Passes applied: {compilation_results['passes_applied']}")
    print(f"Compilation stats: {compilation_results['compilation_stats']}")
    
    return compilation_results


def demonstrate_custom_accelerators():
    """Demonstrate custom accelerator backends."""
    print("\n" + "="*60)
    print("CUSTOM ACCELERATOR DEMONSTRATION")
    print("="*60)
    
    # Create mock workload
    mock_workload = {
        'operations': ['matmul', 'conv2d', 'add', 'relu'],
        'tensor_sizes': [(1024, 1024), (224, 224, 3), (1024,), (1024,)],
        'compute_intensity': 'high'
    }
    
    # Create different accelerator backends
    cerebras_backend = CerebrasLikeBackend(cores=100000, memory_per_core_kb=48)
    systolic_backend = SystolicArrayBackend(array_size=(256, 256))
    
    backends = [cerebras_backend, systolic_backend]
    
    # Compare backends
    comparison = compare_accelerator_backends(mock_workload, backends)
    
    print(f"Best backend: {comparison['best_backend']}")
    print(f"Performance ranking: {comparison['performance_ranking']}")
    
    # Demonstrate execution on best backend
    best_backend = next(b for b in backends if b.name == comparison['best_backend'])
    compiled_graph = best_backend.compile(mock_workload)
    
    import numpy as np
    test_inputs = {'input': np.random.randn(32, 32)}
    results = best_backend.execute(compiled_graph, test_inputs)
    
    print(f"Execution results: {results['execution_stats']}")
    
    return comparison


def demonstrate_end_to_end_optimization():
    """Demonstrate end-to-end optimization pipeline."""
    print("\n" + "="*60)
    print("END-TO-END OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Start with a mock neural network
    model_graph = {
        'layers': [
            {'type': 'conv2d', 'params': {'filters': 64, 'kernel_size': 3}},
            {'type': 'batch_norm', 'params': {}},
            {'type': 'relu', 'params': {}},
            {'type': 'conv2d', 'params': {'filters': 128, 'kernel_size': 3}},
            {'type': 'add', 'params': {}},  # Residual connection
            {'type': 'relu', 'params': {}},
            {'type': 'global_avg_pool', 'params': {}},
            {'type': 'dense', 'params': {'units': 10}}
        ],
        'input_shape': [1, 224, 224, 3]
    }
    
    print(f"Original model: {len(model_graph['layers'])} layers")
    
    # Step 1: Apply fusion passes
    fusion_pass = OperatorFusionPass()
    fused_graph = fusion_pass.apply(model_graph)
    print(f"After fusion: {fusion_pass.fusion_count} fusions applied")
    
    # Step 2: Memory optimization
    from passes.memory import MemoryLayout
    memory_optimizer = MemoryLayoutOptimizer(target_layout=MemoryLayout.NHWC)
    memory_optimized = memory_optimizer.apply(fused_graph)
    print(f"Memory optimizations: {memory_optimizer.optimization_count}")
    
    # Step 3: Backend targeting
    cerebras_backend = CerebrasLikeBackend(cores=50000)
    compiled_graph = cerebras_backend.compile(memory_optimized)
    print(f"Compiled for: {cerebras_backend.name}")
    
    # Step 4: Performance estimation
    import numpy as np
    sample_input = {'input': np.random.randn(1, 224, 224, 3)}
    results = cerebras_backend.execute(compiled_graph, sample_input)
    
    print(f"Estimated performance:")
    print(f"  - Execution time: {results['execution_stats']['execution_time_ms']:.2f}ms")
    print(f"  - Throughput: {results['execution_stats']['throughput_achieved_tops']:.1f} TOPS")
    print(f"  - Cores utilized: {results['execution_stats']['cores_utilized']:,}")
    
    return results


def main():
    """Run all demonstrations."""
    print("ML Compiler Development - Basic Usage Examples")
    print("=" * 80)
    
    try:
        # Run demonstrations
        demonstrate_fusion_passes()
        demonstrate_memory_optimization()
        demonstrate_mlir_compilation()
        demonstrate_custom_accelerators()
        demonstrate_end_to_end_optimization()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 