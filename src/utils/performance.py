"""
Performance measurement and profiling utilities for ML compiler development.

This module provides:
- Performance profiling and benchmarking
- Execution time measurement
- Memory usage tracking
- Hardware utilization monitoring
"""

import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for ML workloads."""
    
    def __init__(self):
        self.measurements = []
        self.current_session = None
        
    def start_profiling(self, session_name: str):
        """Start a new profiling session."""
        self.current_session = {
            'name': session_name,
            'start_time': time.time(),
            'measurements': [],
            'memory_snapshots': []
        }
        logger.info(f"Started profiling session: {session_name}")
        
    def record_measurement(self, operation: str, duration_ms: float, 
                         metadata: Optional[Dict[str, Any]] = None):
        """Record a performance measurement."""
        if not self.current_session:
            logger.warning("No active profiling session")
            return
            
        measurement = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.current_session['measurements'].append(measurement)
        logger.debug(f"Recorded measurement: {operation} - {duration_ms:.2f}ms")
        
    def end_profiling(self) -> Dict[str, Any]:
        """End the current profiling session and return results."""
        if not self.current_session:
            logger.warning("No active profiling session")
            return {}
            
        session = self.current_session
        session['end_time'] = time.time()
        session['total_duration'] = session['end_time'] - session['start_time']
        
        # Calculate statistics
        session['statistics'] = self._calculate_statistics(session['measurements'])
        
        self.measurements.append(session)
        self.current_session = None
        
        logger.info(f"Ended profiling session: {session['name']}")
        return session
        
    def _calculate_statistics(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from measurements."""
        if not measurements:
            return {}
            
        durations = [m['duration_ms'] for m in measurements]
        
        return {
            'total_measurements': len(measurements),
            'total_time_ms': sum(durations),
            'average_time_ms': statistics.mean(durations),
            'median_time_ms': statistics.median(durations),
            'min_time_ms': min(durations),
            'max_time_ms': max(durations),
            'std_dev_ms': statistics.stdev(durations) if len(durations) > 1 else 0.0
        }


class BenchmarkSuite:
    """Comprehensive benchmarking suite for ML workloads."""
    
    def __init__(self):
        self.benchmark_results = []
        
    def run_benchmark(self, model: Any, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive benchmark on a model."""
        logger.info("Starting benchmark suite")
        
        results = {
            'model_info': self._get_model_info(model),
            'benchmark_config': benchmark_config,
            'performance_metrics': {},
            'hardware_utilization': {},
            'memory_usage': {}
        }
        
        # Run different benchmark tests
        results['performance_metrics'] = self._benchmark_performance(model, benchmark_config)
        results['hardware_utilization'] = self._benchmark_hardware_utilization(model)
        results['memory_usage'] = self._benchmark_memory_usage(model)
        
        self.benchmark_results.append(results)
        logger.info("Benchmark suite completed")
        return results
        
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract model information."""
        return {
            'model_type': type(model).__name__,
            'estimated_parameters': 1000000,  # Placeholder
            'estimated_size_mb': 10.0  # Placeholder
        }
        
    def _benchmark_performance(self, model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model performance."""
        num_runs = config.get('num_runs', 10)
        batch_size = config.get('batch_size', 1)
        
        # Simulate performance measurements
        execution_times = []
        for _ in range(num_runs):
            # Simulate execution time with some variance
            base_time = 10.0  # ms
            variance = np.random.normal(0, 1.0)
            execution_time = max(0.1, base_time + variance)
            execution_times.append(execution_time)
            
        return {
            'average_execution_time_ms': statistics.mean(execution_times),
            'min_execution_time_ms': min(execution_times),
            'max_execution_time_ms': max(execution_times),
            'std_dev_ms': statistics.stdev(execution_times),
            'throughput_samples_per_sec': 1000.0 / statistics.mean(execution_times) * batch_size,
            'num_runs': num_runs
        }
        
    def _benchmark_hardware_utilization(self, model: Any) -> Dict[str, Any]:
        """Benchmark hardware utilization."""
        # Simulate hardware utilization metrics
        return {
            'cpu_utilization': 0.75,
            'gpu_utilization': 0.85,
            'memory_bandwidth_utilization': 0.60,
            'cache_hit_rate': 0.92
        }
        
    def _benchmark_memory_usage(self, model: Any) -> Dict[str, Any]:
        """Benchmark memory usage."""
        # Simulate memory usage metrics
        return {
            'peak_memory_mb': 512.0,
            'average_memory_mb': 256.0,
            'memory_efficiency': 0.80,
            'memory_fragmentation': 0.15
        }
        
    def compare_benchmarks(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple benchmark results."""
        if len(benchmark_results) < 2:
            logger.warning("Need at least 2 benchmark results for comparison")
            return {}
            
        comparison = {
            'num_benchmarks': len(benchmark_results),
            'performance_comparison': {},
            'best_performing': None,
            'worst_performing': None
        }
        
        # Extract performance metrics for comparison
        avg_times = []
        for i, result in enumerate(benchmark_results):
            avg_time = result['performance_metrics']['average_execution_time_ms']
            avg_times.append((i, avg_time))
            
        # Sort by execution time (lower is better)
        avg_times.sort(key=lambda x: x[1])
        
        comparison['best_performing'] = avg_times[0][0]
        comparison['worst_performing'] = avg_times[-1][0]
        
        # Calculate speedup comparisons
        baseline_time = avg_times[-1][1]  # Worst performing as baseline
        speedups = []
        
        for idx, avg_time in avg_times:
            speedup = baseline_time / avg_time
            speedups.append({
                'benchmark_index': idx,
                'speedup': speedup,
                'execution_time_ms': avg_time
            })
            
        comparison['performance_comparison']['speedups'] = speedups
        
        logger.info(f"Compared {len(benchmark_results)} benchmarks")
        return comparison 