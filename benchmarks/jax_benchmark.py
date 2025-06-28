"""
JAX Benchmark Implementation
Matrix multiplication and gradient benchmarking with precision profiling
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import time
from typing import Dict, Any, List
import gc
from contextlib import contextmanager
import optax

class JAXBenchmark:
    """JAX-specific benchmark implementation."""
    
    def __init__(self, device: str = "auto"):
        """Initialize JAX benchmark.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'gpu', 'auto')
        """
        # Configure JAX devices
        if device == "auto":
            self.device = jax.devices()[0]
        elif device == "gpu":
            gpu_devices = [d for d in jax.devices() if d.device_kind == 'gpu']
            self.device = gpu_devices[0] if gpu_devices else jax.devices('cpu')[0]
        else:
            self.device = jax.devices('cpu')[0]
        
        self.results = []
        
        # Set default device
        jax.config.update('jax_platform_name', self.device.platform)
    
    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        start = time.perf_counter()
        yield
        # JAX operations are async by default, so we need to block
        jax.block_until_ready(None)
        end = time.perf_counter()
        self.last_time = end - start
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        try:
            if self.device.device_kind == 'gpu':
                # JAX doesn't provide direct memory introspection
                # This is a placeholder for future implementation
                return {
                    "allocated": 0.0,  # Not directly available in JAX
                    "cached": 0.0,
                    "peak": 0.0
                }
        except:
            pass
        return {"allocated": 0.0, "cached": 0.0, "peak": 0.0}
    
    def benchmark_matmul(self, size: int, precision: str = "fp32", warmup_runs: int = 10, benchmark_runs: int = 100) -> Dict[str, Any]:
        """Benchmark matrix multiplication.
        
        Args:
            size: Matrix size (size x size)
            precision: Precision mode ('fp16', 'fp32', 'bf16')
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        # Set up precision
        dtype_map = {
            "fp16": jnp.float16,
            "fp32": jnp.float32,
            "bf16": jnp.bfloat16
        }
        dtype = dtype_map.get(precision, jnp.float32)
        
        # Create test matrices
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        
        A = jax.random.normal(key1, (size, size), dtype=dtype)
        B = jax.random.normal(key2, (size, size), dtype=dtype)
        
        # JIT compile the operation
        @jit
        def matmul_op(a, b):
            return jnp.matmul(a, b)
        
        # Warmup
        for _ in range(warmup_runs):
            result = matmul_op(A, B)
            jax.block_until_ready(result)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            with self.timer():
                result = matmul_op(A, B)
                jax.block_until_ready(result)
            times.append(self.last_time)
        
        memory_stats = self.get_memory_usage()
        
        # Cleanup
        del A, B, result
        gc.collect()
        
        return {
            "framework": "jax",
            "operation": "matmul",
            "size": size,
            "precision": precision,
            "device": str(self.device),
            "times": times,
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "memory_usage": memory_stats,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs
        }
    
    def benchmark_gradient(self, size: int, precision: str = "fp32", warmup_runs: int = 10, benchmark_runs: int = 100) -> Dict[str, Any]:
        """Benchmark gradient computation.
        
        Args:
            size: Input size for gradient computation
            precision: Precision mode ('fp16', 'fp32', 'bf16')  
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        dtype_map = {
            "fp16": jnp.float16,
            "fp32": jnp.float32,
            "bf16": jnp.bfloat16
        }
        dtype = dtype_map.get(precision, jnp.float32)
        
        # Define a simple neural network
        def init_network_params(key, input_size):
            k1, k2, k3 = jax.random.split(key, 3)
            return [
                jax.random.normal(k1, (input_size, input_size * 2), dtype=dtype) * 0.1,
                jax.random.normal(k2, (input_size * 2, input_size), dtype=dtype) * 0.1,
                jax.random.normal(k3, (input_size, 1), dtype=dtype) * 0.1,
            ]
        
        def network(params, x):
            w1, w2, w3 = params
            h1 = jax.nn.relu(jnp.dot(x, w1))
            h2 = jax.nn.relu(jnp.dot(h1, w2))
            return jnp.dot(h2, w3)
        
        def loss_fn(params, x, y):
            pred = network(params, x)
            return jnp.mean((pred - y) ** 2)
        
        # Initialize parameters and data
        key = jax.random.PRNGKey(42)
        key_params, key_data, key_target = jax.random.split(key, 3)
        
        params = init_network_params(key_params, size)
        x = jax.random.normal(key_data, (64, size), dtype=dtype)
        y = jax.random.normal(key_target, (64, 1), dtype=dtype)
        
        # JIT compile gradient computation
        grad_fn = jit(grad(loss_fn))
        
        # Warmup
        for _ in range(warmup_runs):
            grads = grad_fn(params, x, y)
            jax.block_until_ready(grads)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            with self.timer():
                grads = grad_fn(params, x, y)
                jax.block_until_ready(grads)
            times.append(self.last_time)
        
        memory_stats = self.get_memory_usage()
        
        # Cleanup
        del params, x, y, grads
        gc.collect()
        
        return {
            "framework": "jax",
            "operation": "gradient",
            "size": size,
            "precision": precision,
            "device": str(self.device),
            "times": times,
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "memory_usage": memory_stats,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs
        }
    
    def run_full_benchmark(self, sizes: List[int] = [512, 1024, 2048], precisions: List[str] = ["fp32", "bf16"]) -> List[Dict[str, Any]]:
        """Run complete benchmark suite.
        
        Args:
            sizes: List of matrix sizes to benchmark
            precisions: List of precision modes to test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for size in sizes:
            for precision in precisions:
                # Skip fp16/bf16 on CPU for stability
                if precision in ["fp16", "bf16"] and self.device.device_kind == 'cpu':
                    continue
                
                print(f"Running JAX benchmark: size={size}, precision={precision}")
                
                # Matrix multiplication benchmark
                matmul_result = self.benchmark_matmul(size, precision)
                results.append(matmul_result)
                
                # Gradient benchmark
                grad_result = self.benchmark_gradient(size // 4, precision)  # Smaller size for gradient
                results.append(grad_result)
        
        self.results.extend(results)
        return results


if __name__ == "__main__":
    # Example usage
    benchmark = JAXBenchmark()
    results = benchmark.run_full_benchmark()
    
    for result in results:
        print(f"{result['framework']} {result['operation']}: {result['mean_time']:.4f}s (size={result['size']}, precision={result['precision']})")
