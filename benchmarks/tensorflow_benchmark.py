"""
TensorFlow Benchmark Implementation
Matrix multiplication and gradient benchmarking with precision profiling
"""

import tensorflow as tf
import time
from typing import Dict, Any, List
import gc
from contextlib import contextmanager

class TensorFlowBenchmark:
    """TensorFlow-specific benchmark implementation."""
    
    def __init__(self, device: str = "auto"):
        """Initialize TensorFlow benchmark.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'gpu', 'auto')
        """
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        if device == "auto":
            self.device = "/GPU:0" if len(gpus) > 0 else "/CPU:0"
        elif device == "gpu":
            self.device = "/GPU:0"
        else:
            self.device = "/CPU:0"
        
        self.results = []
    
    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.last_time = end - start
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        try:
            if "GPU" in self.device:
                gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                return {
                    "allocated": gpu_details['current'] / 1024**3,  # GB
                    "peak": gpu_details['peak'] / 1024**3,  # GB
                    "cached": 0.0  # TensorFlow doesn't expose cached memory
                }
        except:
            pass
        return {"allocated": 0.0, "peak": 0.0, "cached": 0.0}
    
    def benchmark_matmul(self, size: int, precision: str = "fp32", warmup_runs: int = 10, benchmark_runs: int = 100) -> Dict[str, Any]:
        """Benchmark matrix multiplication.
        
        Args:
            size: Matrix size (size x size)
            precision: Precision mode ('fp16', 'fp32', 'mixed')
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        # Set up precision
        dtype_map = {
            "fp16": tf.float16,
            "fp32": tf.float32,
            "mixed": tf.float32  # Mixed precision handled via policy
        }
        dtype = dtype_map.get(precision, tf.float32)
        
        # Set mixed precision policy if needed
        if precision == "mixed":
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        with tf.device(self.device):
            # Create test matrices
            A = tf.random.normal([size, size], dtype=dtype)
            B = tf.random.normal([size, size], dtype=dtype)
            
            # Warmup
            for _ in range(warmup_runs):
                _ = tf.linalg.matmul(A, B)
            
            # Benchmark
            times = []
            for _ in range(benchmark_runs):
                with self.timer():
                    result = tf.linalg.matmul(A, B)
                times.append(self.last_time)
        
        memory_stats = self.get_memory_usage()
        
        # Reset mixed precision policy
        if precision == "mixed":
            tf.keras.mixed_precision.set_global_policy('float32')
        
        # Cleanup
        del A, B, result
        gc.collect()
        
        return {
            "framework": "tensorflow",
            "operation": "matmul",
            "size": size,
            "precision": precision,
            "device": self.device,
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
            precision: Precision mode ('fp16', 'fp32', 'mixed')
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        dtype_map = {
            "fp16": tf.float16,
            "fp32": tf.float32,
            "mixed": tf.float32
        }
        dtype = dtype_map.get(precision, tf.float32)
        
        # Set mixed precision policy if needed
        if precision == "mixed":
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        with tf.device(self.device):
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(size * 2, activation='relu', input_shape=(size,)),
                tf.keras.layers.Dense(size, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            if precision != "mixed":
                # Cast model weights to desired precision
                for layer in model.layers:
                    if hasattr(layer, 'kernel'):
                        layer.kernel = tf.cast(layer.kernel, dtype)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias = tf.cast(layer.bias, dtype)
            
            optimizer = tf.keras.optimizers.Adam()
            loss_fn = tf.keras.losses.MeanSquaredError()
            
            # Create test data
            x = tf.random.normal([64, size], dtype=dtype)
            target = tf.random.normal([64, 1], dtype=dtype)
            
            # Warmup
            for _ in range(warmup_runs):
                with tf.GradientTape() as tape:
                    predictions = model(x)
                    loss = loss_fn(target, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Benchmark
            times = []
            for _ in range(benchmark_runs):
                with self.timer():
                    with tf.GradientTape() as tape:
                        predictions = model(x)
                        loss = loss_fn(target, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                times.append(self.last_time)
        
        memory_stats = self.get_memory_usage()
        
        # Reset mixed precision policy
        if precision == "mixed":
            tf.keras.mixed_precision.set_global_policy('float32')
        
        # Cleanup
        del model, x, target, predictions, loss, gradients
        gc.collect()
        
        return {
            "framework": "tensorflow",
            "operation": "gradient",
            "size": size,
            "precision": precision,
            "device": self.device,
            "times": times,
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "memory_usage": memory_stats,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs
        }
    
    def run_full_benchmark(self, sizes: List[int] = [512, 1024, 2048], precisions: List[str] = ["fp32", "mixed"]) -> List[Dict[str, Any]]:
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
                # Skip mixed precision on CPU
                if precision == "mixed" and "CPU" in self.device:
                    continue
                
                print(f"Running TensorFlow benchmark: size={size}, precision={precision}")
                
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
    benchmark = TensorFlowBenchmark()
    results = benchmark.run_full_benchmark()
    
    for result in results:
        print(f"{result['framework']} {result['operation']}: {result['mean_time']:.4f}s (size={result['size']}, precision={result['precision']})")
