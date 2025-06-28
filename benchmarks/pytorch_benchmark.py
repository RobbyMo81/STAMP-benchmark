"""
PyTorch Benchmark Implementation
Matrix multiplication and gradient benchmarking with precision profiling
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, List
import gc
from contextlib import contextmanager

class PyTorchBenchmark:
    """PyTorch-specific benchmark implementation."""
    
    def __init__(self, device: str = "auto"):
        """Initialize PyTorch benchmark.
        
        Args:
            device: Device to run benchmarks on ('cpu', 'cuda', 'auto')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.results = []
    
    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.last_time = end - start
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved(self.device) / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            }
        return {"allocated": 0.0, "cached": 0.0, "max_allocated": 0.0}
    
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
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.float32)
        
        # Create test matrices
        A = torch.randn(size, size, dtype=dtype, device=self.device)
        B = torch.randn(size, size, dtype=dtype, device=self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = torch.matmul(A, B)
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            with self.timer():
                result = torch.matmul(A, B)
            times.append(self.last_time)
        
        memory_stats = self.get_memory_usage()
        
        # Cleanup
        del A, B, result
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "framework": "pytorch",
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
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.float32)
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.ReLU(),
            nn.Linear(size * 2, size),
            nn.ReLU(),
            nn.Linear(size, 1)
        ).to(self.device).to(dtype)
        
        criterion = nn.MSELoss()
        
        # Create test data
        x = torch.randn(64, size, dtype=dtype, device=self.device, requires_grad=True)
        target = torch.randn(64, 1, dtype=dtype, device=self.device)
        
        # Warmup
        for _ in range(warmup_runs):
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            with self.timer():
                output = model(x)
                loss = criterion(output, target)
                loss.backward()
            times.append(self.last_time)
            model.zero_grad()
        
        memory_stats = self.get_memory_usage()
        
        # Cleanup
        del model, x, target, output, loss
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "framework": "pytorch",
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
    
    def run_full_benchmark(self, sizes: List[int] = [512, 1024, 2048], precisions: List[str] = ["fp32", "fp16"]) -> List[Dict[str, Any]]:
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
                # Skip fp16 on CPU
                if precision == "fp16" and self.device.type == "cpu":
                    continue
                
                print(f"Running PyTorch benchmark: size={size}, precision={precision}")
                
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
    benchmark = PyTorchBenchmark()
    results = benchmark.run_full_benchmark()
    
    for result in results:
        print(f"{result['framework']} {result['operation']}: {result['mean_time']:.4f}s (size={result['size']}, precision={result['precision']})")
