"""
Profiling utilities for STAMP benchmark
GPU memory tracking, system profiling, and performance monitoring
"""

import psutil
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import threading
import os

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import py3nvml.py3nvml as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class BenchmarkProfiler:
    """System profiling for benchmark runs."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the profiler.
        
        Args:
            log_dir: Directory to save profiling logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.monitor_thread = None
        self.system_stats = []
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "os": os.name,
                "platform": psutil.PLATFORM,
                "python_version": psutil.python_version(),
            },
            "cpu": {
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            }
        }
        
        # Add GPU information if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                info["gpus"] = []
                for gpu in gpus:
                    info["gpus"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "temperature": gpu.temperature,
                        "load": gpu.load
                    })
            except Exception as e:
                info["gpu_error"] = str(e)
        
        return info
    
    def get_gpu_memory_usage(self) -> List[Dict[str, Any]]:
        """Get detailed GPU memory usage."""
        gpu_stats = []
        
        if NVML_AVAILABLE:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_stats.append({
                        "gpu_id": i,
                        "name": name,
                        "memory_total": mem_info.total,
                        "memory_used": mem_info.used,
                        "memory_free": mem_info.free,
                        "memory_percent": (mem_info.used / mem_info.total) * 100
                    })
            except Exception as e:
                gpu_stats.append({"error": str(e)})
        
        elif GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_stats.append({
                        "gpu_id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal * 1024 * 1024,  # Convert to bytes
                        "memory_used": gpu.memoryUsed * 1024 * 1024,
                        "memory_free": gpu.memoryFree * 1024 * 1024,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                    })
            except Exception as e:
                gpu_stats.append({"error": str(e)})
        
        return gpu_stats
    
    def start_monitoring(self, interval: float = 0.1):
        """Start system monitoring in background thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.system_stats = []
        
        def monitor():
            while self.monitoring:
                stats = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "gpu_stats": self.get_gpu_memory_usage()
                }
                self.system_stats.append(stats)
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop system monitoring and return collected stats.
        
        Returns:
            List of system monitoring data points
        """
        if not self.monitoring:
            return []
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        stats = self.system_stats.copy()
        self.system_stats = []
        return stats
    
    def save_benchmark_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file.
        
        Args:
            results: List of benchmark results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Prepare comprehensive log entry
        log_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_info(),
                "total_benchmarks": len(results)
            },
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def profile_function(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function execution.
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Dictionary with profiling results
        """
        # Start monitoring
        self.start_monitoring(interval=0.05)
        
        # Get initial system state
        initial_memory = psutil.virtual_memory().percent
        initial_gpu = self.get_gpu_memory_usage()
        
        # Execute function
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        end_time = time.perf_counter()
        
        # Stop monitoring and get stats
        monitoring_stats = self.stop_monitoring()
        
        # Get final system state
        final_memory = psutil.virtual_memory().percent
        final_gpu = self.get_gpu_memory_usage()
        
        return {
            "execution_time": end_time - start_time,
            "success": success,
            "error": error,
            "result": result,
            "memory_change": final_memory - initial_memory,
            "gpu_initial": initial_gpu,
            "gpu_final": final_gpu,
            "monitoring_stats": monitoring_stats,
            "stats_summary": {
                "avg_cpu": sum(s["cpu_percent"] for s in monitoring_stats) / len(monitoring_stats) if monitoring_stats else 0,
                "max_memory": max(s["memory_percent"] for s in monitoring_stats) if monitoring_stats else 0,
                "min_memory": min(s["memory_percent"] for s in monitoring_stats) if monitoring_stats else 0
            }
        }


class SystemProfiler:
    """Simplified system profiler for quick checks."""
    
    @staticmethod
    def get_quick_stats() -> Dict[str, Any]:
        """Get quick system statistics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu_count": len(GPUtil.getGPUs()) if GPU_AVAILABLE else 0
        }
    
    @staticmethod
    def check_system_requirements() -> Dict[str, bool]:
        """Check if system meets benchmark requirements."""
        requirements = {
            "sufficient_memory": psutil.virtual_memory().total >= 8 * 1024**3,  # 8GB
            "gpu_available": GPU_AVAILABLE and len(GPUtil.getGPUs()) > 0,
            "low_cpu_usage": psutil.cpu_percent(interval=1) < 80,
            "sufficient_disk": psutil.disk_usage('/').free >= 1 * 1024**3  # 1GB free
        }
        
        return requirements


if __name__ == "__main__":
    # Example usage
    profiler = BenchmarkProfiler()
    
    # Get system info
    system_info = profiler.get_system_info()
    print("System Info:", json.dumps(system_info, indent=2, default=str))
    
    # Check requirements
    requirements = SystemProfiler.check_system_requirements()
    print("System Requirements:", requirements)
