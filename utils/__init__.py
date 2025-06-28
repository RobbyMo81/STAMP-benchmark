"""
STAMP Benchmark Utilities
Profiling helpers and precision management utilities
"""

__version__ = "0.1.0"

# Import classes conditionally to avoid import errors
try:
    from .profiler import BenchmarkProfiler, SystemProfiler
    from .precision import PrecisionManager
    
    __all__ = [
        "BenchmarkProfiler",
        "PrecisionManager", 
        "SystemProfiler"
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some utilities may not be available due to missing dependencies: {e}")
    __all__ = []
