"""
Precision management utilities for STAMP benchmark
Handles different precision modes across ML frameworks
"""

from typing import Dict, Any, Union, Optional
from enum import Enum
import warnings

class PrecisionMode(Enum):
    """Supported precision modes."""
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    BF16 = "bf16"
    MIXED = "mixed"

class PrecisionManager:
    """Manages precision settings across different ML frameworks."""
    
    def __init__(self):
        """Initialize precision manager."""
        self.framework_support = self._initialize_framework_support()
        self.current_precision = PrecisionMode.FP32
        
    def _initialize_framework_support(self) -> Dict[str, Dict[str, bool]]:
        """Initialize framework precision support matrix."""
        return {
            "pytorch": {
                "fp16": True,
                "fp32": True,
                "fp64": True,
                "bf16": True,
                "mixed": True
            },
            "tensorflow": {
                "fp16": True,
                "fp32": True,
                "fp64": True,
                "bf16": False,  # Limited support
                "mixed": True
            },
            "jax": {
                "fp16": True,
                "fp32": True,
                "fp64": True,
                "bf16": True,
                "mixed": False  # Handled differently in JAX
            }
        }
    
    def is_precision_supported(self, framework: str, precision: str) -> bool:
        """Check if precision is supported by framework.
        
        Args:
            framework: Framework name ('pytorch', 'tensorflow', 'jax')
            precision: Precision mode string
            
        Returns:
            True if precision is supported
        """
        framework = framework.lower()
        precision = precision.lower()
        
        if framework not in self.framework_support:
            warnings.warn(f"Unknown framework: {framework}")
            return False
        
        return self.framework_support[framework].get(precision, False)
    
    def get_supported_precisions(self, framework: str) -> list:
        """Get list of supported precisions for a framework.
        
        Args:
            framework: Framework name
            
        Returns:
            List of supported precision strings
        """
        framework = framework.lower()
        if framework not in self.framework_support:
            return []
        
        return [
            precision for precision, supported 
            in self.framework_support[framework].items() 
            if supported
        ]
    
    def get_pytorch_dtype(self, precision: str):
        """Get PyTorch dtype for precision mode.
        
        Args:
            precision: Precision mode string
            
        Returns:
            PyTorch dtype object
        """
        try:
            import torch
            dtype_map = {
                "fp16": torch.float16,
                "fp32": torch.float32,
                "fp64": torch.float64,
                "bf16": torch.bfloat16
            }
            return dtype_map.get(precision.lower(), torch.float32)
        except ImportError:
            warnings.warn("PyTorch not available")
            return None
    
    def get_tensorflow_dtype(self, precision: str):
        """Get TensorFlow dtype for precision mode.
        
        Args:
            precision: Precision mode string
            
        Returns:
            TensorFlow dtype object
        """
        try:
            import tensorflow as tf
            dtype_map = {
                "fp16": tf.float16,
                "fp32": tf.float32,
                "fp64": tf.float64,
                "bf16": tf.bfloat16
            }
            return dtype_map.get(precision.lower(), tf.float32)
        except ImportError:
            warnings.warn("TensorFlow not available")
            return None
    
    def get_jax_dtype(self, precision: str):
        """Get JAX dtype for precision mode.
        
        Args:
            precision: Precision mode string
            
        Returns:
            JAX dtype object
        """
        try:
            import jax.numpy as jnp
            dtype_map = {
                "fp16": jnp.float16,
                "fp32": jnp.float32,
                "fp64": jnp.float64,
                "bf16": jnp.bfloat16
            }
            return dtype_map.get(precision.lower(), jnp.float32)
        except ImportError:
            warnings.warn("JAX not available")
            return None
    
    def configure_mixed_precision(self, framework: str, enable: bool = True) -> bool:
        """Configure mixed precision for a framework.
        
        Args:
            framework: Framework name
            enable: Whether to enable mixed precision
            
        Returns:
            True if configuration was successful
        """
        framework = framework.lower()
        
        if framework == "pytorch":
            try:
                import torch
                if enable:
                    # PyTorch AMP is enabled per-operation, not globally
                    print("PyTorch mixed precision: Use torch.cuda.amp.GradScaler and autocast")
                    return True
                return True
            except ImportError:
                return False
        
        elif framework == "tensorflow":
            try:
                import tensorflow as tf
                if enable:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("TensorFlow mixed precision enabled")
                else:
                    tf.keras.mixed_precision.set_global_policy('float32')
                    print("TensorFlow mixed precision disabled")
                return True
            except ImportError:
                return False
        
        elif framework == "jax":
            # JAX handles precision per-array, not globally
            print("JAX precision is handled per-array, not globally")
            return True
        
        return False
    
    def get_precision_info(self, precision: str) -> Dict[str, Any]:
        """Get detailed information about a precision mode.
        
        Args:
            precision: Precision mode string
            
        Returns:
            Dictionary with precision information
        """
        precision = precision.lower()
        
        info_map = {
            "fp16": {
                "name": "Half Precision (FP16)",
                "bits": 16,
                "range": "±65,504",
                "precision_digits": "~3-4",
                "memory_factor": 0.5,
                "speed_factor": "1.5-2x faster",
                "accuracy_impact": "Moderate loss",
                "use_cases": ["Inference", "Memory-constrained training"]
            },
            "fp32": {
                "name": "Single Precision (FP32)",
                "bits": 32,
                "range": "±3.4×10^38",
                "precision_digits": "~7",
                "memory_factor": 1.0,
                "speed_factor": "Baseline",
                "accuracy_impact": "Standard accuracy",
                "use_cases": ["Default training", "General purpose"]
            },
            "fp64": {
                "name": "Double Precision (FP64)",
                "bits": 64,
                "range": "±1.8×10^308",
                "precision_digits": "~15-17",
                "memory_factor": 2.0,
                "speed_factor": "0.5x slower",
                "accuracy_impact": "Highest accuracy",
                "use_cases": ["Scientific computing", "High precision requirements"]
            },
            "bf16": {
                "name": "Brain Float 16 (BF16)",
                "bits": 16,
                "range": "±3.4×10^38 (same as FP32)",
                "precision_digits": "~2-3",
                "memory_factor": 0.5,
                "speed_factor": "1.5-2x faster",
                "accuracy_impact": "Less loss than FP16",
                "use_cases": ["Training with reduced memory", "Modern hardware"]
            },
            "mixed": {
                "name": "Mixed Precision",
                "bits": "16+32",
                "range": "Variable",
                "precision_digits": "Variable",
                "memory_factor": 0.6,
                "speed_factor": "1.3-1.8x faster",
                "accuracy_impact": "Minimal with proper scaling",
                "use_cases": ["Optimal training", "Best of both worlds"]
            }
        }
        
        return info_map.get(precision, {
            "name": "Unknown Precision",
            "error": f"Unknown precision mode: {precision}"
        })
    
    def validate_precision_config(self, framework: str, precision: str, device: str = "cpu") -> Dict[str, Any]:
        """Validate a precision configuration.
        
        Args:
            framework: Framework name
            precision: Precision mode
            device: Target device ('cpu', 'cuda', 'gpu')
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        framework = framework.lower()
        precision = precision.lower()
        device = device.lower()
        
        # Check framework support
        if not self.is_precision_supported(framework, precision):
            result["valid"] = False
            result["errors"].append(f"{framework} does not support {precision}")
        
        # Device-specific checks
        if device == "cpu":
            if precision in ["fp16", "bf16"]:
                result["warnings"].append(f"{precision} on CPU may be slower than fp32")
            if precision == "mixed":
                result["warnings"].append("Mixed precision on CPU provides limited benefits")
        
        # Framework-specific recommendations
        if framework == "pytorch" and precision == "mixed":
            result["recommendations"].append("Use torch.cuda.amp.autocast() and GradScaler")
        
        if framework == "tensorflow" and precision == "mixed":
            result["recommendations"].append("Set mixed precision policy globally")
        
        if framework == "jax" and precision == "mixed":
            result["warnings"].append("JAX doesn't have built-in mixed precision like PyTorch/TF")
        
        return result
    
    def get_optimal_precision(self, framework: str, use_case: str = "training", device: str = "gpu") -> str:
        """Get optimal precision recommendation.
        
        Args:
            framework: Framework name
            use_case: Use case ('training', 'inference', 'research')
            device: Target device
            
        Returns:
            Recommended precision mode
        """
        framework = framework.lower()
        use_case = use_case.lower()
        device = device.lower()
        
        # Inference optimizations
        if use_case == "inference":
            if device == "gpu" and self.is_precision_supported(framework, "fp16"):
                return "fp16"
            return "fp32"
        
        # Training optimizations
        if use_case == "training":
            if device == "gpu":
                if self.is_precision_supported(framework, "mixed"):
                    return "mixed"
                elif self.is_precision_supported(framework, "bf16"):
                    return "bf16"
            return "fp32"
        
        # Research/high precision
        if use_case == "research":
            return "fp64" if self.is_precision_supported(framework, "fp64") else "fp32"
        
        return "fp32"


if __name__ == "__main__":
    # Example usage
    pm = PrecisionManager()
    
    # Check framework support
    frameworks = ["pytorch", "tensorflow", "jax"]
    for framework in frameworks:
        supported = pm.get_supported_precisions(framework)
        print(f"{framework}: {supported}")
    
    # Get precision info
    for precision in ["fp16", "fp32", "bf16", "mixed"]:
        info = pm.get_precision_info(precision)
        print(f"\n{precision}: {info['name']}")
        print(f"  Memory factor: {info.get('memory_factor', 'N/A')}")
        print(f"  Speed factor: {info.get('speed_factor', 'N/A')}")
    
    # Validate configurations
    config = pm.validate_precision_config("pytorch", "mixed", "cuda")
    print(f"\nPyTorch mixed precision validation: {config}")
