"""
Configuration management utilities for ML compiler development.

This module provides:
- Compiler configuration management
- Backend configuration
- Optimization settings
- Environment setup
"""

import logging
from typing import Dict, List, Any, Optional, Union
import os
import json

logger = logging.getLogger(__name__)


class CompilerConfig:
    """Configuration manager for ML compiler settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._get_default_config()
        if config_file:
            self.load_config(config_file)
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default compiler configuration."""
        return {
            'optimization': {
                'level': 2,
                'enable_fusion': True,
                'enable_memory_optimization': True,
                'enable_constant_folding': True,
                'enable_dead_code_elimination': True
            },
            'compilation': {
                'target_backend': 'cpu',
                'precision': 'float32',
                'batch_size': 1,
                'input_shape': [1, 3, 224, 224]
            },
            'debug': {
                'enable_profiling': False,
                'verbose_logging': False,
                'save_intermediate_ir': False
            },
            'performance': {
                'num_benchmark_runs': 10,
                'warmup_runs': 3,
                'enable_auto_tuning': False
            }
        }
        
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            self._merge_config(file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config file {config_file}: {e}")
            
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config."""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
                    
        merge_dict(self.config, new_config)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'optimization.level')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
            
        config_ref[keys[-1]] = value
        
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization-specific configuration."""
        return self.config.get('optimization', {})
        
    def get_compilation_config(self) -> Dict[str, Any]:
        """Get compilation-specific configuration."""
        return self.config.get('compilation', {})


class BackendConfig:
    """Configuration manager for hardware backend settings."""
    
    def __init__(self, backend_type: str = "cpu"):
        self.backend_type = backend_type
        self.config = self._get_backend_config(backend_type)
        
    def _get_backend_config(self, backend_type: str) -> Dict[str, Any]:
        """Get configuration for specific backend type."""
        configs = {
            'cpu': {
                'optimization_level': 3,
                'vectorization': True,
                'target_arch': 'x86_64',
                'num_threads': os.cpu_count() or 4,
                'memory_pool_size_mb': 1024
            },
            'cuda': {
                'device_id': 0,
                'compute_capability': '7.5',
                'memory_pool_size_mb': 8192,
                'kernel_fusion': True,
                'tensor_core_usage': True
            },
            'opencl': {
                'platform_id': 0,
                'device_id': 0,
                'memory_pool_size_mb': 4096,
                'work_group_size': 256
            },
            'cerebras': {
                'cores': 400000,
                'memory_per_core_kb': 48,
                'dataflow_optimization': True,
                'sparsity_support': True
            }
        }
        
        return configs.get(backend_type, configs['cpu'])
        
    def get_backend_specific_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        return self.config
        
    def update_config(self, updates: Dict[str, Any]):
        """Update backend configuration."""
        self.config.update(updates)
        logger.info(f"Updated {self.backend_type} backend configuration") 