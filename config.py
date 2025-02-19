"""
Configuration management module for SafeContext library.

This module handles loading and managing configuration settings for the SafeContext
library, including directive classification thresholds, embedding model parameters,
and other configurable settings. Settings can be loaded from environment variables
or configuration files, with sensible defaults provided.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration settings for the embedding model."""
    model_name: str = "text-embedding-ada-002"
    max_tokens: int = 8191
    batch_size: int = 100
    cache_dir: str = "embeddings"

@dataclass
class ClassificationConfig:
    """Configuration settings for directive classification."""
    confidence_threshold: float = 0.85
    max_context_length: int = 1000
    min_directive_length: int = 10

@dataclass
class PreprocessConfig:
    """Configuration settings for document preprocessing."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    supported_formats: tuple = ('txt', 'pdf', 'docx', 'md')

@dataclass
class SafeContextConfig:
    """Main configuration class for SafeContext library."""
    model: ModelConfig = ModelConfig()
    classification: ClassificationConfig = ClassificationConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    log_level: str = "INFO"
    enable_cache: bool = True
    cache_dir: str = ".safecontext_cache"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SafeContextConfig':
        """Create a config instance from a dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        classification_config = ClassificationConfig(**config_dict.get('classification', {}))
        preprocess_config = PreprocessConfig(**config_dict.get('preprocess', {}))
        
        return cls(
            model=model_config,
            classification=classification_config,
            preprocess=preprocess_config,
            log_level=config_dict.get('log_level', cls.log_level),
            enable_cache=config_dict.get('enable_cache', cls.enable_cache),
            cache_dir=config_dict.get('cache_dir', cls.cache_dir)
        )

def load_config(config_path: Optional[str] = None) -> SafeContextConfig:
    """
    Load configuration from multiple sources in order of precedence:
    1. Environment variables
    2. Configuration file (if provided)
    3. Default values
    
    Args:
        config_path: Optional path to a JSON configuration file
    
    Returns:
        SafeContextConfig object with merged configuration settings
    """
    # Start with default config
    config_dict = {}
    
    # Load from config file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict.update(json.load(f))
    
    # Override with environment variables
    env_prefix = "SAFECONTEXT_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Convert environment variable names to config structure
            config_key = key[len(env_prefix):].lower()
            if '_' in config_key:
                category, setting = config_key.split('_', 1)
                if category not in config_dict:
                    config_dict[category] = {}
                config_dict[category][setting] = _convert_value(value)
            else:
                config_dict[config_key] = _convert_value(value)
    
    return SafeContextConfig.from_dict(config_dict)

def _convert_value(value: str) -> Any:
    """Convert string values from environment variables to appropriate types."""
    # Try to convert to boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Try to convert to number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        # Try to convert to tuple if comma-separated
        if ',' in value:
            return tuple(v.strip() for v in value.split(','))
        return value

# Global config instance
config = load_config()
