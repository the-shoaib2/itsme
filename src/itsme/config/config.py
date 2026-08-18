"""
Configuration Management Module for ItsMe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from itsme.utils.exceptions import ConfigurationError

# Auto load .env if present
load_dotenv()

def load_yaml(config_path: str) -> dict[str, Any]:
    """Load YAML config file with error handling."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML config {config_path}: {e}")

class Config:
    """
    Unified Configuration Manager for ItsMe.
    Merges YAML configs with environment variable overrides.
    """
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
            
        self.raw_config = load_yaml(config_path) if Path(config_path).exists() else {}
        self._apply_env_overrides()
        
    def _apply_env_overrides(self):
        """Override configuration with environment variables."""
        env_mappings = {
            "MODEL_NAME": ("model", "name"),
            "DATA_DIR": ("paths", "data_dir"),
            "MODEL_DIR": ("paths", "models_dir"),
            "RUNS_DIR": ("paths", "runs_dir"),
            "SAMPLE_RATE": ("model", "sample_rate"),
            "SPEAKER_ID": ("model", "speaker_id"),
            "WHISPER_MODEL": ("transcription", "model"),
            "DEVICE": ("hardware", "device"),
        }
        
        for env_key, keys in env_mappings.items():
            val = os.getenv(env_key)
            if val is not None:
                curr = self.raw_config
                for key in keys[:-1]:
                    if key not in curr:
                        curr[key] = {}
                    curr = curr[key]
                # Cast numeric types if appropriate
                if val.isdigit():
                    val = int(val)
                curr[keys[-1]] = val

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Access nested config keys using dot notation, e.g. config.get('model.name')
        """
        keys = key_path.split(".")
        curr = self.raw_config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def to_dict(self) -> dict[str, Any]:
        return self.raw_config

def get_config(config_path: str | None = None) -> Config:
    return Config(config_path)
