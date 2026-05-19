# src/config.py
import copy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str, override_paths: list[str] = None) -> dict:
    with open(base_path) as f:
        config = yaml.safe_load(f)
    for path in (override_paths or []):
        with open(path) as f:
            config = _deep_merge(config, yaml.safe_load(f))
    return config