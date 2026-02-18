"""
Configuration loader.

Reads config.yaml from the project root. Falls back to sensible
defaults if the file is missing or a key is absent — the app always
starts even without a config file.
"""

from pathlib import Path

import yaml

from src.logger import get_logger

log = get_logger("config")

# Look for config.yaml next to the project root (where main.py lives)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_DEFAULTS = {
    "moondream": {
        "endpoint": "http://localhost:2020/v1",
    },
    "ollama": {
        "url": "http://localhost:11434",
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "temperature": 0.1,
        "num_predict": 256,
        "timeout": 30,
    },
    "app": {
        "host": "0.0.0.0",
        "port": 7860,
        "theme": "hmb/amethyst",
        "share": False,
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Merge overrides into defaults, preserving nested structure."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict:
    """
    Load configuration from YAML file, merged with defaults.

    Args:
        path: Optional override path to the config file.

    Returns:
        Complete config dict with all keys guaranteed present.
    """
    config_path = path or _CONFIG_PATH

    if config_path.exists():
        try:
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            log.info(f"Loaded config from {config_path}")
            return _deep_merge(_DEFAULTS, user_config)
        except Exception as e:
            log.warning(f"Failed to read {config_path}: {e} — using defaults")
            return _DEFAULTS.copy()
    else:
        log.info("No config.yaml found — using defaults")
        return _DEFAULTS.copy()


# Module-level singleton so all imports share the same config
config = load_config()
