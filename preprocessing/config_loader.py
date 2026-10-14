"""
Small shared helper for loading config/config.yaml from anywhere in the project.
"""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path = None) -> dict:
    """Load config/config.yaml (or a given path) and return it as a dict."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
