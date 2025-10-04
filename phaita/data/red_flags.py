"""Curated red-flag guidance for respiratory conditions.

Each entry maps an ICD-10 respiratory condition code to symptoms that
should trigger escalation and a short piece of clinical advice for the
triage assistant to surface to users.

Red flag definitions are now loaded from config/red_flags.yaml to allow
clinicians to easily update definitions without modifying Python code.
"""

from pathlib import Path
from typing import Dict, List, Union
import yaml


def _load_red_flags_from_yaml() -> Dict[str, Dict[str, Union[List[str], str]]]:
    """Load red flag definitions from YAML configuration file.
    
    Returns:
        Dictionary mapping ICD-10 codes to red flag data with 'symptoms' and 'escalation' keys.
    """
    # Default to config/red_flags.yaml relative to project root
    config_path = Path(__file__).resolve().parents[2] / "config" / "red_flags.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            red_flags_data = yaml.safe_load(f) or {}
        return red_flags_data
    except FileNotFoundError:
        # Return empty dict if config file doesn't exist
        return {}
    except yaml.YAMLError as e:
        # Log error and return empty dict
        print(f"Warning: Failed to load red flags from {config_path}: {e}")
        return {}


# Load red flags from YAML configuration
RESPIRATORY_RED_FLAGS: Dict[str, Dict[str, Union[List[str], str]]] = _load_red_flags_from_yaml()


__all__ = ["RESPIRATORY_RED_FLAGS"]
