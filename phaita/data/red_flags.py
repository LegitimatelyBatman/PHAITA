"""Curated red-flag guidance for respiratory conditions.

Each entry maps an ICD-10 respiratory condition code to symptoms that
should trigger escalation and a short piece of clinical advice for the
triage assistant to surface to users.

Red flag definitions are loaded from:
- New: config/medical_knowledge.yaml (consolidated medical knowledge)
- Legacy: config/red_flags.yaml (backward compatibility)

Clinicians can update definitions by editing the appropriate YAML file.
"""

from pathlib import Path
from typing import Dict, List, Union
import yaml


def _load_red_flags_from_yaml() -> Dict[str, Dict[str, Union[List[str], str]]]:
    """Load red flag definitions from YAML configuration file.
    
    Tries to load from medical_knowledge.yaml first (new structure),
    falls back to red_flags.yaml (legacy).
    
    Returns:
        Dictionary mapping ICD-10 codes to red flag data with 'symptoms' and 'escalation' keys.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Try new consolidated config first
    medical_knowledge_path = project_root / "config" / "medical_knowledge.yaml"
    if medical_knowledge_path.exists():
        try:
            with open(medical_knowledge_path, "r", encoding="utf-8") as f:
                medical_config = yaml.safe_load(f) or {}
            if 'red_flags' in medical_config:
                return medical_config['red_flags']
        except (yaml.YAMLError, KeyError) as e:
            print(f"Warning: Failed to load red flags from medical_knowledge.yaml: {e}")
    
    # Fall back to legacy red_flags.yaml
    legacy_path = project_root / "config" / "red_flags.yaml"
    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            red_flags_data = yaml.safe_load(f) or {}
        return red_flags_data
    except FileNotFoundError:
        # Return empty dict if config file doesn't exist
        return {}
    except yaml.YAMLError as e:
        # Log error and return empty dict
        print(f"Warning: Failed to load red flags from {legacy_path}: {e}")
        return {}


# Load red flags from YAML configuration
RESPIRATORY_RED_FLAGS: Dict[str, Dict[str, Union[List[str], str]]] = _load_red_flags_from_yaml()


__all__ = ["RESPIRATORY_RED_FLAGS"]
