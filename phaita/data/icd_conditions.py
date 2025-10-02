"""Respiratory condition catalogue that loads from physician-editable config."""

from __future__ import annotations

import copy
import os
import random
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml


class RespiratoryConditions:
    """Runtime-configurable database of respiratory conditions."""

    _CONFIG_ENV_VAR = "PHAITA_RESPIRATORY_CONFIG"
    _DEFAULT_CONFIG_FILENAME = "respiratory_conditions.yaml"

    _lock = threading.RLock()
    _CONDITIONS: Dict[str, Dict] = {}
    _config_path: Path = Path(
        os.getenv(
            _CONFIG_ENV_VAR,
            Path(__file__).resolve().parents[2]
            / "config"
            / _DEFAULT_CONFIG_FILENAME,
        )
    )
    _reload_hooks: List[Callable[[Dict[str, Dict]], None]] = []

    REQUIRED_FIELDS = {"name", "symptoms", "severity_indicators", "lay_terms"}

    @classmethod
    def get_config_path(cls) -> Path:
        """Return the current config path used for loading conditions."""

        return cls._config_path

    @classmethod
    def register_reload_hook(
        cls, callback: Callable[[Dict[str, Dict]], None]
    ) -> None:
        """Register a callback that is invoked whenever the catalogue reloads."""

        with cls._lock:
            if callback not in cls._reload_hooks:
                cls._reload_hooks.append(callback)

    @classmethod
    def unregister_reload_hook(
        cls, callback: Callable[[Dict[str, Dict]], None]
    ) -> None:
        """Remove a previously registered reload callback."""

        with cls._lock:
            if callback in cls._reload_hooks:
                cls._reload_hooks.remove(callback)

    @classmethod
    def _notify_reload(cls) -> None:
        snapshot = copy.deepcopy(cls._CONDITIONS)
        for callback in list(cls._reload_hooks):
            try:
                callback(snapshot)
            except Exception:
                # Hooks should not break reloading pipeline.
                continue

    @classmethod
    def _resolve_config_path(cls, config_path: Optional[Path]) -> Path:
        if config_path is not None:
            return Path(config_path)
        return Path(
            os.getenv(
                cls._CONFIG_ENV_VAR,
                cls._config_path,
            )
        )

    @classmethod
    def _ensure_loaded(cls) -> None:
        with cls._lock:
            if not cls._CONDITIONS:
                cls._load_conditions()

    @classmethod
    def _load_conditions(cls, *, config_path: Optional[Path] = None) -> None:
        path = cls._resolve_config_path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Respiratory condition config not found at '{path}'."
                " Set the PHAITA_RESPIRATORY_CONFIG environment variable or"
                " call RespiratoryConditions.reload(config_path=...) with a valid file."
            )

        with path.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle) or {}

        validated = cls._validate(raw_data)
        cls._CONDITIONS = validated
        cls._config_path = path

    @classmethod
    def _validate(cls, data: Mapping) -> Dict[str, Dict]:
        if not isinstance(data, Mapping):
            raise ValueError("Respiratory condition config must be a mapping of code -> data")

        validated: Dict[str, Dict] = {}
        for code, condition in data.items():
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Condition codes must be non-empty strings")
            if not isinstance(condition, Mapping):
                raise ValueError(f"Condition '{code}' must be a mapping")

            missing = cls.REQUIRED_FIELDS - set(condition.keys())
            if missing:
                raise ValueError(
                    f"Condition '{code}' missing required fields: {sorted(missing)}"
                )

            validated_condition = {
                "name": cls._validate_str(condition["name"], f"{code}.name"),
                "symptoms": cls._validate_str_list(
                    condition["symptoms"], f"{code}.symptoms"
                ),
                "severity_indicators": cls._validate_str_list(
                    condition["severity_indicators"], f"{code}.severity_indicators"
                ),
                "lay_terms": cls._validate_str_list(
                    condition["lay_terms"], f"{code}.lay_terms"
                ),
            }

            if "description" in condition:
                validated_condition["description"] = cls._validate_str(
                    condition["description"], f"{code}.description"
                )

            validated[code] = validated_condition

        if not validated:
            raise ValueError("Respiratory condition config cannot be empty")

        return validated

    @staticmethod
    def _validate_str(value: object, location: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location} must be a non-empty string")
        return value

    @staticmethod
    def _validate_str_list(value: object, location: str) -> List[str]:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError(f"{location} must be a list of strings")
        result: List[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{location} entries must be non-empty strings")
            result.append(item)
        if not result:
            raise ValueError(f"{location} must contain at least one entry")
        return result

    @classmethod
    def reload(cls, *, config_path: Optional[Path] = None) -> Dict[str, Dict]:
        """Reload the respiratory condition catalogue from disk."""

        with cls._lock:
            cls._load_conditions(config_path=config_path)
            cls._notify_reload()
            return copy.deepcopy(cls._CONDITIONS)

    @classmethod
    def get_all_conditions(cls) -> Dict[str, Dict]:
        """
        Get all respiratory conditions.

        Returns:
            Dictionary mapping ICD-10 codes to condition data
        """
        cls._ensure_loaded()
        return copy.deepcopy(cls._CONDITIONS)

    @classmethod
    def get_condition_by_code(cls, code: str) -> Dict:
        """
        Get a specific condition by ICD-10 code.
        
        Args:
            code: ICD-10 code (e.g., 'J45.9')
            
        Returns:
            Condition data dictionary

        Raises:
            KeyError: If code not found
        """
        cls._ensure_loaded()
        if code not in cls._CONDITIONS:
            raise KeyError(f"Condition code '{code}' not found")
        return copy.deepcopy(cls._CONDITIONS[code])

    @classmethod
    def get_random_condition(cls) -> Tuple[str, Dict]:
        """
        Get a random respiratory condition.

        Returns:
            Tuple of (code, condition_data)
        """
        cls._ensure_loaded()
        code = random.choice(list(cls._CONDITIONS.keys()))
        return code, copy.deepcopy(cls._CONDITIONS[code])

    @classmethod
    def get_symptoms_for_condition(cls, code: str) -> List[str]:
        """
        Get all symptoms for a condition.
        
        Args:
            code: ICD-10 code
            
        Returns:
            List of symptoms
        """
        condition = cls.get_condition_by_code(code)
        return condition["symptoms"] + condition["severity_indicators"]
    
    @classmethod
    def get_lay_terms_for_condition(cls, code: str) -> List[str]:
        """
        Get lay language terms for a condition.
        
        Args:
            code: ICD-10 code
            
        Returns:
            List of lay terms
        """
        condition = cls.get_condition_by_code(code)
        return condition["lay_terms"]
    
    @classmethod
    def get_condition_names(cls) -> List[str]:
        """
        Get list of all condition names.
        
        Returns:
            List of condition names
        """
        cls._ensure_loaded()
        return [data["name"] for data in cls._CONDITIONS.values()]

    @classmethod
    def search_by_symptom(cls, symptom: str) -> List[Tuple[str, Dict]]:
        """
        Find conditions that have a specific symptom.
        
        Args:
            symptom: Symptom to search for
            
        Returns:
            List of (code, condition_data) tuples
        """
        cls._ensure_loaded()
        results = []
        for code, data in cls._CONDITIONS.items():
            all_symptoms = data["symptoms"] + data["severity_indicators"]
            if symptom in all_symptoms:
                results.append((code, copy.deepcopy(data)))
        return results

    @classmethod
    def get_vocabulary(cls, conditions: Optional[Dict[str, Dict]] = None) -> Dict[str, List[str]]:
        """Return the canonical vocabulary derived from the condition catalogue."""

        source = conditions or cls.get_all_conditions()
        symptoms: List[str] = []
        severity: List[str] = []
        lay_terms: List[str] = []
        condition_names: List[str] = []

        for data in source.values():
            symptoms.extend(data.get("symptoms", []))
            severity.extend(data.get("severity_indicators", []))
            lay_terms.extend(data.get("lay_terms", []))
            condition_names.append(data.get("name", ""))

        # Deduplicate while preserving order
        def _unique(values: Iterable[str]) -> List[str]:
            seen = set()
            ordered: List[str] = []
            for value in values:
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
            return ordered

        return {
            "symptoms": _unique(symptoms),
            "severity_indicators": _unique(severity),
            "lay_terms": _unique(lay_terms),
            "condition_names": _unique(condition_names),
        }
