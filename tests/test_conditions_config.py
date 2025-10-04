from pathlib import Path

import json
from pathlib import Path

import pytest
import yaml

from phaita.data.icd_conditions import RespiratoryConditions
from phaita.data.forum_scraper import ForumDataAugmentation
from phaita.models.generator import SymptomGenerator


@pytest.fixture()
def restore_default_config():
    original_path = RespiratoryConditions.get_config_path()
    try:
        yield original_path
    finally:
        RespiratoryConditions.reload(config_path=original_path)


def _write_config(path: Path, data: dict) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


def _minimal_condition(code: str = "X00.0") -> dict:
    return {
        code: {
            "name": "Test Condition",
            "symptoms": ["mystery_symptom"],
            "severity_indicators": ["scary_indicator"],
            "lay_terms": ["mystery illness"],
            "description": "A configurable test condition",
        }
    }


def test_reload_from_physician_config(tmp_path, restore_default_config):
    config_path = _write_config(tmp_path / "resp.yaml", _minimal_condition())
    hook_payloads = []

    def _hook(data):
        hook_payloads.append(json.loads(json.dumps(data)))

    RespiratoryConditions.register_reload_hook(_hook)
    try:
        RespiratoryConditions.reload(config_path=config_path)
        conditions = RespiratoryConditions.get_all_conditions()
        assert "X00.0" in conditions
        assert conditions["X00.0"]["name"] == "Test Condition"
        vocabulary = RespiratoryConditions.get_vocabulary()
        assert "mystery_symptom" in vocabulary["symptoms"]
        assert hook_payloads, "Reload hook should be invoked"
        assert "X00.0" in hook_payloads[-1]
    finally:
        RespiratoryConditions.unregister_reload_hook(_hook)


def test_invalid_config_raises(tmp_path, restore_default_config):
    invalid_config = {"BAD": {"name": "oops"}}
    config_path = _write_config(tmp_path / "invalid.yaml", invalid_config)
    with pytest.raises(ValueError):
        RespiratoryConditions.reload(config_path=config_path)


def test_forum_augmentation_uses_config_vocabulary(tmp_path, restore_default_config):
    config = _minimal_condition()
    config["X00.0"]["lay_terms"].append("rare phrase")
    config_path = _write_config(tmp_path / "forum.yaml", config)
    RespiratoryConditions.reload(config_path=config_path)
    augmenter = ForumDataAugmentation(conditions=RespiratoryConditions.get_all_conditions())
    complaints = augmenter.get_forum_complaints_for_pretraining(max_complaints=5)
    assert any("rare phrase" in complaint for complaint in complaints)


def test_symptom_generator_reloads_when_config_changes(tmp_path, restore_default_config):
    first_config = _minimal_condition("X00.0")
    first_path = _write_config(tmp_path / "first.yaml", first_config)
    RespiratoryConditions.reload(config_path=first_path)
    generator = SymptomGenerator()
    assert "X00.0" in generator.bayesian_network.conditions

    updated_config = _minimal_condition("X01.0")
    updated_path = _write_config(tmp_path / "updated.yaml", updated_config)
    RespiratoryConditions.reload(config_path=updated_path)
    assert "X01.0" in generator.bayesian_network.conditions
    assert "X00.0" not in generator.bayesian_network.conditions
