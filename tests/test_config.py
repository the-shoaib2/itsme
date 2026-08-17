"""
Unit Tests for Config Manager and Environment Merging.
"""

from itsme.config.config import Config


def test_config_defaults():
    cfg = Config("configs/config.yaml")
    assert cfg.get("model.sample_rate") == 24000
    assert cfg.get("project.speaker") == "itsme"

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "TestModel/CosyVoice-Test")
    monkeypatch.setenv("SAMPLE_RATE", "48000")
    
    cfg = Config("configs/config.yaml")
    assert cfg.get("model.name") == "TestModel/CosyVoice-Test"
    assert cfg.get("model.sample_rate") == 48000
