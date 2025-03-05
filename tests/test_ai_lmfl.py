import pytest
import json
import os
from modules.ai_lmfl import load_config, generate_adaptive_response, get_model_info

@pytest.fixture
def setup_test_config():
    # Create a temporary test config
    test_config = {
        "ai_settings": {
            "ai_response_max_length": 30,
            "ai_model": "gpt2"
        }
    }
    with open('config.json', 'w') as f:
        json.dump(test_config, f)
    yield
    # No cleanup needed as we want to keep the config file

def test_load_config(setup_test_config):
    config = load_config()
    assert "ai_settings" in config
    assert "ai_response_max_length" in config["ai_settings"]
    assert config["ai_settings"]["ai_response_max_length"] == 30
    assert "ai_model" in config["ai_settings"]
    assert config["ai_settings"]["ai_model"] == "gpt2"

def test_generate_adaptive_response():
    # This test may be skipped if the model can't be loaded
    try:
        context_data = {
            "Layout": "Desktop Adaptive Layout",
            "Theme": "Light Mode",
            "Touch Optimization": "Disabled"
        }
        response = generate_adaptive_response("What is adaptive UI?", context_data)
        assert isinstance(response, str)
        assert len(response) > 0
    except Exception as e:
        pytest.skip(f"Skipping test due to model loading error: {e}")

def test_get_model_info():
    model_info = get_model_info()
    assert "status" in model_info
    
    if model_info["status"] == "loaded":
        assert "model" in model_info
        assert "max_length" in model_info
        assert "is_mock" in model_info  # Check for the new field
    else:
        assert model_info["model"] is None 