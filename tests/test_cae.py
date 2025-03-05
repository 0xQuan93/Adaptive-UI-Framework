import pytest
from modules.cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation

def test_detect_user_context():
    context = detect_user_context()
    assert "Device Type" in context
    assert "Operating System" in context
    assert isinstance(context["Device Type"], str)
    assert isinstance(context["Operating System"], str)

def test_detect_ui_preferences():
    preferences = detect_ui_preferences()
    assert "Dark Mode Enabled" in preferences
    assert isinstance(preferences["Dark Mode Enabled"], bool)
    assert "Preferred Font Size" in preferences
    assert isinstance(preferences["Preferred Font Size"], str)

def test_generate_ui_adaptation():
    context = {"Device Type": "Desktop", "Operating System": "Windows"}
    preferences = {
        "Dark Mode Enabled": True, 
        "Preferred Font Size": "Large",
        "High Contrast Mode": False
    }
    adaptation = generate_ui_adaptation(context, preferences)
    assert adaptation["Theme"] == "Dark Mode"
    assert adaptation["Layout"] == "Desktop Adaptive Layout"
    assert adaptation["Font Size"] == "Large" 