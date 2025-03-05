import pytest
import json
import os
from modules.mfl import store_user_feedback, analyze_adaptation_effectiveness, get_adaptation_recommendations, reset_feedback_data

# Setup and teardown for tests
@pytest.fixture
def setup_test_data():
    # Reset the feedback data before each test
    reset_feedback_data()
    yield
    # Reset again after the test
    reset_feedback_data()

def test_store_user_feedback(setup_test_data):
    user_input = "I like this feature."
    sentiment_data = {"Sentiment": "Positive", "Confidence": 0.9}
    adaptation_data = {"Theme": "Dark Mode"}

    store_user_feedback(user_input, sentiment_data, adaptation_data)
    
    with open("data/user_feedback.json", "r") as file:
        feedback_list = json.load(file)

    assert len(feedback_list) > 0
    assert feedback_list[-1]["Sentiment"]["Sentiment"] == "Positive"
    assert feedback_list[-1]["User Input"] == "I like this feature."

def test_analyze_adaptation_effectiveness(setup_test_data):
    # Store some test feedback data
    store_user_feedback("I love this!", {"Sentiment": "Positive", "Confidence": 0.9}, {})
    store_user_feedback("This is okay", {"Sentiment": "Neutral", "Confidence": 0.1}, {})
    store_user_feedback("I hate this", {"Sentiment": "Negative", "Confidence": -0.8}, {})
    
    effectiveness = analyze_adaptation_effectiveness()
    
    assert "effectiveness" in effectiveness
    assert "sample_size" in effectiveness
    assert effectiveness["sample_size"] == 3
    # One out of three entries is positive, so effectiveness should be ~33.33%
    assert 30 <= effectiveness["effectiveness"] <= 35

def test_get_adaptation_recommendations(setup_test_data):
    # Test with insufficient data (empty)
    recommendations = get_adaptation_recommendations()
    assert "Not enough feedback data" in recommendations[0]
    
    # Add some data but not enough (less than 3)
    store_user_feedback("I love this!", {"Sentiment": "Positive", "Confidence": 0.9}, {})
    recommendations = get_adaptation_recommendations()
    assert "Not enough feedback data" in recommendations[0]
    
    # Add more test data to have at least 3 entries
    store_user_feedback("I hate this", {"Sentiment": "Negative", "Confidence": -0.8}, {})
    store_user_feedback("This is terrible", {"Sentiment": "Negative", "Confidence": -0.9}, {})
    
    recommendations = get_adaptation_recommendations()
    assert len(recommendations) > 0
    # The data has 1 positive and 2 negative, so should suggest revising UI
    assert "Consider revising UI adaptations" in recommendations[0] 