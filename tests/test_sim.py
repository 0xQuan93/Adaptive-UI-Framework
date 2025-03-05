import pytest
from modules.sim import analyze_sentiment_vader

def test_positive_sentiment():
    result = analyze_sentiment_vader("I love this UI!")
    assert result["Sentiment"] == "Positive"
    assert result["Confidence"] > 0

def test_negative_sentiment():
    result = analyze_sentiment_vader("I hate using this interface.")
    assert result["Sentiment"] == "Negative"
    assert result["Confidence"] < 0

def test_neutral_sentiment():
    result = analyze_sentiment_vader("This is a user interface")
    assert result["Sentiment"] == "Neutral"
    assert -0.1 < result["Confidence"] < 0.1 