import pytest
from modules.eotm import calculate_user_trust_score, get_trust_metrics, generate_transparency_report

def test_trust_score_calculation():
    # Test with empty data
    assert calculate_user_trust_score([]) == 0
    
    # Test with mixed feedback
    feedback_data = [
        {"Sentiment": {"Sentiment": "Positive"}},
        {"Sentiment": {"Sentiment": "Positive"}},
        {"Sentiment": {"Sentiment": "Negative"}}
    ]
    trust_score = calculate_user_trust_score(feedback_data)
    assert trust_score == 66.67  # 2 out of 3 positive = 66.67%
    
    # Test with all positive
    all_positive = [{"Sentiment": {"Sentiment": "Positive"}} for _ in range(5)]
    assert calculate_user_trust_score(all_positive) == 100.0
    
    # Test with all negative
    all_negative = [{"Sentiment": {"Sentiment": "Negative"}} for _ in range(5)]
    assert calculate_user_trust_score(all_negative) == 0.0

def test_get_trust_metrics():
    # Test with empty data
    metrics = get_trust_metrics([])
    assert metrics["trust_score"] == 0
    assert metrics["feedback_count"] == 0
    assert metrics["trust_trend"] == "neutral"
    
    # Test with some data
    feedback_data = [
        {"Sentiment": {"Sentiment": "Positive"}},
        {"Sentiment": {"Sentiment": "Negative"}},
        {"Sentiment": {"Sentiment": "Neutral"}}
    ]
    metrics = get_trust_metrics(feedback_data)
    assert metrics["trust_score"] == 33.33  # 1 out of 3 positive
    assert metrics["feedback_count"] == 3
    assert "sentiment_distribution" in metrics
    assert metrics["sentiment_distribution"]["Positive"] == 1
    assert metrics["sentiment_distribution"]["Negative"] == 1
    assert metrics["sentiment_distribution"]["Neutral"] == 1

def test_generate_transparency_report():
    report = generate_transparency_report()
    assert "ai_decisions_explained" in report
    assert "user_data_collection" in report
    assert "data_retention_policy" in report
    assert "user_controls_available" in report
    assert "report_generated" in report
    assert isinstance(report["user_data_collection"], list)
    assert len(report["user_data_collection"]) > 0 