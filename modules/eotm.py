# Engagement Optimization & Trust Module (EOTM)
import json
from datetime import datetime

def calculate_user_trust_score(feedback_data):
    '''
    Calculates a trust score for AI adaptation based on user feedback trends.
    :param feedback_data: List of feedback entries.
    :return: Trust score (0-100).
    '''
    if not feedback_data:
        return 0
        
    positive_feedback_count = sum(1 for entry in feedback_data if entry["Sentiment"]["Sentiment"] == "Positive")
    total_feedback = len(feedback_data)
    
    return round((positive_feedback_count / total_feedback) * 100, 2) if total_feedback > 0 else 0

def get_trust_metrics(feedback_data):
    '''
    Provides detailed trust metrics based on user feedback.
    :param feedback_data: List of feedback entries.
    :return: Dictionary with trust metrics.
    '''
    if not feedback_data:
        return {
            "trust_score": 0,
            "feedback_count": 0,
            "trust_trend": "neutral",
            "last_updated": datetime.now().isoformat()
        }
    
    # Calculate sentiment distribution
    sentiment_counts = {
        "Positive": sum(1 for entry in feedback_data if entry["Sentiment"]["Sentiment"] == "Positive"),
        "Neutral": sum(1 for entry in feedback_data if entry["Sentiment"]["Sentiment"] == "Neutral"),
        "Negative": sum(1 for entry in feedback_data if entry["Sentiment"]["Sentiment"] == "Negative")
    }
    
    total = len(feedback_data)
    trust_score = round((sentiment_counts["Positive"] / total) * 100, 2) if total > 0 else 0
    
    # Determine trust trend (if we have enough data)
    trust_trend = "neutral"
    if len(feedback_data) >= 5:
        recent_data = feedback_data[-5:]
        older_data = feedback_data[:-5]
        
        if older_data:  # Only calculate trend if we have older data to compare
            recent_positive = sum(1 for entry in recent_data if entry["Sentiment"]["Sentiment"] == "Positive")
            recent_score = (recent_positive / len(recent_data)) * 100
            
            older_positive = sum(1 for entry in older_data if entry["Sentiment"]["Sentiment"] == "Positive")
            older_score = (older_positive / len(older_data)) * 100
            
            if recent_score > older_score + 5:
                trust_trend = "improving"
            elif recent_score < older_score - 5:
                trust_trend = "declining"
    
    return {
        "trust_score": trust_score,
        "feedback_count": total,
        "sentiment_distribution": sentiment_counts,
        "trust_trend": trust_trend,
        "last_updated": datetime.now().isoformat()
    }

def generate_transparency_report():
    '''
    Generates a transparency report for the AI system.
    :return: Dictionary with transparency metrics.
    '''
    return {
        "ai_decisions_explained": True,
        "user_data_collection": [
            "User input text",
            "Sentiment analysis",
            "UI adaptation preferences"
        ],
        "data_retention_policy": "30 days",
        "user_controls_available": [
            "Opt out of adaptation",
            "Delete feedback history",
            "Customize UI preferences"
        ],
        "report_generated": datetime.now().isoformat()
    }
