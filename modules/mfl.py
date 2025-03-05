# Memetic Feedback Loop (MFL)
import json
import os
from datetime import datetime

user_feedback_data = []

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load existing feedback data if any
if os.path.exists('data/user_feedback.json'):
    try:
        with open('data/user_feedback.json', 'r') as file:
            user_feedback_data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # Initialize empty list if file is empty or corrupted
        user_feedback_data = []

def reset_feedback_data():
    '''
    Resets the feedback data - mainly used for testing.
    '''
    global user_feedback_data
    user_feedback_data = []
    with open('data/user_feedback.json', 'w') as file:
        json.dump(user_feedback_data, file)

def store_user_feedback(user_input, sentiment_data, adaptation_data):
    '''
    Stores implicit and explicit user feedback.
    :param user_input: User input text.
    :param sentiment_data: Sentiment analysis result.
    :param adaptation_data: UI adaptation recommendation.
    '''
    feedback_entry = {
        "User Input": user_input,
        "Sentiment": sentiment_data,
        "Adaptation": adaptation_data,
        "Timestamp": datetime.now().isoformat()
    }
    user_feedback_data.append(feedback_entry)

    # Store data persistently
    with open("data/user_feedback.json", "w") as file:
        json.dump(user_feedback_data, file, indent=4)
        
def get_user_feedback_history():
    '''
    Retrieves all stored user feedback.
    :return: List of feedback entries.
    '''
    return user_feedback_data

def analyze_adaptation_effectiveness():
    '''
    Analyzes the effectiveness of UI adaptations based on sentiment trends.
    :return: Dictionary with effectiveness metrics.
    '''
    if not user_feedback_data:
        return {"effectiveness": 0, "sample_size": 0}
        
    positive_reactions = sum(1 for entry in user_feedback_data 
                           if entry["Sentiment"]["Sentiment"] == "Positive")
    total_entries = len(user_feedback_data)
    
    return {
        "effectiveness": round((positive_reactions / total_entries) * 100, 2),
        "sample_size": total_entries,
        "positive_reactions": positive_reactions
    }

def get_adaptation_recommendations():
    '''
    Provides recommendations for UI improvements based on feedback analysis.
    :return: List of recommendations.
    '''
    if not user_feedback_data or len(user_feedback_data) < 3:
        return ["Not enough feedback data to generate recommendations"]
        
    # Simple analysis based on sentiment patterns
    sentiment_count = {
        "Positive": sum(1 for entry in user_feedback_data if entry["Sentiment"]["Sentiment"] == "Positive"),
        "Neutral": sum(1 for entry in user_feedback_data if entry["Sentiment"]["Sentiment"] == "Neutral"),
        "Negative": sum(1 for entry in user_feedback_data if entry["Sentiment"]["Sentiment"] == "Negative")
    }
    
    recommendations = []
    
    if sentiment_count["Negative"] > sentiment_count["Positive"]:
        recommendations.append("Consider revising UI adaptations as negative feedback exceeds positive")
        
    if sentiment_count["Neutral"] > (sentiment_count["Positive"] + sentiment_count["Negative"]):
        recommendations.append("UI adaptations may not be impactful enough - increase adaptation visibility")
    
    return recommendations if recommendations else ["Current adaptations are effective based on feedback"]
