# Memetic Feedback Loop (MFL)
import json

user_feedback_data = []

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
        "Adaptation": adaptation_data
    }
    user_feedback_data.append(feedback_entry)

    # Simulate storing data persistently
    with open("data/user_feedback.json", "w") as file:
        json.dump(user_feedback_data, file, indent=4)
