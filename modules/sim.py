# Sentient Interaction Model (SIM)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Sentiment Analysis Engine
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(user_input):
    '''
    Analyzes user sentiment using VADER NLP model for conversational AI.
    :param user_input: String input from the user.
    :return: Dictionary with sentiment classification and confidence score.
    '''
    sentiment_scores = vader_analyzer.polarity_scores(user_input)
    sentiment_label = "Positive" if sentiment_scores['compound'] >= 0.05 else "Negative" if sentiment_scores['compound'] <= -0.05 else "Neutral"
    
    return {
        "Sentiment": sentiment_label,
        "Confidence": round(sentiment_scores['compound'], 4)
    }
