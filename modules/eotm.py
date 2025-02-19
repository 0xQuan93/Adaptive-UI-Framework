# Engagement Optimization & Trust Module (EOTM)

def calculate_user_trust_score(feedback_data):
    '''
    Calculates a trust score for AI adaptation based on user feedback trends.
    :param feedback_data: List of feedback entries.
    :return: Trust score (0-100).
    '''
    positive_feedback_count = sum(1 for entry in feedback_data if entry["Sentiment"]["Sentiment"] == "Positive")
    total_feedback = len(feedback_data)
    
    return round((positive_feedback_count / total_feedback) * 100, 2) if total_feedback > 0 else 0
