# Adaptive UI Framework - Main Entry Point

from sim import analyze_sentiment_vader
from cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation
from mfl import store_user_feedback
from ai_lmfl import generate_adaptive_response
from eotm import calculate_user_trust_score

# Initialize system context
context_data = detect_user_context()
ui_preferences = detect_ui_preferences()
ui_adaptations = generate_ui_adaptation(context_data, ui_preferences)

print("Adaptive UI System Initialized...")
print(f"Detected Context: {context_data}")
print(f"UI Preferences: {ui_preferences}")
print(f"Generated UI Adaptations: {ui_adaptations}")

# Simulating a user interaction
user_input = input("Enter a message for AI interaction: ")
sentiment_data = analyze_sentiment_vader(user_input)
ai_response = generate_adaptive_response(user_input, ui_adaptations)

# Store user feedback
store_user_feedback(user_input, sentiment_data, ui_adaptations)

print(f"Sentiment Analysis: {sentiment_data}")
print(f"AI Response: {ai_response}")

# Calculate and display user trust score
trust_score = calculate_user_trust_score([
    {"Sentiment": sentiment_data}  # Simulated user feedback history
])
print(f"User Trust Score: {trust_score}%")
