# Adaptive UI Framework - Main Entry Point
import json
import os
import sys

from modules.sim import analyze_sentiment_vader
from modules.cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation
from modules.mfl import store_user_feedback, analyze_adaptation_effectiveness, get_adaptation_recommendations
from modules.ai_lmfl import generate_adaptive_response, get_model_info, load_config
from modules.eotm import calculate_user_trust_score, get_trust_metrics, generate_transparency_report

def load_config_file():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: Could not load config.json. Using default settings.")
        return {}

def main():
    """Main function to run the Adaptive UI Framework"""
    # Load configuration
    config = load_config_file()
    
    # Print welcome message
    print("\n" + "="*50)
    print("Adaptive UI Framework - Sentient AI Integration")
    print("="*50 + "\n")
    
    # Initialize system context
    print("Initializing system...")
    context_data = detect_user_context()
    ui_preferences = detect_ui_preferences()
    ui_adaptations = generate_ui_adaptation(context_data, ui_preferences)
    
    # Display system information
    print("\nSystem Context Detected:")
    print(f"  Device Type: {context_data['Device Type']}")
    print(f"  Operating System: {context_data['Operating System']} {context_data['OS Version']}")
    
    print("\nUI Preferences:")
    for key, value in ui_preferences.items():
        print(f"  {key}: {value}")
    
    print("\nGenerated UI Adaptations:")
    for key, value in ui_adaptations.items():
        print(f"  {key}: {value}")
    
    # Display AI model information
    model_info = get_model_info()
    print("\nAI Model Information:")
    if model_info["status"] == "loaded":
        print(f"  Model: {model_info['model']}")
        print(f"  Max Response Length: {model_info['max_length']}")
    else:
        print("  Warning: AI model could not be loaded. Using fallback responses.")
    
    # Interactive loop
    print("\n" + "-"*50)
    print("Interactive Mode - Type 'exit' to quit")
    print("-"*50 + "\n")
    
    while True:
        # Get user input
        user_input = input("\nEnter a message for AI interaction: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nExiting Adaptive UI Framework. Goodbye!")
            break
        
        # Process user input
        sentiment_data = analyze_sentiment_vader(user_input)
        ai_response = generate_adaptive_response(user_input, ui_adaptations)
        
        # Store user feedback
        store_user_feedback(user_input, sentiment_data, ui_adaptations)
        
        # Display results
        print(f"\nSentiment Analysis: {sentiment_data['Sentiment']} (Confidence: {sentiment_data['Confidence']})")
        print(f"AI Response: {ai_response}")
        
        # Calculate and display user trust metrics
        trust_metrics = get_trust_metrics([{"Sentiment": sentiment_data}])  # Using current interaction for demo
        print(f"\nUser Trust Score: {trust_metrics['trust_score']}%")
        
        # Show adaptation recommendations periodically
        if len(user_input) % 3 == 0:  # Just a simple trigger for demo purposes
            recommendations = get_adaptation_recommendations()
            print("\nAdaptation Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
