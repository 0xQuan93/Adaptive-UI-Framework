from flask import Flask, request, jsonify
import json
import os

# Import modules from the Adaptive UI Framework
from modules.sim import analyze_sentiment_vader
from modules.cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation
from modules.mfl import store_user_feedback
from modules.ai_lmfl import generate_adaptive_response
from modules.eotm import calculate_user_trust_score

app = Flask(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize user_feedback.json if it doesn't exist
if not os.path.exists('data/user_feedback.json'):
    with open('data/user_feedback.json', 'w') as f:
        json.dump([], f)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of user input"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    result = analyze_sentiment_vader(data['text'])
    return jsonify(result)

@app.route('/api/context', methods=['GET'])
def get_context():
    """Get user context information"""
    context = detect_user_context()
    return jsonify(context)

@app.route('/api/preferences', methods=['GET'])
def get_preferences():
    """Get user UI preferences"""
    preferences = detect_ui_preferences()
    return jsonify(preferences)

@app.route('/api/adaptation', methods=['POST'])
def get_adaptation():
    """Generate UI adaptation based on context and preferences"""
    data = request.json
    if not data or 'context' not in data or 'preferences' not in data:
        return jsonify({"error": "Missing 'context' or 'preferences' field in request"}), 400
    
    adaptation = generate_ui_adaptation(data['context'], data['preferences'])
    return jsonify(adaptation)

@app.route('/api/response', methods=['POST'])
def get_ai_response():
    """Generate AI response based on user input and context"""
    data = request.json
    if not data or 'input' not in data or 'context' not in data:
        return jsonify({"error": "Missing 'input' or 'context' field in request"}), 400
    
    response = generate_adaptive_response(data['input'], data['context'])
    return jsonify({"response": response})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Store user feedback"""
    data = request.json
    if not data or 'input' not in data or 'sentiment' not in data or 'adaptation' not in data:
        return jsonify({"error": "Missing required fields in request"}), 400
    
    store_user_feedback(data['input'], data['sentiment'], data['adaptation'])
    return jsonify({"status": "success"})

@app.route('/api/trust', methods=['POST'])
def get_trust_score():
    """Calculate trust score based on feedback data"""
    data = request.json
    if not data or 'feedback' not in data:
        return jsonify({"error": "Missing 'feedback' field in request"}), 400
    
    trust_score = calculate_user_trust_score(data['feedback'])
    return jsonify({"trust_score": trust_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 