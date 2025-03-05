from flask import Flask, request, jsonify
import json
import os
import traceback

# Set the AI_DEBUG environment variable
os.environ['AI_DEBUG'] = 'true'

# Import modules from the Adaptive UI Framework
from modules.sim import analyze_sentiment_vader
from modules.cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation
from modules.mfl import store_user_feedback
from modules.ai_lmfl import generate_adaptive_response, get_model_info
from modules.eotm import calculate_user_trust_score

app = Flask(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize user_feedback.json if it doesn't exist
if not os.path.exists('data/user_feedback.json'):
    with open('data/user_feedback.json', 'w') as f:
        json.dump([], f)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/api/response', methods=['POST'])
def get_ai_response():
    """Generate AI response based on user input and context"""
    start_time = None
    try:
        print("[DEBUG] Received API request to /api/response")
        data = request.json
        print(f"[DEBUG] Request data: {json.dumps(data) if data else 'None'}")
        
        if not data or 'input' not in data or 'context' not in data:
            return jsonify({"error": "Missing 'input' or 'context' field in request"}), 400
        
        print(f"[DEBUG] Processing request with input: {data['input']}")
        print(f"[DEBUG] Context: {json.dumps(data['context'])}")
        
        # Generate adaptive response
        response = generate_adaptive_response(data['input'], data['context'])
        
        print(f"[DEBUG] Generated response: {response[:100]}...")
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"[ERROR] Exception in API endpoint: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error", 
            "message": str(e), 
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Print model info before starting server
    print("\n=== Model Information ===")
    model_info = get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print("========================\n")
    
    print("Starting Flask server with debug mode enabled...")
    app.run(debug=True, host='0.0.0.0', port=5001) 