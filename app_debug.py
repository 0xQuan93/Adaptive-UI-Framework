from flask import Flask, request, jsonify
import json
import os

# Import only essential modules or create mock implementations
# No model loading here

app = Flask(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize user_feedback.json if it doesn't exist
if not os.path.exists('data/user_feedback.json'):
    with open('data/user_feedback.json', 'w') as f:
        json.dump([], f)

@app.route('/api/response', methods=['POST'])
def get_ai_response():
    """Generate mock AI response based on user input and context"""
    data = request.json
    if not data or 'input' not in data or 'context' not in data:
        return jsonify({"error": "Missing 'input' or 'context' field in request"}), 400
    
    # Mock response - no model loading
    user_input = data['input']
    context = data['context']
    
    # Generate a simple response for testing
    response = f"Mock response to: '{user_input}'. Context received: {json.dumps(context)}"
    
    print(f"Received request with input: {user_input}")
    print(f"Context: {json.dumps(context)}")
    print(f"Responding with: {response}")
    
    return jsonify({"response": response})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Debug server is running"})

if __name__ == '__main__':
    print("Starting debug Flask server without model loading...")
    app.run(debug=True, host='0.0.0.0', port=5000) 