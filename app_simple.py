from flask import Flask, request, jsonify
import json
import os
import time

app = Flask(__name__)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Mock AI response function
def mock_generate_response(user_input, context_data):
    """
    Generate a mock AI response based on user input and context.
    """
    # Simulate processing time
    time.sleep(0.5)
    
    # Create a context-aware response
    layout = context_data.get('Layout', 'Standard')
    theme = context_data.get('Theme', 'Standard')
    touch = context_data.get('Touch Optimization', 'Standard')
    
    # Basic keyword matching for responses
    if 'adaptive ui' in user_input.lower():
        response = f"Adaptive UI is a framework that adjusts the interface based on user context. "
        response += f"For your {layout} device with {theme}, we optimize the experience accordingly."
        
        if 'Enabled' in touch:
            response += " Your touch optimization is enabled for better mobile interaction."
    elif 'sentiment' in user_input.lower() or 'analysis' in user_input.lower():
        response = f"Our sentiment analysis module helps detect user emotions and adapts the UI accordingly. "
        response += f"This works well in your {layout} environment with {theme}."
    else:
        response = f"I'm your adaptive UI assistant. I notice you're using a {layout} device with {theme}. "
        response += f"How can I help you with your adaptive interface today?"
    
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "message": "Simple API server is running",
        "implementation": "mock"
    })

@app.route('/api/response', methods=['POST'])
def get_ai_response():
    """Generate mock AI response based on user input and context"""
    try:
        data = request.json
        print(f"[INFO] Received request with data: {json.dumps(data) if data else 'None'}")
        
        if not data or 'input' not in data or 'context' not in data:
            return jsonify({"error": "Missing 'input' or 'context' field in request"}), 400
        
        user_input = data['input']
        context = data['context']
        
        print(f"[INFO] Processing request with input: {user_input}")
        print(f"[INFO] Context: {json.dumps(context)}")
        
        start_time = time.time()
        response = mock_generate_response(user_input, context)
        elapsed_time = time.time() - start_time
        
        print(f"[INFO] Generated response in {elapsed_time:.2f} seconds")
        print(f"[INFO] Response: {response}")
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == '__main__':
    print("\n=== Simple API Server (Mock Implementation) ===")
    print("Starting Flask server on port 5000...")
    app.run(debug=True, host='0.0.0.0', port=5000) 