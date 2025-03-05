#!/usr/bin/env python3
"""
Test the Adaptive UI Framework API with Phi-2 model.
"""

import requests
import json
import time

def test_api(input_text, context):
    """
    Test the API endpoint with the given input and context.
    
    Args:
        input_text (str): The input text to send to the API
        context (dict): The context information
        
    Returns:
        dict: The API response
    """
    url = "http://localhost:5000/api/response"
    
    payload = {
        "input": input_text,
        "context": context
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers)
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        return {
            "success": True,
            "data": response.json(),
            "time": elapsed_time
        }
    else:
        return {
            "success": False,
            "status_code": response.status_code,
            "message": f"API request failed: {response.text}",
            "time": elapsed_time
        }

def main():
    """Main function to test the API."""
    print("=" * 80)
    print("Testing Adaptive UI Framework API with Phi-2 Model")
    print("=" * 80)
    
    # Define test contexts
    test_contexts = [
        {
            "Layout": "Desktop",
            "Theme": "Light Mode",
            "Touch Optimization": "Disabled"
        },
        {
            "Layout": "Mobile",
            "Theme": "Dark Mode",
            "Touch Optimization": "Enabled"
        }
    ]
    
    # Define test queries
    test_queries = [
        "What is adaptive UI?",
        "How does your framework use sentiment analysis?",
        "Can you explain how your system adapts to user preferences?"
    ]
    
    # Run tests
    for i, context in enumerate(test_contexts):
        print(f"\n\n--- Context {i+1}: {json.dumps(context, indent=2)} ---")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            result = test_api(query, context)
            
            if result["success"]:
                print(f"Response ({result['time']:.2f} seconds):")
                print(result["data"]["response"])
            else:
                print(f"Error ({result['time']:.2f} seconds):")
                print(result["message"])
            
            # Add a small delay between requests
            time.sleep(1)
    
    print("\n" + "=" * 80)
    print("API testing completed")
    print("=" * 80)

if __name__ == "__main__":
    main() 