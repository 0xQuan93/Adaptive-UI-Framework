#!/usr/bin/env python3
"""
Test the debug version of the Adaptive UI Framework API.
"""

import requests
import json
import time

def test_api(input_text, context):
    """
    Test the API endpoint with the given input and context.
    """
    url = "http://localhost:5000/api/response"
    
    payload = {
        "input": input_text,
        "context": context
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {url} with payload: {json.dumps(payload)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        elapsed_time = time.time() - start_time
        
        print(f"Got response in {elapsed_time:.2f} seconds with status code: {response.status_code}")
        
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
    except requests.exceptions.RequestException as e:
        print(f"Request failed with error: {str(e)}")
        return {
            "success": False,
            "message": f"Request failed: {str(e)}",
            "time": 0
        }

def test_health():
    """Test the health check endpoint."""
    url = "http://localhost:5000/api/health"
    
    print(f"Checking server health at {url}")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"Health check successful: {response.json()}")
            return True
        else:
            print(f"Health check failed with status code {response.status_code}: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Health check failed with error: {str(e)}")
        return False

def main():
    """Main function to test the API."""
    print("=" * 80)
    print("Testing Debug Flask Server (No Model Loading)")
    print("=" * 80)
    
    # First check if the server is healthy
    if not test_health():
        print("Server health check failed. Make sure the debug server is running.")
        return
    
    # Simple test context
    test_context = {
        "Layout": "Desktop",
        "Theme": "Light Mode",
        "Touch Optimization": "Disabled"
    }
    
    # Test queries
    test_queries = [
        "What is adaptive UI?",
        "Simple test query"
    ]
    
    # Run tests
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        result = test_api(query, test_context)
        
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