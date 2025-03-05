#!/usr/bin/env python3
"""
Test script for TinyLlama integration with the Adaptive UI Framework.
This script tests the TinyLlama model's performance with the framework.
"""

import sys
import json
from modules.ai_lmfl import generate_adaptive_response, get_model_info

def main():
    """Main function to test the TinyLlama model."""
    print("=" * 80)
    print("Testing TinyLlama Integration with Adaptive UI Framework")
    print("=" * 80)
    
    # Get model information
    model_info = get_model_info()
    print("\nModel Information:")
    print(json.dumps(model_info, indent=2))
    
    # Test different context settings
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
        },
        {
            "Layout": "Tablet",
            "Theme": "High Contrast",
            "Touch Optimization": "Enabled"
        }
    ]
    
    # Test queries
    test_queries = [
        "What is adaptive UI?",
        "How does the framework handle user preferences?",
        "Tell me about sentiment analysis in your system",
        "Can you explain the Memetic Feedback Loop?",
        "How is AI integrated into the user interface?"
    ]
    
    # Run tests with different contexts and queries
    for i, context in enumerate(test_contexts):
        print(f"\n\n--- Testing Context {i+1} ---")
        print(f"Context: {json.dumps(context, indent=2)}")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            try:
                response = generate_adaptive_response(query, context)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error generating response: {e}")
                
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)

if __name__ == "__main__":
    main() 