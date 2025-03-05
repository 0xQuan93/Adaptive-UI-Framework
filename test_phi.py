#!/usr/bin/env python3
"""
Test Microsoft's Phi-2 model integration with the Adaptive UI Framework.
This script tests the model with various queries and contexts.
"""

from modules.ai_lmfl import generate_adaptive_response, get_model_info
import time
import json

def main():
    """Test function for Phi-2 model."""
    print("=" * 80)
    print("Testing Microsoft Phi-2 with Adaptive UI Framework")
    print("=" * 80)
    
    # Get model info
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
    
    # Test different queries
    test_queries = [
        "What is adaptive UI?",
        "How does sentiment analysis work in your framework?",
        "Can you explain the user feedback loop?",
        "What are the technical requirements for this framework?",
        "Tell me about responsive design vs. adaptive design."
    ]
    
    # Test simple queries with different contexts
    print("\n\n=== Testing Simple Queries with Different Contexts ===")
    
    for i, context in enumerate(test_contexts):
        print(f"\n\n--- Context {i+1}: {json.dumps(context, indent=2)} ---")
        
        query = "What is adaptive UI?"
        print(f"\nQuery: {query}")
        start_time = time.time()
        response = generate_adaptive_response(query, context)
        generate_time = time.time() - start_time
        
        print(f"Response ({generate_time:.2f} seconds):")
        print(response)
    
    # Test various queries with a consistent context
    print("\n\n=== Testing Various Queries with Consistent Context ===")
    context = test_contexts[0]  # Use desktop context
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        start_time = time.time()
        response = generate_adaptive_response(query, context)
        generate_time = time.time() - start_time
        
        print(f"Response ({generate_time:.2f} seconds):")
        print(response)
    
    print("\n" + "=" * 80)
    print("Testing completed")
    print("=" * 80)

if __name__ == "__main__":
    main() 