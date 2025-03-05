#!/usr/bin/env python3
"""
Quick test for Phi-2 model integration with Adaptive UI Framework.
"""

from modules.ai_lmfl import generate_adaptive_response, get_model_info

def main():
    """Test function to check if Phi-2 is working."""
    print("=" * 80)
    print("Testing Phi-2 with Adaptive UI Framework")
    print("=" * 80)
    
    # Get model info
    model_info = get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test a simple query
    test_context = {
        "Layout": "Desktop",
        "Theme": "Light Mode", 
        "Touch Optimization": "Disabled"
    }
    
    test_query = "What is adaptive UI?"
    
    print(f"\nQuery: {test_query}")
    print("Generating response...")
    
    response = generate_adaptive_response(test_query, test_context)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main() 