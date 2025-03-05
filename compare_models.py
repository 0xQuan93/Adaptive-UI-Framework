#!/usr/bin/env python3
"""
Compare different language models for the Adaptive UI Framework.
This script allows testing different models and checking their performance.
"""

import os
import json
import time
import argparse

def modify_config(model_name):
    """Update the config.json file with the specified model."""
    config_path = 'config.json'
    
    # Read the current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update the model
    config['ai_settings']['ai_model'] = model_name
    
    # Write back the updated config
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    print(f"Updated config to use model: {model_name}")

def test_model(model_name):
    """Test a specific model and measure performance."""
    # First, modify the config
    modify_config(model_name)
    
    # Clear any cached imports
    if 'modules.ai_lmfl' in sys.modules:
        del sys.modules['modules.ai_lmfl']
    
    # Import the module fresh
    start_time = time.time()
    from modules.ai_lmfl import generate_adaptive_response, get_model_info
    load_time = time.time() - start_time
    
    # Get and print model info
    model_info = get_model_info()
    print("\nModel Information:")
    print(json.dumps(model_info, indent=2))
    print(f"Model load time: {load_time:.2f} seconds")
    
    # Test with a simple query
    test_context = {
        "Layout": "Desktop",
        "Theme": "Light Mode", 
        "Touch Optimization": "Disabled"
    }
    
    test_query = "What is adaptive UI and how does it work?"
    
    # Generate a response and measure time
    start_time = time.time()
    response = generate_adaptive_response(test_query, test_context)
    generation_time = time.time() - start_time
    
    print(f"\nQuery: {test_query}")
    print(f"Response: {response}")
    print(f"Generation time: {generation_time:.2f} seconds")
    
    return {
        "model": model_name,
        "info": model_info,
        "load_time": load_time,
        "generation_time": generation_time,
        "response": response
    }

def main():
    """Main function to run the model comparison."""
    parser = argparse.ArgumentParser(description='Compare language models for the Adaptive UI Framework')
    parser.add_argument('--models', nargs='+', default=["gpt2", "distilgpt2"],
                        help='List of models to test (default: gpt2 distilgpt2)')
    
    args = parser.parse_args()
    models = args.models
    
    print("=" * 80)
    print("Language Model Comparison for Adaptive UI Framework")
    print("=" * 80)
    
    results = []
    
    for model in models:
        print(f"\n\nTesting model: {model}")
        print("-" * 40)
        
        try:
            result = test_model(model)
            results.append(result)
        except Exception as e:
            print(f"Error testing model {model}: {e}")
    
    # Print comparison
    print("\n\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    
    # Table header
    print(f"{'Model':30} | {'Load Time':10} | {'Generation Time':15} | {'Type':15}")
    print("-" * 80)
    
    for result in results:
        model_type = result['info'].get('is_mock', True) and "Mock" or (
            result['info'].get('is_small_model', False) and "Small LLM" or "Standard"
        )
        print(f"{result['model']:30} | {result['load_time']:.2f}s      | {result['generation_time']:.2f}s            | {model_type:15}")
    
    # Return to the default model
    modify_config("gpt2")

if __name__ == "__main__":
    import sys
    main() 