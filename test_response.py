from modules.ai_lmfl import generate_adaptive_response, get_model_info

# Print model info
model_info = get_model_info()
print("Model Information:")
for key, value in model_info.items():
    print(f"  {key}: {value}")
print("\n")

# Test response generation
print("Testing adaptive response generation...\n")

# Test 1: Adaptive UI query
context_data = {
    "Layout": "Desktop", 
    "Theme": "Light Mode", 
    "Touch Optimization": "Disabled"
}
response = generate_adaptive_response("Tell me about adaptive UI", context_data)
print("Query: Tell me about adaptive UI")
print(f"Response: {response}")
print("\n")

# Test 2: Greeting
response = generate_adaptive_response("Hello, how are you today?", context_data)
print("Query: Hello, how are you today?")
print(f"Response: {response}")
print("\n")

# Test 3: Technical query
response = generate_adaptive_response("How does sentiment analysis work in this framework?", context_data)
print("Query: How does sentiment analysis work in this framework?")
print(f"Response: {response}")
print("\n")

print("Testing completed!") 