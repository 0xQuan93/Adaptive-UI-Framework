# AI-Language Model Fusion Layer (AI-LMFL)
import json
import os
import random
import sys
import traceback
import time

# Debug mode flag
DEBUG_MODE = os.environ.get('AI_DEBUG', 'false').lower() == 'true'

# Global AI pipeline
ai_pipeline = None

# Try to import transformers and necessary libraries, with fallback to mock implementation
try:
    try:
        from transformers import pipeline as transformers_pipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Check if CUDA is available
        CUDA_AVAILABLE = torch.cuda.is_available()
        if CUDA_AVAILABLE:
            print("[INFO] CUDA is available, will use GPU acceleration")
        else:
            print("[WARNING] CUDA is not available, will use CPU only (slower)")
        
        # Try to import quantization libraries for CPUs
        try:
            import bitsandbytes
            import accelerate
            print("[INFO] Quantization libraries available (bitsandbytes, accelerate)")
        except ImportError:
            print("[WARNING] Quantization libraries not available (install bitsandbytes, accelerate for better CPU performance)")
        
        HAS_REQUIRED_PACKAGES = True
    except ImportError as e:
        print(f"[WARNING] Unable to import required packages: {e}")
        HAS_REQUIRED_PACKAGES = False
        
except Exception as e:
    print(f"[ERROR] Unexpected error when importing dependencies: {e}")
    HAS_REQUIRED_PACKAGES = False

# Enhanced mock implementation for the transformers pipeline
class EnhancedMockPipeline:
    """A more sophisticated mock of the transformers pipeline for development and testing."""
    
    def __init__(self, task, model):
        self.task = task
        self.model = model
        self.responses = {
            "greeting": [
                "Hello! How can I assist you with the adaptive UI today?",
                "Hi there! I'm your AI assistant with the adaptive UI framework.",
                "Greetings! I'm here to help with your questions."
            ],
            "about_ui": [
                "The adaptive UI framework dynamically adjusts to user preferences and context.",
                "Our UI system analyzes your sentiment and adapts the interface accordingly.",
                "The framework uses AI to provide a personalized user experience based on your needs."
            ],
            "technical": [
                "The framework uses a sentiment analysis module to detect emotions in user input.",
                "Our system incorporates context awareness through platform detection and user modeling.",
                "The adaptive components are built on a feedback loop that continuously improves the UI."
            ],
            "default": [
                "I understand your question. Let me provide some information about that.",
                "That's an interesting topic. Here's what I can tell you.",
                "I'd be happy to help with that. Here's what I know."
            ]
        }
        print(f"[INFO] Enhanced Mock Pipeline initialized - task: {task}, model: {model}")
    
    def __call__(self, prompt, max_length=50, do_sample=True, **kwargs):
        """Generate a response based on the input prompt."""
        print(f"[INFO] Generating response for: {prompt[:50]}...")
        
        # Determine response category based on prompt keywords
        category = "default"
        if any(word in prompt.lower() for word in ["hello", "hi", "hey", "greetings"]):
            category = "greeting"
        elif any(word in prompt.lower() for word in ["ui", "interface", "adaptive", "adapt"]):
            category = "about_ui"
        elif any(word in prompt.lower() for word in ["how", "technical", "system", "framework"]):
            category = "technical"
        
        # Select a random response from the appropriate category
        response = random.choice(self.responses[category])
        
        # If the prompt is a specific question, add some context from the prompt
        if "?" in prompt:
            context = f" Regarding '{prompt.split('?')[0]}?', "
            response = context + response
        
        # Personalize the response based on the context information in the prompt
        if "Theme: Dark Mode" in prompt:
            response += " I notice you're using our dark mode interface."
        elif "Theme: Light Mode" in prompt:
            response += " I see you're using our light mode interface."
        
        if "Device: Enabled" in prompt or "Mobile" in prompt:
            response += " Our touch-optimized interface should work well on your mobile device."
        
        # Return in the format expected by the transformers pipeline
        return [{'generated_text': response}]

# Configuration
CONFIG_FILE = 'config.json'

def load_config():
    '''
    Loads configuration settings from config.json
    :return: Dictionary of configuration settings
    '''
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
        return config
    except (json.JSONDecodeError, FileNotFoundError):
        # Return default config if file is missing or corrupted
        return {
            "ai_settings": {
                "ai_response_max_length": 50,
                "ai_model": "gpt2"
            }
        }

# Load configuration
config = load_config()

# Initialize AI model
try:
    model_name = config.get("ai_settings", {}).get("ai_model", "gpt2")
    
    if HAS_REQUIRED_PACKAGES:
        print(f"[INFO] Initializing real transformers pipeline with model: {model_name}")
        try:
            # Check if using a small language model
            if "llama" in model_name.lower() or "phi" in model_name.lower() or "mistral" in model_name.lower():
                print(f"[INFO] Using optimized loading for small model: {model_name}")
                try:
                    print("[INFO] Loading small language model for CPU usage")
                    # For CPU usage, we'll use a smaller model with full precision
                    # but optimize with torch.float16 if possible
                    if CUDA_AVAILABLE:
                        # With GPU, use half precision
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            device_map="auto"
                        )
                    else:
                        # Without GPU, still try to optimize memory usage
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            low_cpu_mem_usage=True
                        )
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Fix for models where pad_token is not defined or same as eos_token
                    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
                        print("[INFO] Setting custom pad token for the tokenizer")
                        # Use a different token as pad token (common approach)
                        if "phi" in model_name.lower():
                            # Phi models often use this special token
                            tokenizer.pad_token = tokenizer.eos_token
                            # Configure model to work with this arrangement
                            model.config.pad_token_id = model.config.eos_token_id
                            print("[INFO] Using EOS token as PAD token for Phi model with special handling")
                        else:
                            # For other models, use a general approach
                            tokenizer.pad_token = "[PAD]"
                            # Add the new token if it doesn't exist
                            if tokenizer.pad_token_id is None:
                                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                                # Resize token embeddings if we added a new token
                                model.resize_token_embeddings(len(tokenizer))
                    
                    class CustomSmallModelPipeline:
                        def __init__(self, model, tokenizer, model_name):
                            self.model = model
                            self.tokenizer = tokenizer
                            self.model_name = model_name
                            
                        def __call__(self, prompt, max_length=50, do_sample=True, temperature=0.7, top_p=0.9, **kwargs):
                            # Special handling for different model families
                            if "llama" in self.model_name.lower():
                                # Special handling for Llama models
                                if not prompt.startswith("<s>"):
                                    # Ensure proper formatting for Llama
                                    if "[INST]" not in prompt:
                                        prompt = f"<s>[INST] {prompt} [/INST]"
                            
                            # Properly handle padding for generation
                            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                            
                            # Move to GPU if available
                            if CUDA_AVAILABLE:
                                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                
                            # Generate output with proper padding configuration
                            with torch.no_grad():
                                try:
                                    outputs = self.model.generate(
                                        inputs.input_ids,
                                        attention_mask=inputs.attention_mask,
                                        max_new_tokens=max_length,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_p=top_p,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **{k: v for k, v in kwargs.items() if k not in ['return_full_text']}
                                    )
                                except Exception as e:
                                    print(f"[WARNING] Error in generation with attention mask: {e}")
                                    # Fall back to generation without attention mask if needed
                                    outputs = self.model.generate(
                                        inputs.input_ids,
                                        max_new_tokens=max_length,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_p=top_p,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **{k: v for k, v in kwargs.items() if k not in ['return_full_text']}
                                    )
                            
                            # Decode the output
                            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Strip the prompt if return_full_text is False
                            if not kwargs.get('return_full_text', True):
                                # Try to remove the prompt part - but this is tricky and model-specific
                                if prompt in generated_text:
                                    generated_text = generated_text[len(prompt):].strip()
                                # Simpler approach: if we have an "Answer:" in the response, take what comes after
                                elif "Answer:" in generated_text:
                                    generated_text = generated_text.split("Answer:", 1)[1].strip()
                                # For Llama models, try to extract just the assistant's response
                                elif "[/INST]" in prompt and "[/INST]" in generated_text:
                                    generated_text = generated_text.split("[/INST]", 1)[1].strip()
                            
                            return [{'generated_text': generated_text}]
                    
                    ai_model = CustomSmallModelPipeline(model, tokenizer, model_name)
                    print(f"[INFO] Successfully initialized optimized small model: {model_name}")
                    is_mock = False
                    is_small_model = True
                except Exception as e:
                    print(f"[WARNING] Failed to load small language model: {e}")
                    print("[INFO] Falling back to standard pipeline")
                    ai_model = transformers_pipeline("text-generation", model=model_name)
                    is_mock = False
                    is_small_model = False
            else:
                # Standard pipeline for other models
                ai_model = transformers_pipeline("text-generation", model=model_name)
                print(f"[INFO] Successfully initialized real transformers model: {model_name}")
                is_mock = False
                is_small_model = False
        except Exception as e:
            print(f"[WARNING] Failed to load real transformers model: {e}")
            print("[INFO] Falling back to enhanced mock implementation")
            ai_model = EnhancedMockPipeline("text-generation", model_name)
            is_mock = True
            is_small_model = False
    else:
        print("[INFO] Initializing enhanced mock language model...")
        ai_model = EnhancedMockPipeline("text-generation", model_name)
        is_mock = True
        is_small_model = False
        print(f"[INFO] Successfully initialized mock model: {model_name}")
except Exception as e:
    print(f"[ERROR] Error initializing AI model: {e}")
    print("[ERROR] Using fallback model behavior")
    ai_model = None
    is_mock = True
    is_small_model = False

def generate_adaptive_response(user_input, context_data):
    """
    Generate an AI response that adapts to the user input and context.
    
    Args:
        user_input (str): The user's input text
        context_data (dict): Context information about the user's environment
        
    Returns:
        str: The generated response
    """
    start_time = time.time()
    
    if DEBUG_MODE:
        print(f"[DEBUG] generate_adaptive_response called with input: {user_input}")
        print(f"[DEBUG] Context data: {json.dumps(context_data)}")
    
    try:
        # Load configuration for AI settings
        config = load_config()
        
        # Check if we're in mock mode (either by configuration or lack of dependencies)
        if not HAS_REQUIRED_PACKAGES or config.get("use_mock_ai", False):
            if DEBUG_MODE:
                print("[DEBUG] Using mock AI implementation")
            # Use the mock implementation
            mock_pipeline = EnhancedMockPipeline(
                task="text-generation",
                model="mock-model"
            )
            
            # Generate a mock response based on user input and context
            prompt = format_prompt_with_context(user_input, context_data)
            if DEBUG_MODE:
                print(f"[DEBUG] Generated prompt: {prompt}")
                
            response = mock_pipeline(
                prompt,
                max_length=100,
                do_sample=True
            )
            
            if DEBUG_MODE:
                print(f"[DEBUG] Mock response generated in {time.time() - start_time:.2f} seconds")
            
            return clean_response(response[0]["generated_text"])
        
        # Use the actual AI model
        global ai_pipeline
        if ai_pipeline is None:
            if DEBUG_MODE:
                print("[DEBUG] Initializing AI pipeline")
            ai_pipeline = initialize_ai_pipeline(config)
        
        # Get the maximum response length from config
        max_length = config.get("ai_settings", {}).get("ai_response_max_length", 500)
        
        # Format the prompt with context information
        prompt = format_prompt_with_context(user_input, context_data)
        if DEBUG_MODE:
            print(f"[DEBUG] Generated prompt: {prompt}")
            print(f"[DEBUG] Max response length: {max_length}")
        
        # Generate response using the AI pipeline
        response = ai_pipeline(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract and clean the generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0]["generated_text"]
        else:
            generated_text = str(response)
            
        cleaned_response = clean_response(generated_text)
        
        if DEBUG_MODE:
            print(f"[DEBUG] AI response generated in {time.time() - start_time:.2f} seconds")
            print(f"[DEBUG] Raw response: {generated_text[:100]}...")
            print(f"[DEBUG] Cleaned response: {cleaned_response[:100]}...")
            
        return cleaned_response
        
    except Exception as e:
        print(f"[ERROR] Error generating response: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Return a fallback response in case of error
        return f"I apologize, but I encountered an error processing your request. Please try again. (Error: {str(e)[:100]})"

def get_model_info():
    '''
    Returns information about the currently loaded AI model.
    :return: Dictionary with model information.
    '''
    if ai_model is None:
        return {"status": "error", "model": None}
    
    if is_mock:
        model_type = "enhanced mock"
    elif is_small_model:
        model_type = "optimized small LLM"
    else:
        model_type = "real transformers"
    
    return {
        "status": "loaded",
        "model": config.get("ai_settings", {}).get("ai_model", "gpt2") + f" ({model_type})",
        "max_length": config.get("ai_settings", {}).get("ai_response_max_length", 50),
        "capabilities": ["sentiment-aware responses", "context adaptation", "personalized interactions"],
        "is_mock": is_mock,
        "is_small_model": is_small_model if not is_mock else False,
        "using_gpu": CUDA_AVAILABLE and not is_mock
    }

def format_prompt_with_context(user_input, context_data):
    """
    Format the prompt with context information for better AI responses.
    
    Args:
        user_input (str): The user's input text
        context_data (dict): Context information about the user's environment
        
    Returns:
        str: Formatted prompt with context
    """
    # Create a summary of the context
    context_summary = (f"User Context: {context_data.get('Layout', 'Standard Layout')}, "
                      f"Theme: {context_data.get('Theme', 'Standard Theme')}, "
                      f"Touch Optimization: {context_data.get('Touch Optimization', 'Unknown Device')}")
    
    # Get the model name from config
    config = load_config()
    model_name = config.get("ai_settings", {}).get("ai_model", "gpt2").lower()
    
    # Adjust prompt format based on model family
    if "llama" in model_name:
        # Llama-specific prompting
        prompt = f"""<s>[INST] <<SYS>>
You are an AI assistant for the Adaptive UI Framework. Your responses should be helpful, accurate and tailored to the user's context.
<</SYS>>

User context: {context_summary}

{user_input} [/INST]
"""
    elif "phi" in model_name:
        # Phi-specific prompting
        prompt = f"""Instruction: You are an AI assistant for the Adaptive UI Framework. Provide a helpful, accurate, and concise response to the user's query. Ensure your response is relevant to the user's context.

Context Information: {context_summary}

User Query: {user_input}

Response:"""
    else:
        # Default format for other models
        prompt = f"""Context: {context_summary}
Question: {user_input}
Answer:"""
    
    return prompt

def clean_response(generated_text):
    """
    Clean the generated text to get a proper response.
    
    Args:
        generated_text (str): The raw generated text from the model
        
    Returns:
        str: Cleaned response
    """
    if not generated_text:
        return "I apologize, but I couldn't generate a response."
    
    response = generated_text.strip()
    
    # Check for various pattern markers and clean them
    if "Response:" in response:
        response = response.split("Response:", 1)[1].strip()
    elif "Answer:" in response:
        response = response.split("Answer:", 1)[1].strip()
    
    # Clean up special tokens that might appear
    special_tokens = [
        "<|AI|>", "<|assistant|>", "<|user|>", "<|endofgeneration|>", 
        "<|endoftext|>", "<|beginofstoryusingtemplates|>", "<s>", "</s>"
    ]
    for token in special_tokens:
        if token in response:
            response = response.replace(token, "").strip()
    
    # Remove any chat pattern if it exists
    chat_patterns = [
        "<|User|>", "<|Assistant|>", "|User|", "|Assistant|",
        "User:", "Assistant:"
    ]
    for pattern in chat_patterns:
        if pattern in response:
            # Try to keep just the assistant's response
            parts = response.split(pattern)
            if len(parts) > 1:
                # Try to use the most relevant part (usually after Assistant)
                for i, part in enumerate(parts):
                    if i > 0 and ("Assistant" in parts[i-1] or "AI" in parts[i-1]):
                        response = part.strip()
                        break
                else:
                    # If no clear assistant part, join all parts
                    response = " ".join([p.strip() for p in parts if p.strip()])
    
    return response
