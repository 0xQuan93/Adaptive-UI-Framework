# AI-Language Model Fusion Layer (AI-LMFL)
import json
import os
import random

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
            import bitsandbytes as bnb
            from accelerate import init_empty_weights, infer_auto_device_map
            QUANTIZATION_AVAILABLE = True
            print("[INFO] Quantization libraries available (bitsandbytes, accelerate)")
        except ImportError:
            QUANTIZATION_AVAILABLE = False
            print("[WARNING] Quantization libraries not available, will use full precision")
        
        REAL_TRANSFORMERS_AVAILABLE = True
        print("[INFO] Successfully imported transformers library")
    except (ImportError, OSError) as e:
        # Handle both import errors and DLL loading errors
        REAL_TRANSFORMERS_AVAILABLE = False
        print(f"[WARNING] Transformers library not available: {e}")
        print("[WARNING] Using enhanced mock implementation")
except Exception as e:
    REAL_TRANSFORMERS_AVAILABLE = False
    print(f"[WARNING] Unexpected error loading transformers: {e}")
    print("[WARNING] Using enhanced mock implementation")

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
    
    if REAL_TRANSFORMERS_AVAILABLE:
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
                            
                            inputs = self.tokenizer(prompt, return_tensors="pt")
                            
                            # Move to GPU if available
                            if CUDA_AVAILABLE:
                                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                
                            # Generate output
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    inputs.input_ids,
                                    max_new_tokens=max_length,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                    top_p=top_p,
                                    pad_token_id=self.tokenizer.eos_token_id,
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
    '''
    Uses an AI-Language Model to generate adaptive responses.
    :param user_input: User query.
    :param context_data: UI adaptation recommendations.
    :return: AI-generated response.
    '''
    if ai_model is None:
        # Fallback if model fails to load
        return f"I understand your question about '{user_input}'. Currently using adaptive UI with {context_data['Theme']} theme."
        
    try:
        # Get max response length from config
        max_length = config.get("ai_settings", {}).get("ai_response_max_length", 50)
        
        # Prepare context for better response
        context_summary = (f"User Context: {context_data.get('Layout', 'Standard Layout')}, "
                          f"Theme: {context_data.get('Theme', 'Standard Theme')}, "
                          f"Device: {context_data.get('Touch Optimization', 'Unknown Device')}")
        
        # Different approach for real transformers vs mock
        if not is_mock:
            # For real transformers, create a more specific prompt
            if is_small_model:
                # Special prompting for small language models (Llama, Phi, etc.)
                if "llama" in model_name.lower():
                    # Llama-specific prompting
                    ai_prompt = f"""<s>[INST] <<SYS>>
You are an AI assistant for the Adaptive UI Framework. Your responses should be helpful, accurate and tailored to the user's context.
<</SYS>>

User context: {context_summary}

{user_input} [/INST]
"""
                else:
                    # Default format for other small models
                    ai_prompt = f"""Context: {context_summary}
Question: {user_input}
Answer:"""
            else:
                # For GPT-2 and other standard models
                if "adaptive ui" in user_input.lower() or "ui" in user_input.lower():
                    ai_prompt = f"Question: {user_input}\nAnswer: Adaptive UI is a user interface that dynamically adjusts based on user preferences, context, and behavior."
                elif "hello" in user_input.lower() or "hi" in user_input.lower():
                    ai_prompt = f"Question: {user_input}\nAnswer: Hello! I'm your AI assistant for the adaptive UI framework. How can I help you today?"
                else:
                    ai_prompt = f"Question: {user_input}\nAnswer: "
                
            # Generate with real transformers
            print(f"[INFO] Using real transformers with prompt: {ai_prompt[:50]}...")
            response_obj = ai_model(
                ai_prompt, 
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            # Extract response
            generated_text = response_obj[0]['generated_text']
            
            # Clean the response based on model type
            if is_small_model:
                # For small models, the response may already be clean
                response = generated_text.strip()
            else:
                # Clean the response for other models, removing the question part if it exists
                if "Answer:" in generated_text:
                    response = generated_text.split("Answer:", 1)[1].strip()
                else:
                    response = generated_text.strip()
                
            # Add context-aware personalization if not already in the response
            if "Theme: Dark Mode" in context_summary and "dark mode" not in response.lower():
                response += " I notice you're using our dark mode interface."
            elif "Theme: Light Mode" in context_summary and "light mode" not in response.lower():
                response += " I see you're using our light mode interface."
            
            if ("Device: Enabled" in context_summary or "Mobile" in context_summary) and "mobile" not in response.lower():
                response += " Our touch-optimized interface should work well on your mobile device."
        else:
            # Use the mock implementation
            ai_prompt = f"{context_summary}. Respond to: {user_input}"
            response = ai_model(ai_prompt, max_length=max_length, do_sample=True)[0]['generated_text']
            
        return response
    except Exception as e:
        # Fallback for any errors
        print(f"[ERROR] Error generating AI response: {e}")
        return f"I understand your question about '{user_input}', but I'm having trouble generating a response right now."

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
