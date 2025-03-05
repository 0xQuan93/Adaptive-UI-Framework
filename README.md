# Adaptive User Interface Framework

## Overview

This repository contains the **Adaptive User Interface Framework** that integrates **Sentient AI and Large Language Models (LLMs)** into digital and virtual environments. This framework dynamically adjusts UI components based on **user sentiment, context, and behavioral interactions**.

## Features
- **Sentient AI Integration**: Captures **user emotions, cognitive load, and intent** using NLP and behavioral tracking.
- **Adaptive UI System**: Detects **device type, OS, UI preferences, and screen settings** to adjust UI dynamically.
- **Memetic Feedback Loop (MFL)**: Continuously **learns from user behavior to refine UI interactions**.
- **AI-Language Model Fusion**: Enhances **AI response accuracy** and **personalized UX**.
- **Trust & Ethics Module**: Ensures **AI transparency and user control over UI adaptations**.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/adaptive-ui-framework.git
cd adaptive-ui-framework
```

### Setting up a virtual environment (Recommended)

Create and activate a virtual environment to isolate dependencies:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Hugging Face Transformers Integration

This framework integrates with the Hugging Face `transformers` library for enhanced language model capabilities:

1. **Prerequisites**:
   - For Windows users: Install Microsoft Visual C++ Redistributable (https://aka.ms/vs/16/release/vc_redist.x64.exe)
   - For GPU acceleration: Install CUDA and cuDNN if a compatible GPU is available

2. **Testing the Transformers Integration**:
   ```bash
   python test_response.py
   ```
   This will show if the real transformers model is loaded or if it's using the enhanced mock implementation.

3. **Fallback System**:
   The framework includes an enhanced mock implementation that will be used automatically if:
   - The transformers library cannot be imported
   - Required DLLs are missing
   - Model loading fails for any reason

## Usage

### Command Line Interface

Run the framework in interactive mode:

```bash
python main.py
```

### Web API

Start the Flask API server:

```bash
python app.py
```

Access the API endpoints:
- `GET /api/context` - Get user context information
- `GET /api/preferences` - Get UI preferences
- `POST /api/sentiment` - Analyze sentiment of text
- `POST /api/adaptation` - Generate UI adaptation
- `POST /api/response` - Generate AI response
- `POST /api/feedback` - Submit user feedback
- `POST /api/trust` - Get trust score

### Docker Deployment

Build and run with Docker:

```bash
docker build -t adaptive-ui-framework .
docker run -p 5000:5000 adaptive-ui-framework
```

## Configuration

Customize the framework by editing `config.json`:

```json
{
    "ui_settings": {
        "dark_mode": false,
        "high_contrast_mode": false,
        "default_font_size": "Medium"
    },
    "ai_settings": {
        "ai_model": "gpt2",
        "ai_response_max_length": 50
    }
}
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_sim.py
```

## Repository Structure

```
adaptive-ui-framework/
│── README.md              # Project documentation
│── requirements.txt       # Required dependencies
│── main.py                # Entry point for running the framework
│── app.py                 # Flask API server
│── config.json            # Configuration settings
│── Dockerfile             # Docker configuration
│── modules/
│   ├── sim.py             # Sentient Interaction Model
│   ├── cae.py             # Contextual Adaptation Engine
│   ├── mfl.py             # Memetic Feedback Loop
│   ├── ai_lmfl.py         # AI-Language Model Fusion Layer
│   ├── eotm.py            # Engagement Optimization & Trust Module
│── data/
│   ├── user_feedback.json # User feedback storage
│── tests/
│   ├── test_sim.py        # Tests for Sentient Interaction Model
│   ├── test_cae.py        # Tests for Contextual Adaptation Engine
│   ├── test_mfl.py        # Tests for Memetic Feedback Loop
│   ├── test_ai_lmfl.py    # Tests for AI-LM Fusion Layer
│   ├── test_eotm.py       # Tests for Trust Module
│── diagrams/
│   ├── system_architecture.png   # System Architecture Diagram
```

## Additional Documentation

- [HOWTO.md](HOWTO.md) - Detailed usage instructions
- [TESTS.md](TESTS.md) - Test cases documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide

## License

This project is licensed under the **MIT License**.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contributors

- **0xQuan** - [Profile](https://github.com/0xQuan93)

