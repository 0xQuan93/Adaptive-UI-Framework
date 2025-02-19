# Test Cases - Adaptive UI Framework

## Overview

This document provides unit tests for the core modules of the **Adaptive User Interface Framework**.

### **Test Setup**

Ensure that all dependencies are installed:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/
```

---

## **1. Test Cases for Sentient Interaction Model (SIM)**
File: `tests/test_sim.py`

```python
import pytest
from sim import analyze_sentiment_vader

def test_positive_sentiment():
    result = analyze_sentiment_vader("I love this UI!")
    assert result["Sentiment"] == "Positive"

def test_negative_sentiment():
    result = analyze_sentiment_vader("I hate using this interface.")
    assert result["Sentiment"] == "Negative"

def test_neutral_sentiment():
    result = analyze_sentiment_vader("The UI is okay, nothing special.")
    assert result["Sentiment"] == "Neutral"
```

---

## **2. Test Cases for Contextual Adaptation Engine (CAE)**
File: `tests/test_cae.py`

```python
import pytest
from cae import detect_user_context, detect_ui_preferences, generate_ui_adaptation

def test_detect_user_context():
    context = detect_user_context()
    assert "Device Type" in context
    assert "Operating System" in context

def test_detect_ui_preferences():
    preferences = detect_ui_preferences()
    assert "Dark Mode Enabled" in preferences
    assert "Font Size" in preferences

def test_generate_ui_adaptation():
    context = {"Device Type": "Desktop"}
    preferences = {"Dark Mode Enabled": True, "Font Size": "Large"}
    adaptation = generate_ui_adaptation(context, preferences)
    assert adaptation["Theme"] == "Dark Mode"
    assert adaptation["Font Size"] == "Large"
```

---

## **3. Test Cases for Memetic Feedback Loop (MFL)**
File: `tests/test_mfl.py`

```python
import pytest
import json
from mfl import store_user_feedback

def test_store_user_feedback():
    feedback_list = []
    user_input = "I like this feature."
    sentiment_data = {"Sentiment": "Positive", "Confidence": 0.9}
    adaptation_data = {"Theme": "Dark Mode"}

    store_user_feedback(user_input, sentiment_data, adaptation_data)
    
    with open("data/user_feedback.json", "r") as file:
        feedback_list = json.load(file)

    assert len(feedback_list) > 0
    assert feedback_list[-1]["Sentiment"]["Sentiment"] == "Positive"
```

---

## **4. Test Cases for AI-Language Model Fusion Layer (AI-LMFL)**
File: `tests/test_ai_lmfl.py`

```python
import pytest
from ai_lmfl import generate_adaptive_response

def test_generate_adaptive_response():
    context = {"Layout": "Desktop", "Theme": "Light Mode"}
    response = generate_adaptive_response("What is adaptive UI?", context)
    assert isinstance(response, str)
    assert len(response) > 10
```

---

## **5. Test Cases for Engagement Optimization & Trust Module (EOTM)**
File: `tests/test_eotm.py`

```python
import pytest
from eotm import calculate_user_trust_score

def test_trust_score_calculation():
    feedback_data = [
        {"Sentiment": {"Sentiment": "Positive"}},
        {"Sentiment": {"Sentiment": "Positive"}},
        {"Sentiment": {"Sentiment": "Negative"}}
    ]
    trust_score = calculate_user_trust_score(feedback_data)
    assert trust_score == 66.67  # 2 out of 3 positive = 66.67%
```

---

## **Running the Tests**

Run all tests using:

```bash
pytest tests/
```

Run a specific test file:

```bash
pytest tests/test_sim.py
```

Run a single test case:

```bash
pytest tests/test_sim.py::test_positive_sentiment
```

---

## **Conclusion**

This test suite ensures that all **core modules** function correctly and provides a **baseline for debugging future updates**.

Happy Testing! 🚀
