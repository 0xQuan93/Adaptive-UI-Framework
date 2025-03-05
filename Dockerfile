FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory and initialize user_feedback.json
RUN mkdir -p data
RUN echo "[]" > data/user_feedback.json

# Expose port 5000 for potential Flask API
EXPOSE 5000

# Run the main application
CMD ["python", "main.py"] 