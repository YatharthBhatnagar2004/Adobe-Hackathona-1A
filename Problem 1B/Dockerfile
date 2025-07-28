# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the application files
COPY solve_1b.py .
COPY download_models.py .

# Create necessary directories
RUN mkdir -p input output models

# Copy input files if they exist
COPY input/ ./input/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "solve_1b.py"] 