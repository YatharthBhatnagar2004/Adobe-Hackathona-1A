#!/bin/bash

# Docker commands for running solve_1b.py

echo "🐳 Building Docker image..."
docker build -t adobe-hackathon-1b .

echo "🚀 Running the application..."
docker run --rm -it \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/models:/app/models" \
    adobe-hackathon-1b

echo "✅ Done! Check the output/ directory for results." 