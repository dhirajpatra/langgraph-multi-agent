#!/bin/bash

# Start Ollama server in the background
exec ollama serve & 
OLLAMA_PID=$!

# Ensure model directory exists
mkdir -p /root/.ollama/models

# Pull model if not already present
# MODEL_NAME="deepseek-r1:1.5b"
MODEL_NAME="llama3.1:8b"
# MODEL_NAME="qwen2.5:7b-instruct"
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Model not found. Pulling $MODEL_NAME..."
  until ollama pull "$MODEL_NAME"; do
    echo "Retrying model download..."
    sleep 5
  done
else
  echo "Model already exists. Skipping pull."
fi

# Wait for the Ollama server to be available
echo "Waiting for the Ollama server to start..."
until curl -s http://localhost:11434 > /dev/null; do
  echo "Ollama server is still starting..."
  sleep 2
done
echo "Ollama server is running."

# Keep container alive by waiting for Ollama process
wait $OLLAMA_PID
