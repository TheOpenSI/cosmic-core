# NVIDIA GPU Ollama setup (works on Linux only)
FROM ollama/ollama:latest AS base

# Port
EXPOSE 11434/tcp