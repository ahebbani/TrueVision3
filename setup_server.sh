#!/bin/bash
set -e

echo "TrueVision Server Setup"

sudo apt update
sudo apt install -y python3-pip python3-venv curl

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements-server.txt

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling llama3.1:8b..."
ollama pull llama3.1:8b

echo "Server setup complete."
