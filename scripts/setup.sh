#!/bin/bash
# setup.sh - One-command setup for MSA8770 Legal Demo
# Usage: ./scripts/setup.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  MSA8770 Legal Demo - Setup Script"
echo "========================================"

cd "$PROJECT_DIR"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

echo ""
echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "      Virtual environment created."
else
    echo "      Virtual environment already exists."
fi

echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Dependencies installed."

echo ""
echo "[4/4] Downloading spaCy language model..."
python -m spacy download en_core_web_sm -q
echo "      spaCy model downloaded."

# Create data directory
mkdir -p data

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Ensure .env file has your API keys:"
echo "     - OPENAI_API_KEY"
echo "     - QDRANT_API_KEY"
echo "     - COURTLISTENER_API_KEY"
echo ""
echo "  2. Generate data (if not already done):"
echo "     ./scripts/generate_data.sh"
echo ""
echo "  3. Run the demo:"
echo "     ./scripts/run_demo.sh"
echo ""
