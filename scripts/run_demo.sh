#!/bin/bash
# run_demo.sh - Launch the Legal Demo Dashboard
# Usage: ./scripts/run_demo.sh [--port PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default port
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./scripts/run_demo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to run server on (default: 8000)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Check for required data files
if [ ! -f "data/summarization_with_metadata_clusters.csv" ]; then
    echo "WARNING: Data files not found."
    echo "Run ./scripts/generate_data.sh first to generate the corpus."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "========================================"
echo "  MSA8770 Legal Demo Dashboard"
echo "========================================"
echo ""
echo "Starting server on http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
