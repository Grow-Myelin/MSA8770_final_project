#!/bin/bash
# generate_data.sh - Run the full data generation pipeline
# Usage: ./scripts/generate_data.sh [--count N] [--skip-sample]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
SAMPLE_COUNT=100
SKIP_SAMPLE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count)
            SAMPLE_COUNT="$2"
            shift 2
            ;;
        --skip-sample)
            SKIP_SAMPLE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/generate_data.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --count N       Number of documents to sample (default: 100)"
            echo "  --skip-sample   Skip sampling if data already exists"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  MSA8770 Legal Demo - Data Pipeline"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Sample count: $SAMPLE_COUNT"
echo "  Skip sample: $SKIP_SAMPLE"
echo ""

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Create data directory
mkdir -p data
mkdir -p corpus

# ============================================
# STEP 1: Sample court opinions
# ============================================
echo ""
echo "[1/6] Sampling court opinions from CourtListener..."

if [ "$SKIP_SAMPLE" = true ] && [ -f "corpus/comprehensive_cases.json" ]; then
    echo "      Skipping - corpus/comprehensive_cases.json already exists"
else
    # Check for API token
    if [ -z "$COURTLISTENER_TOKEN" ] && [ -z "$COURTLISTENER_API_KEY" ]; then
        # Try to load from .env
        if [ -f ".env" ]; then
            export $(grep -v '^#' .env | grep COURTLISTENER | xargs)
        fi
    fi

    # Use either token name
    TOKEN="${COURTLISTENER_TOKEN:-$COURTLISTENER_API_KEY}"

    if [ -z "$TOKEN" ]; then
        echo "ERROR: No CourtListener API token found."
        echo "Set COURTLISTENER_TOKEN or COURTLISTENER_API_KEY in .env"
        exit 1
    fi

    echo "      Sampling $SAMPLE_COUNT documents (this may take a while)..."
    python courtlistener_sampler.py \
        --token "$TOKEN" \
        --count "$SAMPLE_COUNT" \
        --strategy diverse \
        --output corpus

    echo "      Sampling complete."
fi

# ============================================
# STEP 2: Convert JSON to CSV
# ============================================
echo ""
echo "[2/6] Converting JSON to CSV format..."

if [ -f "corpus/comprehensive_cases.json" ]; then
    python scripts/convert_to_csv.py \
        corpus/comprehensive_cases.json \
        data/summarization_extract_clean.csv
    echo "      Conversion complete."
elif [ -f "corpus/summarization_extract.json" ]; then
    python scripts/convert_to_csv.py \
        corpus/summarization_extract.json \
        data/summarization_extract_clean.csv
    echo "      Conversion complete."
else
    echo "ERROR: No JSON file found to convert."
    echo "Expected: corpus/comprehensive_cases.json or corpus/summarization_extract.json"
    exit 1
fi

# ============================================
# STEP 3: Run NER classification
# ============================================
echo ""
echo "[3/6] Running NER classification..."

python agents/NER_classifier.py
echo "      NER classification complete."

# ============================================
# STEP 4: Run clustering
# ============================================
echo ""
echo "[4/6] Running semantic clustering..."

python agents/clustering_tool.py
echo "      Clustering complete."

# ============================================
# STEP 5: Generate cluster labels with LLM
# ============================================
echo ""
echo "[5/6] Generating cluster labels with LLM..."

python agents/llm_cluster.py
echo "      Cluster labeling complete."

# ============================================
# STEP 6: Populate vector database
# ============================================
echo ""
echo "[6/6] Populating Qdrant vector database..."

python agents/vector_db.py
echo "      Vector database populated."

# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
echo "  Data Pipeline Complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - data/summarization_extract_clean.csv"
echo "  - data/summarization_with_metadata.csv"
echo "  - data/summarization_with_metadata_clusters.csv"
echo "  - data/cluster_stats.csv"
echo "  - data/cluster_labels.csv"
echo ""
echo "Vector database populated in Qdrant cloud."
echo ""
echo "Next step: Run the demo with ./scripts/run_demo.sh"
echo ""
