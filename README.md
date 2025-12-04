# MSA8770 Multi-Agent Legal Analysis System

A multi-agent system for analyzing U.S. appellate court opinions, combining RAG, NLP entity extraction, and LLM-based clustering.

## Overview

This system provides three complementary analytical views of legal cases:

| Agent | Purpose | Technology |
|-------|---------|------------|
| **RAG Agent** | Doctrinal questions, standards of review | Qdrant vector DB + OpenAI |
| **Metadata Agent** | Entity extraction (orgs, places, remedies) | spaCy NER + rules |
| **Cluster Agent** | Topic/theme classification | KMeans + GPT labeling |

The system includes a **FastAPI web dashboard** for interactive querying and a **LangGraph memory demo** for multi-turn conversations.

## Architecture

```
┌─────────────────────────────────────────┐
│     Web Dashboard (localhost:8000)      │
│     FastAPI + Jinja2 HTML interface     │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ↓          ↓          ↓
┌────────┐ ┌──────────┐ ┌─────────────┐
│  RAG   │ │ Metadata │ │  Cluster    │
│ Agent  │ │  Agent   │ │   Agent     │
└────┬───┘ └────┬─────┘ └──────┬──────┘
     │          │              │
     ↓          ↓              ↓
 Qdrant DB   spaCy NER    KMeans + GPT
     │          │              │
     └──────────┴──────────────┘
                │
    ┌───────────┴───────────┐
    │  1,105 Legal Opinions │
    │  (CourtListener.com)  │
    └───────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd MSA8770_final_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Create `.env` in the project root:

```
OPENAI_API_KEY=your_openai_key_here
QDRANT_API_KEY=your_qdrant_key_here
```

### 3. Run the Web Dashboard

```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### 4. Run CLI Demo (Optional)

```bash
python agents/legal_agent.py
```

## Project Structure

```
MSA8770_final_project/
├── app/                    # FastAPI web application
│   ├── main.py            # App initialization
│   ├── api.py             # REST API endpoints
│   ├── templates/         # HTML dashboard
│   └── static/            # CSS styles
├── agents/                 # Multi-agent system
│   ├── legal_tools.py     # Shared tools (RAG, metadata, clustering)
│   ├── legal_agent.py     # CLI demo
│   ├── cluster_agent.py   # Topic classification
│   ├── metadata_agent.py  # Entity extraction
│   ├── orchestrator.py    # Agent orchestration
│   └── vector_db.py       # Qdrant interface
├── data/                   # Processed datasets
├── corpus/                 # Raw legal opinions
├── eval/                   # Evaluation methodology
├── scripts/                # Utility scripts
└── docs/                   # Additional documentation
```

## Data Pipeline

The system uses ~1,105 legal opinions from [CourtListener.com](https://www.courtlistener.com/):

```
Raw opinions → NER extraction → Clustering → Labeled dataset
```

Data files (in `data/`):
- `summarization_extract_clean.csv` - Base corpus
- `summarization_with_metadata.csv` - With NER metadata
- `summarization_with_metadata_clusters_labeled.csv` - Final labeled dataset

## Documentation

- [CourtListener Sampler](docs/COURTLISTENER_SAMPLER.md) - Data collection tool
- [Sampler Quick Start](docs/SAMPLER_QUICKSTART.md) - 5-minute guide
- [Local Testing Guide](docs/TESTING_LOCALLY.md) - Testing instructions
- [Agents Documentation](agents/README.md) - Agent implementation details
- [Evaluation Methodology](eval/EVALUATION_METHODOLOGY.md) - RAG evaluation

## Technologies

- **Framework**: FastAPI, LangChain, LangGraph
- **Vector DB**: Qdrant Cloud
- **LLM**: OpenAI GPT-4o-mini
- **NLP**: spaCy (en_core_web_sm)
- **ML**: scikit-learn (KMeans clustering)

## License

Educational and research use. Legal data sourced from CourtListener (Free Law Project).
