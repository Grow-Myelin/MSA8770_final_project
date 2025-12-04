# 🧠⚖️ Multi-Agent Legal Case Analysis

This project implements a **multi-agent system** for analyzing U.S. appellate opinions using:

- **RAG (Retrieval-Augmented Generation)** over a Qdrant vector DB  
- **spaCy NER + rule-based metadata extraction**  
- **LLM-labeled clustering** (KMeans + GPT)  
- A small **LangGraph ReAct + memory demo**

The system shows how different agents can give complementary views of the *same* case:
- doctrinal rules / standards of review (RAG),
- corpus-level topic/cluster,
- structured metadata (area of law, remedies, orgs, places).

---

## 📚 Data

The original dataset (~1,000 opinions) is **not included** because of size and licensing.

- Source: [CourtListener](https://www.courtlistener.com/) REST API.
- Expected input:  
  `data/summarization_extract_clean.csv` (case texts + ids).

The preprocessing scripts (not required to run the demo, but included) generate:

- `data/summarization_with_metadata.csv`
- `data/summarization_with_metadata_clusters.csv`
- `data/cluster_stats.csv`
- `data/cluster_labels.csv`

---

## ⚙️ Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install \
  qdrant-client \
  langchain-openai \
  langchain-ollama \
  langgraph \
  spacy \
  scikit-learn \
  python-dotenv \
  pandas \
  numpy

python -m spacy download en_core_web_sm

# Create .env file in project root with:
OPENAI_API_KEY=sk-your-openai-key-here
QDRANT_API_KEY=your-qdrant-api-key-here

```

# Core Components

`agents/legal_tools.py`

Defines three tools used by all agents: legal_rag(arguments: {question, mode, k})

Retrieves relevant passages from Qdrant and answers legal questions Modes: "overview", "outcome", "rules", "reasoning"

extract_metadata(arguments: {text? | question?, k?}) Either takes raw text, or uses RAG with a question

Returns: organizations, places, area_of_law, remedy_type

explore_topic(arguments: {cluster_id}) Uses cluster_labels.csv + cluster_stats.csv

Returns: cluster_id, label, description

`agents/cluster_agent.py`  Given a case_id, looks it up in the clustered CSV

Calls explore_topic and returns a short, human-readable explanation of the case’s cluster/topic

`agents/metadata_agent.py`  Given a case_id, retrieves its text

Calls extract_metadata and returns area of law, remedies, orgs, places

`agents/legal_agent.py` (main demo)

## Runs the three-agent pipeline for a chosen case:

Q1 – Doctrinal / RAG
Uses legal_rag to answer:

“What standard of review applies in unemployment-benefit administrative appeals?”

Q2 – Cluster / Topic
Uses explain_case_cluster(CASE_ID) to show cluster id, label, and description.

Q3 – Metadata / Classifier
Uses explain_case_metadata(CASE_ID) to show area_of_law, remedies, orgs, places.

At the end of the script there is also a 2-turn LangGraph memory demo using MemorySaver:

Turn 1: standard-of-review question

Turn 2: follow-up about remedies / area of law, relying on the same conversation context.

```
Activate the environment and run:

source venv/bin/activate
python agents/legal_agent.py

```

**You will see:**

Q1: STANDARD OF REVIEW – doctrinal/RAG rules

Q2: CLUSTER & TOPIC – semantic cluster explanation

Q3: METADATA – area of law, remedies, orgs, places

MEMORY DEMO – two-turn session showing short-term memory across turns.

