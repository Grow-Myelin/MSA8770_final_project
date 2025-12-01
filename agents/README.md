# This project implements a multi-agent system for analyzing legal case texts using:

**RAG (Retrieval-Augmented Generation)**

**NER + Metadata Extraction (spaCy)**

**LLM-based Clustering**

**LangGraph ReAct Multi-Tool Agent**

⚙️ Installation & Setup
1. Create & Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate

2. Install Dependencies
pip install qdrant-client langchain-openai langgraph spacy scikit-learn python-dotenv pandas
python -m spacy download en_core_web_sm

3. Environment Variables
Create a .env file in the project root:
OPENAI_API_KEY=your_key_here
QDRANT_API_KEY=your_key_here

📁 Project Structure
agents/

  `rag_phase1.py`           # RAG: Retrieval + 4 legal modes
  
  `NER_classifier.py `      # spaCy NER + metadata classification
  
  `llm_cluster.py  `        # Cluster label lookup
  
  `legal_tools.py`          # Wraps RAG + NER + Clustering as tools
  
  `legal_graph_agent.py`    # Main LangGraph ReAct agent
  
  `vector_db.py `           # create vectore data base of text embeddings and upsert in qdrant 

## Running the Agent

Activate environment (if not already):

source venv/bin/activate

Run the main agent: ( make sure legal_tools.py is in the directory)

python agents/legal_langgraph_agent.py


You should see:

🚀 LangGraph Legal Agent Ready
🔧 Tool called: legal_rag
========== FINAL ANSWER ==========
...

🧪 Example Queries

You can ask the agent things like:

What legal rule applies in the prescriptive easement case?


The agent will automatically decide whether to use:

RAG retrieval

Metadata extraction

Topic/cluster exploration

🛠️ What Each Script Does
rag_phase1.py

RAG pipeline that:

Retrieves relevant cases from Qdrant , Builds grounded context , Answers in 4 legal modes:

overview

outcome

rules

reasoning

NER_classifier.py Extracts: ORGs

Places (GPE/LOC)

Area of law

Remedy type

llm_cluster.py Returns: Cluster label , Cluster description

legal_tools.py Wraps all components as tools so LangGraph can call them


legal_langgraph_agent.py: The final LangGraph ReAct agent that orchestrates:

The RAG tool

The NER/metadata tool

The topic/cluster tool

It handles reasoning, tool-calling, and final answer generation.
