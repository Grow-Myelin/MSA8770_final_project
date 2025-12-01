import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Import your helpers
from agents.rag_phase1 import answer
from agents.NER_classifier import run_spacy, classify_area_of_law, classify_remedy_type
from agents.llm_cluster import ask_llm_for_label   # renamed to match your file

load_dotenv()

# ----------------------------
# TOOL 1 — RAG QUERY TOOL
# ----------------------------
@tool
def legal_rag(question: str, mode: str = "overview", k: int = 3):
    """
    Retrieve legal answers using RAG + 4 analysis modes.
    """
    print("🔧 Tool called: legal_rag")
    return answer(question, mode=mode, k=k)

# ----------------------------
# TOOL 2 — METADATA EXTRACTION
# ----------------------------
@tool
def extract_metadata(text: str):
    """
    Extract organizations, places, area of law, and remedy type.
    """
    print("🔧 Tool called: extract_metadata")

    orgs, places = run_spacy(text)
    area = classify_area_of_law(text, orgs)
    remedies = classify_remedy_type(text)

    return {
        "organizations": orgs,
        "places": places,
        "area_of_law": area,
        "remedy_type": remedies,
    }

# ----------------------------
# TOOL 3 — TOPIC / CLUSTER EXPLORATION
# ----------------------------
@tool
def explore_topic(cluster_id: int):
    """
    Returns cluster label + LLM description for a cluster_id.
    """
    print("🔧 Tool called: explore_topic")
    return ask_llm_for_label(cluster_id)

# ----------------------------
# LLM + SYSTEM PROMPT
# ----------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """
You are a legal analysis assistant with access to 3 powerful tools:

1) legal_rag(question, mode, k) → retrieve legal answers from case law
2) extract_metadata(text) → extract orgs, places, area of law, remedies
3) explore_topic(cluster_id) → cluster labels and descriptions

Follow ReAct reasoning:
- Think step-by-step
- Use tools when needed
- NEVER invent legal facts not present in the retrieved context.
"""

# ----------------------------
# BUILD THE REACT AGENT — CLEAN
# ----------------------------
agent = create_react_agent(
    model,
    [
        legal_rag,
        extract_metadata,
        explore_topic
    ]
)

# ----------------------------
# EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("🚀 LangGraph Legal Agent Ready\n")

    user_question = "What legal rule applied in the prescriptive easement case?"

    result = agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    })

    print("\n========== FINAL ANSWER ==========")
    print(result["messages"][-1].content)
