# agents/legal_agent.py

from agents.legal_tools import legal_rag
from agents.cluster_agent import explain_case_cluster
from agents.metadata_agent import explain_case_metadata

print("⚖️  Multi-Agent Legal System (RAG + Clustering + Metadata)")

# --------------------------------------
# SHARED CASE CONTEXT
# --------------------------------------

# High-level descriptor for the kind of case we care about (for Q1 RAG query)
CASE_QUERY = (
    "Unemployment-benefit administrative appeal case involving review "
    "of an unemployment law judge's decision."
)

# Specific case_id from your CSV for Q2 + Q3
# 🔧 change this to any real id in summarization_with_metadata_clusters.csv
CASE_ID = "case_10670882_opinion_11137469"


# --------------------------------------
# AGENT 1: DOCTRINAL / RAG (STANDARD OF REVIEW)
# --------------------------------------
def run_review_agent(question: str) -> str:
    """
    Use the legal_rag tool directly to answer a doctrinal question.

    We combine the case descriptor with the question to help retrieval.
    """
    full_question = f"{CASE_QUERY}\n\nQuestion: {question}"

    result = legal_rag.invoke({
        "arguments": {
            "question": full_question,
            "mode": "rules",
            "k": 5,
        }
    })

    # legal_rag returns resp.content (a string), but invoke wraps it:
    # If it's already a string, just return it.
    return result if isinstance(result, str) else str(result)


# --------------------------------------
# MAIN DEMO
# --------------------------------------
if __name__ == "__main__":
    # ---------------- Q1: standard of review (RAG) ----------------
    q1 = "What standard of review applies in unemployment-benefit administrative appeals?"

    print("\n========== Q1: STANDARD OF REVIEW ==========\n")
    answer_q1 = run_review_agent(q1)
    print(answer_q1)

    # ---------------- Q2: cluster & topic (cluster_agent) ----------------
    print("\n========== Q2: CLUSTER & TOPIC ==========\n")
    cluster_explanation = explain_case_cluster(CASE_ID)
    print(cluster_explanation)

    # ---------------- Q3: metadata (metadata_agent) ----------------
    print("\n========== Q3: METADATA ==========\n")
    metadata_explanation = explain_case_metadata(CASE_ID)
    print(metadata_explanation)
