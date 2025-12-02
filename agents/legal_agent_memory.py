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
# MAIN DEMO (NO MEMORY) + MEMORY SESSION
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

    # ==============================================================
    #   SHORT MEMORY SESSION (LIKE YOUR COURSE EXAMPLE)
    # ==============================================================

    print("\n========== MEMORY DEMO (2-TURN SESSION) ==========\n")

    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from agents.legal_tools import legal_rag, extract_metadata, explore_topic

    # ---- MODEL & TOOLS FOR MEMORY DEMO ----
    chat_model = ChatOllama(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        temperature=0,
    )

    Tools = [legal_rag, extract_metadata, explore_topic]

    system_message = (
        "You are a legal assistant working with three tools:\n"
        "- legal_rag(question, mode, k): use this for doctrinal questions, "
        "such as standards of review or key legal rules. Always include a "
        "'question' field when calling it.\n"
        "- explore_topic(cluster_id): use this when the user explicitly asks about a cluster id.\n"
        "- extract_metadata(text or question, k): use this to extract organizations, places, "
        "area_of_law, and remedy_type.\n\n"
        "If you need information from the corpus, call legal_rag first. "
        "Use the conversation history to handle follow-up questions."
    )

    # ---- MEMORY: CHECKPOINTER + GRAPH ----
    checkpointer = MemorySaver()

    graph = create_react_agent(
        model=chat_model,
        tools=Tools,
        checkpointer=checkpointer,
    )

    # Same thread_id across turns = shared memory
    cfg = {"configurable": {"thread_id": "legal-session-001"}}

    # ---- Turn 1: User asks a doctrinal question ----
    init_query = "What standard of review applies in unemployment-benefit administrative appeals?"

    init_messages = {
        "messages": [
            SystemMessage(content=system_message),
            HumanMessage(content=init_query),
        ]
    }

    print("---- Turn 1: Ask about standard of review ----\n")
    state1 = graph.invoke(init_messages, config=cfg)
    turn1_answer = state1["messages"][-1].content
    print(turn1_answer)

    # ---- Turn 2: Follow-up that relies on memory ----
    follow_up = {
        "messages": [
            HumanMessage(
                content=(
                    "Based on the same type of unemployment-benefit appeal you just considered, "
                    "summarize in 2–3 sentences what remedies are typically involved and "
                    "which area of law this falls under."
                )
            )
        ]
    }

    print("\n---- Turn 2: Follow-up using prior context ----\n")
    state2 = graph.invoke(follow_up, config=cfg)
    turn2_answer = state2["messages"][-1].content
    print(turn2_answer)

    print("\n==== Memory Demo Complete: Same thread_id kept the context across turns ====\n")
