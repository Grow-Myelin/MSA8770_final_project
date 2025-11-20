import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "https://7effb179-5ca2-44cf-b618-404399f77d64.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Text_Analysis_final"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def get_points(query_text: str, limit: int = 3):
    """Retrieve top-k similar cases from Qdrant."""
    query_vec = embeddings.embed_query(query_text)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit,
        with_payload=True,
    )
    return response.points

def build_context(points, max_chars: int = 4000) -> str:
    """Concatenate retrieved texts into a single context string."""
    chunks = []
    total = 0

    for p in points:
        text = p.payload.get("text", "")
        if not text:
            continue
        if total + len(text) > max_chars:
            text = text[: max_chars - total]
        chunks.append(text)
        total += len(text)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."

# We look for the four modes on the court cases: overview, outcome , rules , reasoning.

def get_system_prompt(mode: str) -> str:
    base = (
        "You are a legal assistant working ONLY from the provided court-opinion context. "
        "If the context is insufficient to answer, reply exactly with: 'Insufficient context'.\n\n"
    )

    if mode == "overview":
        task = (
            "Task: Summarize the case and state the main legal issue in 3 to 5 sentences. "
            "Write clearly for a law student, but do not invent any facts."
        )
    elif mode == "outcome":
        task = (
            "Task: Explain what the court decided (outcome) and, if present, the procedural "
            "posture (e.g., appeal from trial court) and type of remedy or relief at issue."
        )
    elif mode == "rules":
        task = (
            "Task: Extract and clearly state the key legal rule(s) or test(s) the court applies "
            "in this case. Quote or paraphrase accurately, and mention the legal standard "
            "for the main doctrine if it appears (e.g., prescriptive easement)."
        )
    elif mode == "reasoning":
        task = (
            "Task: Explain the court's reasoning: why it reached its decision and how it "
            "handled the main arguments (e.g., permissive vs. adverse use). Focus on logic, "
            "not just restating the outcome."
        )
    else:
        task = "Task: Answer the question using only the context."

    return base + task

def answer(question: str, mode: str = "overview", k: int = 3) -> str:
    # 1. Retrieve
    points = get_points(question, limit=k)

    # 2. Build context
    context = build_context(points)

    # 3. Build messages
    system_prompt = get_system_prompt(mode)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"User question: {question}\n\nContext:\n{context}"
        ),
    ]

    # 4. Call LLM
    resp = llm.invoke(messages)
    return resp.content
if __name__ == "__main__":
    q = "Summarize the case where the court discusses prescriptive easement and private road access."
    print("=== OVERVIEW ===")
    print(answer(q, mode="overview"))

    print("\n=== OUTCOME ===")
    print(answer(q, mode="outcome"))
