# agents/legal_tools.py
"""
Legal RAG tools for retrieving and analyzing case law.

Supports both chunked and full-document retrieval modes.
Chunked mode provides better context by using structure-aware chunks
that are used in full without truncation.
"""

import os
import pandas as pd
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
import spacy
from dotenv import load_dotenv

load_dotenv()

# -------------------------
#  GLOBAL CLIENTS & MODELS
# -------------------------
QDRANT_URL = "https://a1c4fe30-e27e-4b18-9384-f1fa8b530103.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection names
COLLECTION_NAME_CHUNKED = "Text_Analysis_chunked"
COLLECTION_NAME_FULL = "Text_Analysis_final"

# Default to chunked collection (better retrieval)
# Set USE_CHUNKED_COLLECTION = False to use full documents
USE_CHUNKED_COLLECTION = True

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# -------------------------
#        RAG HELPERS
# -------------------------
def get_collection_name() -> str:
    """
    Get the active collection name based on configuration.

    Falls back to full-document collection if chunked collection
    doesn't exist or is unavailable.
    """
    if USE_CHUNKED_COLLECTION:
        try:
            collections = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME_CHUNKED in collections:
                return COLLECTION_NAME_CHUNKED
            else:
                print(f"Warning: Chunked collection '{COLLECTION_NAME_CHUNKED}' not found, "
                      f"falling back to '{COLLECTION_NAME_FULL}'")
                return COLLECTION_NAME_FULL
        except Exception as e:
            print(f"Warning: Could not check collections ({e}), "
                  f"falling back to '{COLLECTION_NAME_FULL}'")
            return COLLECTION_NAME_FULL
    return COLLECTION_NAME_FULL


def get_points(query_text: str, limit: int = 5, collection: str = None):
    """
    Query Qdrant for relevant points.

    Args:
        query_text: The query to embed and search
        limit: Maximum number of results to return
        collection: Collection name (defaults to configured collection)

    Returns:
        List of Qdrant points with payloads
    """
    if collection is None:
        collection = get_collection_name()

    query_vec = embeddings.embed_query(query_text)
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=limit,
        with_payload=True
    )
    return response.points


def build_context(points, max_total: int = 12000, max_chunks_per_case: int = 2):
    """
    Build context from retrieved chunks/documents.

    IMPORTANT: Uses chunks IN FULL - no truncation.
    Stops adding when budget is reached rather than truncating content.

    Args:
        points: List of Qdrant points
        max_total: Maximum total characters (default 12000)
        max_chunks_per_case: Max chunks from same case for diversity (default 2)

    Returns:
        Formatted context string with case metadata
    """
    chunks = []
    total = 0
    case_chunk_counts = {}  # Track chunks per case

    for p in points:
        text = p.payload.get("text") or p.payload.get("text_clean", "")
        if not text:
            continue

        case_id = p.payload.get("case_id", "unknown")
        case_name = p.payload.get("case_name", "Unknown Case")
        chunk_type = p.payload.get("chunk_type", "")

        # Limit chunks per case to ensure diversity
        case_chunk_counts[case_id] = case_chunk_counts.get(case_id, 0) + 1
        if case_chunk_counts[case_id] > max_chunks_per_case:
            continue

        # Check if adding this chunk would exceed budget
        # STOP adding rather than truncating
        if total + len(text) > max_total:
            break

        # Add chunk with metadata
        chunks.append({
            "text": text,
            "case_id": case_id,
            "case_name": case_name,
            "chunk_type": chunk_type
        })
        total += len(text)

    if not chunks:
        return "No relevant context found."

    # Format with metadata headers
    formatted = []
    for c in chunks:
        # Include chunk type if available (helps LLM understand context)
        if c["chunk_type"]:
            header = f"[{c['case_name']} - {c['chunk_type']}]"
        else:
            header = f"[{c['case_name']}]"
        formatted.append(f"{header}\n{c['text']}")

    return "\n\n---\n\n".join(formatted)


def get_system_prompt(mode: str) -> str:
    """Get system prompt for the specified analysis mode."""
    base = (
        "You are a legal assistant working ONLY from the provided context. "
        "The context contains excerpts from court opinions, organized by case name and section type. "
        "Base your answer strictly on this context. "
        "If the context is insufficient to answer the question, say so clearly.\n\n"
    )
    modes = {
        "overview": "Summarize the case(s) and identify the main legal issue(s).",
        "outcome": "Describe the court's decision and the type of relief granted or denied.",
        "rules": "Extract the key legal rules, tests, or standards applied by the court.",
        "reasoning": "Explain the court's reasoning and how it handled the parties' arguments."
    }
    return base + modes.get(mode, "Answer using only the context provided.")


# -------------------------
#   NER + METADATA HELPERS
# -------------------------
def run_spacy(text: str):
    """Extract organizations and places using spaCy NER."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    header = text[:2000]
    doc = nlp(header)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return orgs, places


def classify_area_of_law(text: str, orgs):
    """Classify area of law based on keywords in text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()

    # Criminal law
    if any(kw in t for kw in [
        "defendant was convicted", "sentenced to", "indictment", "felony",
        "misdemeanor", "prison", "probation", "parole", "murder", "assault"
    ]):
        return "criminal"

    # Constitutional law
    if any(kw in t for kw in [
        "first amendment", "fourth amendment", "due process",
        "equal protection", "constitutional right", "civil rights"
    ]):
        return "constitutional"

    # Environmental law
    if any(kw in t for kw in [
        "environmental protection", "clean air", "clean water", "epa", "pollution"
    ]):
        return "environmental"

    # Immigration law
    if any(kw in t for kw in [
        "immigration", "deportation", "asylum", "visa", "removal proceedings"
    ]):
        return "immigration"

    # Administrative law
    if any(kw in t for kw in [
        "administrative review", "agency decision", "regulatory", "nrc", "fcc"
    ]):
        return "administrative"

    # Employment law
    if any(kw in t for kw in [
        "unemployment benefits", "employment security", "wrongful termination",
        "workplace discrimination", "osha", "labor relations"
    ]):
        return "employment"

    # Property law
    if any(kw in t for kw in [
        "easement", "quiet title", "adverse possession", "eminent domain",
        "zoning", "foreclosure", "real property"
    ]):
        return "property"

    # Family law
    if any(kw in t for kw in [
        "divorce", "custody", "child support", "visitation", "alimony", "adoption"
    ]):
        return "family"

    # Torts
    if any(kw in t for kw in [
        "negligence", "duty of care", "personal injury", "malpractice",
        "defamation", "wrongful death"
    ]):
        return "torts"

    # Contracts
    if any(kw in t for kw in [
        "breach of contract", "contract dispute", "promissory", "warranty"
    ]):
        return "contracts"

    # Bankruptcy
    if any(kw in t for kw in ["bankruptcy", "chapter 7", "chapter 11", "debtor"]):
        return "bankruptcy"

    return None


def classify_remedy_type(text: str):
    """Classify types of remedies mentioned in text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    r = []
    if "injunction" in t or "restraining order" in t:
        r.append("injunctive relief")
    if "declaratory" in t:
        r.append("declaratory judgment")
    if "damages" in t:
        r.append("damages")
    if "unemployment benefits" in t:
        r.append("benefits")
    if "administrative review" in t:
        r.append("administrative review")
    # dedupe while preserving order
    return list(dict.fromkeys(r))


# -------------------------
#        RAG TOOL
# -------------------------
@tool
def legal_rag(arguments: dict):
    """
    Retrieve relevant case passages from Qdrant and answer a legal question.

    Uses structure-aware chunks for better context (when available).
    Chunks are used in full without truncation.

    arguments = {
        "question": str,
        "mode": str = "overview" | "outcome" | "rules" | "reasoning",
        "k": int = 5 (number of chunks to retrieve)
    }
    """
    question = arguments.get("question")
    mode = arguments.get("mode", "overview")
    k = arguments.get("k", 5)

    # Retrieve more chunks for better coverage, build_context will limit
    points = get_points(question, limit=k * 2)
    context = build_context(points, max_total=12000, max_chunks_per_case=2)
    system = get_system_prompt(mode)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")
    ]

    resp = llm.invoke(messages)
    return resp.content


# -------------------------
#   METADATA TOOL (RAG-AWARE)
# -------------------------
@tool
def extract_metadata(arguments: dict):
    """
    Extract metadata (organizations, places, area_of_law, remedy_type).

    You can either:
      - pass raw case text: {"text": "..."}
      - OR let the tool retrieve context from Qdrant:
            {"question": "...", "k": 5}

    arguments = {
        "text": str (optional),
        "question": str (optional),
        "k": int = 5 (optional, only with question)
    }
    """
    text = arguments.get("text")

    # If no direct text, pull it via RAG using a question
    if not text:
        question = arguments.get("question")
        if not question:
            return {
                "error": "Provide either 'text' or 'question' for metadata extraction."
            }
        k = arguments.get("k", 5)
        points = get_points(question, limit=k)
        text = build_context(points, max_total=8000)

    orgs, places = run_spacy(text)
    area = classify_area_of_law(text, orgs)
    remedy = classify_remedy_type(text)

    return {
        "organizations": orgs,
        "places": places,
        "area_of_law": area,
        "remedy_type": remedy
    }


# -------------------------
#       TOPIC TOOL
# -------------------------
try:
    # cluster_labels.csv has: cluster_id, label, description
    labels_df = pd.read_csv("./data/cluster_labels.csv")

    # cluster_stats.csv has: cluster_id, n_cases, sample_case, area_mode
    stats_df = pd.read_csv("./data/cluster_stats.csv")

    # Merge to get label + description + sample_case in one table
    df_clusters = labels_df.merge(
        stats_df[["cluster_id", "sample_case"]],
        on="cluster_id",
        how="left"
    )

    # Ensure cluster_id is int
    if "cluster_id" in df_clusters.columns:
        df_clusters["cluster_id"] = df_clusters["cluster_id"].astype(int)
except Exception:
    df_clusters = pd.DataFrame()


@tool
def explore_topic(arguments: dict):
    """
    Look up information about a cluster.

    arguments = { "cluster_id": int }

    Returns:
        {
            "cluster_id": int,
            "label": str,
            "description": str,
            "sample_case": str | None
        }
    """
    if df_clusters.empty:
        return "Cluster metadata not available (cluster_labels.csv / cluster_stats.csv missing or unreadable)."

    cluster_id = arguments.get("cluster_id")

    if cluster_id is None:
        return "You must provide 'cluster_id' in arguments, e.g. {\"cluster_id\": 3}."

    # Make sure we compare ints
    try:
        cluster_id = int(cluster_id)
    except ValueError:
        return f"Invalid cluster_id: {cluster_id}"

    if cluster_id not in df_clusters["cluster_id"].tolist():
        return f"Cluster {cluster_id} not found in cluster_labels/cluster_stats."

    row = df_clusters[df_clusters["cluster_id"] == cluster_id].iloc[0]

    return {
        "cluster_id": int(row["cluster_id"]),
        "label": row["label"],
        "description": row["description"],
        "sample_case": row.get("sample_case", None),
    }
