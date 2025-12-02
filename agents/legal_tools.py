# agents/legal_tools.py

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
QDRANT_URL = "https://7effb179-5ca2-44cf-b618-404399f77d64.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Text_Analysis_final"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# -------------------------
#        RAG HELPERS
# -------------------------
def get_points(query_text: str, limit: int = 3):
    query_vec = embeddings.embed_query(query_text)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit,
        with_payload=True
    )
    return response.points


def build_context(points, max_chars: int = 3500):
    """
    Build a text context from Qdrant points.

    Tries payload["text"], falls back to payload["text_clean"] if needed.
    """
    chunks = []
    total = 0
    for p in points:
        text = (
            p.payload.get("text")
            or p.payload.get("text_clean", "")
        )
        if not text:
            continue
        if total + len(text) > max_chars:
            text = text[: max_chars - total]
        chunks.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n---\n".join(chunks) if chunks else "No relevant context."


def get_system_prompt(mode: str) -> str:
    base = (
        "You are a legal assistant working ONLY from the provided context. "
        "If the context is insufficient, answer exactly: 'Insufficient context'.\n\n"
    )
    modes = {
        "overview": "Summarize the case and identify the main legal issue.",
        "outcome": "Describe the court's decision and the type of relief.",
        "rules": "Extract the key legal rules or tests used.",
        "reasoning": "Explain the court’s reasoning and argument handling."
    }
    return base + modes.get(mode, "Answer using only the context.")


# -------------------------
#   NER + METADATA HELPERS
# -------------------------
def run_spacy(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    header = text[:2000]
    doc = nlp(header)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return orgs, places


def classify_area_of_law(text: str, orgs):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()

    if "easement" in t or "quiet title" in t:
        return "property"
    if "unemployment benefits" in t or "employment security" in t:
        return "employment"
    if "administrative review" in t:
        return "administrative"
    if "defendant was convicted" in t or "felony" in t:
        return "criminal"
    if "custody" in t or "child support" in t:
        return "family"
    if "negligence" in t or "duty of care" in t:
        return "torts"
    if "contract" in t or "lease agreement" in t:
        return "contracts"
    return None


def classify_remedy_type(text: str):
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

    arguments = {
        "question": str,
        "mode": str = "overview" | "outcome" | "rules" | "reasoning",
        "k": int = 3
    }
    """
    question = arguments.get("question")
    mode = arguments.get("mode", "overview")
    k = arguments.get("k", 3)

    points = get_points(question, limit=k)
    context = build_context(points)
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
        text = build_context(points)

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
