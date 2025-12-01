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
    header = text[:2000]
    doc = nlp(header)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return orgs, places


def classify_area_of_law(text: str, orgs: list[str]):
    t = text.lower()
    o = [x.lower() for x in orgs]

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
    return list(dict.fromkeys(r))  # dedupe


# -------------------------
#        RAG TOOL
# -------------------------
@tool
def legal_rag(question: str, mode: str = "overview", k: int = 3):
    """
    Retrieve relevant cases from Qdrant and answer using RAG.
    Modes: overview, outcome, rules, reasoning.
    """
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
#       NER TOOL
# -------------------------
@tool
def extract_metadata(text: str):
    """
    Extract orgs, places, area of law, and remedy types from a case.
    """
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
#     TOPIC / CLUSTER TOOL
# -------------------------
# These two lookup tables must already exist (built in notebook)
try:
    df_clusters = pd.read_csv("./data/cluster_labels.csv")
except:
    df_clusters = pd.DataFrame()

@tool
def explore_topic(cluster_id: int):
    """
    Return the cluster label + description + a few sample cases.
    """
    if cluster_id not in df_clusters["cluster_id"].tolist():
        return f"Cluster {cluster_id} not found."

    row = df_clusters[df_clusters["cluster_id"] == cluster_id].iloc[0]

    return {
        "cluster_id": int(row["cluster_id"]),
        "label": row["cluster_label"],
        "description": row["cluster_description"],
        "sample_case": row["sample_case"]
    }
