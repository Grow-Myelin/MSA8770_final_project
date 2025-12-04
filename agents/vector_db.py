"""
vector_db.py - Populate Qdrant vector database with legal documents

Supports two modes:
- Chunked (default): Structure-aware chunking for better retrieval
- Full document: Original behavior for backward compatibility

Usage:
    python agents/vector_db.py              # Chunked mode (recommended)
    python agents/vector_db.py --no-chunk   # Full document mode
"""

import os
import sys
import argparse
import pandas as pd

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configuration
QDRANT_URL = "https://a1c4fe30-e27e-4b18-9384-f1fa8b530103.us-east-1-1.aws.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection names
COLLECTION_NAME_CHUNKED = "Text_Analysis_chunked"
COLLECTION_NAME_FULL = "Text_Analysis_final"  # Original collection

# Data paths
DATA_DIR = Path("./data")
CSV_NAME = "summarization_extract_clean.csv"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def create_collection(client: QdrantClient, collection_name: str, dim: int):
    """Create or recreate a Qdrant collection."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    print(f"Creating collection: {collection_name} (dim={dim})")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )


def upsert_batched(client: QdrantClient, collection_name: str,
                   points: list, batch_size: int = 100):
    """Upsert points in batches."""
    total = len(points)
    for start in tqdm(range(0, total, batch_size), desc="Upserting"):
        end = min(start + batch_size, total)
        batch = points[start:end]
        client.upsert(collection_name=collection_name, points=batch)


def populate_chunked(df: pd.DataFrame, client: QdrantClient,
                     embeddings: OpenAIEmbeddings):
    """
    Populate vector database with chunked documents.

    Each document is split into smaller chunks based on its structure.
    This enables better retrieval of relevant passages.
    """
    from agents.chunker import chunk_document, analyze_document

    print("\n=== Chunked Ingestion Mode ===")
    print(f"Processing {len(df)} documents...")

    # Chunk all documents
    all_chunks = []
    chunk_stats = {"total_docs": len(df), "total_chunks": 0, "doc_types": {}}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
        text = row["text_clean"] if pd.notna(row["text_clean"]) else ""
        case_id = str(row["id"])
        case_name = str(row["case_name"]) if pd.notna(row["case_name"]) else "Unknown"

        if not text.strip():
            continue

        # Analyze for stats
        analysis = analyze_document(text)
        doc_type = analysis["doc_type"]
        chunk_stats["doc_types"][doc_type] = chunk_stats["doc_types"].get(doc_type, 0) + 1

        # Chunk the document
        chunks = chunk_document(text, case_id, case_name)
        all_chunks.extend(chunks)

    chunk_stats["total_chunks"] = len(all_chunks)
    print(f"\nChunking complete:")
    print(f"  Documents: {chunk_stats['total_docs']}")
    print(f"  Total chunks: {chunk_stats['total_chunks']}")
    print(f"  Avg chunks/doc: {chunk_stats['total_chunks'] / chunk_stats['total_docs']:.1f}")
    print(f"  Document types: {chunk_stats['doc_types']}")

    # Embed all chunks
    print("\nEmbedding chunks...")
    chunk_texts = [c["text"] for c in all_chunks]

    # Embed in batches to avoid rate limits
    all_vectors = []
    for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), desc="Embedding"):
        batch = chunk_texts[i:i + BATCH_SIZE]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

    dim = len(all_vectors[0])
    print(f"Embedding dimension: {dim}")

    # Create collection
    create_collection(client, COLLECTION_NAME_CHUNKED, dim)

    # Create points
    points = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, all_vectors)):
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload=chunk
        ))

    # Upsert
    upsert_batched(client, COLLECTION_NAME_CHUNKED, points, BATCH_SIZE)

    print(f"\nFinished: {len(points)} chunks in '{COLLECTION_NAME_CHUNKED}'")
    return chunk_stats


def populate_full_documents(df: pd.DataFrame, client: QdrantClient,
                            embeddings: OpenAIEmbeddings):
    """
    Populate vector database with full documents (original behavior).

    Each document is stored as a single embedding.
    """
    print("\n=== Full Document Mode ===")
    print(f"Processing {len(df)} documents...")

    # Get texts
    texts = df["text_clean"].fillna("").tolist()

    # Embed
    print("Embedding documents...")
    vectors = embeddings.embed_documents(texts)
    dim = len(vectors[0])
    print(f"Embedding dimension: {dim}")

    # Create collection
    create_collection(client, COLLECTION_NAME_FULL, dim)

    # Create points
    points = []
    for i, row in df.iterrows():
        points.append(PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "case_id": str(row["id"]),
                "case_name": str(row["case_name"]) if pd.notna(row["case_name"]) else "Unknown",
                "text": row["text_clean"] if pd.notna(row["text_clean"]) else "",
            }
        ))

    # Upsert
    upsert_batched(client, COLLECTION_NAME_FULL, points, BATCH_SIZE)

    print(f"\nFinished: {len(points)} documents in '{COLLECTION_NAME_FULL}'")


def main():
    parser = argparse.ArgumentParser(description="Populate Qdrant vector database")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Use full documents instead of chunking")
    parser.add_argument("--csv", type=str, default=str(DATA_DIR / CSV_NAME),
                        help="Path to CSV file")
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv)
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} documents")

    # Initialize clients
    print(f"\nConnecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Populate
    if args.no_chunk:
        populate_full_documents(df, client, embeddings)
    else:
        populate_chunked(df, client, embeddings)


if __name__ == "__main__":
    main()
