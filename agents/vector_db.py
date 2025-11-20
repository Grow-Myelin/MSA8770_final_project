
import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
#from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL ="https://7effb179-5ca2-44cf-b618-404399f77d64.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Text_Analysis_final"



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
data_dir=Path("./data")
csv_name="summarization_extract_clean.csv"
csv_path = data_dir / csv_name
# 1. Load the raw CSV saved by fetch_data.py

df = pd.read_csv(csv_path)
text = df["text_clean"].fillna("").tolist()

vectors = embeddings.embed_documents(text)
dim = len(vectors[0])
print("Dimension: ", dim)

# For demo-purposes
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print("Deleting collection: ", COLLECTION_NAME)
    client.delete_collection(COLLECTION_NAME)

client.create_collection(collection_name=COLLECTION_NAME, 
vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

points = []

for i, row in df.iterrows():
    points.append(
        PointStruct(
            id=i,  # simple integer ID for Qdrant
            vector=vectors[i],
            payload={
                "case_id": row["id"],       #  real case id stays here
                "case_name": row["case_name"],
                "text": row["text_clean"],
            },
        )
    )



BATCH_SIZE = 100
total_points = len(points)

for start in range(0, total_points, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_points)
    batch = points[start:end]
    print(f"Upserting points {start}–{end} / {total_points}")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )


print(f"Finished upserting {total_points} points into '{COLLECTION_NAME}'.")
