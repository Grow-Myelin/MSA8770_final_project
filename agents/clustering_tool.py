import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import KMeans

# --------------------
# CONFIG
# --------------------
DATA_DIR = Path("./data")
INPUT_CSV = DATA_DIR / "summarization_with_metadata.csv"
OUTPUT_CSV = DATA_DIR / "summarization_with_metadata_clusters.csv"
CLUSTER_STATS_CSV = DATA_DIR / "cluster_stats.csv"

TEXT_COL = "text_clean"
ID_COL = "id"

N_CLUSTERS = 10  # you can change this to 8, 12, etc.


def main():
    # ---- 1. Load env + CSV ----
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment")

    print(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in CSV")

    texts = df[TEXT_COL].fillna("").tolist()
    print(f"Loaded {len(texts)} documents for clustering.")

    # ---- 2. Compute embeddings ----
    print("Creating embeddings with text-embedding-3-small ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # This returns a list of 1536-dim vectors
    vectors = embeddings.embed_documents(texts)
    X = np.array(vectors, dtype="float32")
    print(f"Embeddings shape: {X.shape}")  # (n_docs, 1536)

    # ---- 3. Run KMeans clustering ----
    print(f"Running KMeans with n_clusters={N_CLUSTERS} ...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(X)
    print("Clustering finished.")

    # ---- 4. Attach cluster IDs to DataFrame ----
    df["cluster_id"] = cluster_ids

    # ---- 5. Basic cluster stats ----
    print("Building cluster stats...")
    cluster_counts = (
        df.groupby("cluster_id")
        .agg(
            n_cases=("cluster_id", "size"),
            sample_case=("case_name", "first") if "case_name" in df.columns else (ID_COL, "first"),
            area_mode=("area_of_law", lambda x: x.value_counts(dropna=True).index[0] if len(x.dropna()) > 0 else None),
        )
        .reset_index()
    )

    print("Cluster summary:")
    print(cluster_counts)

    # Save outputs
    print(f"Saving clustered data to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saving cluster stats to: {CLUSTER_STATS_CSV}")
    cluster_counts.to_csv(CLUSTER_STATS_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
