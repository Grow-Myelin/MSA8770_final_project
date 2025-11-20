import os
from pathlib import Path
import random
import textwrap

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------
# CONFIG
# --------------------
DATA_DIR = Path("./data")
INPUT_CSV = DATA_DIR / "summarization_with_metadata_clusters.csv"
OUTPUT_LABELS_CSV = DATA_DIR / "cluster_labels.csv"
OUTPUT_FULL_CSV = DATA_DIR / "summarization_with_metadata_clusters_labeled.csv"

TEXT_COL = "text_clean"
CASE_COL = "case_name"  # fallback to "id" if not present
CLUSTER_COL = "cluster_id"

# How many example cases per cluster to show the LLM
N_EXAMPLES_PER_CLUSTER = 5
# Max characters per text snippet sent to LLM
MAX_SNIPPET_CHARS = 800


# --------------------
# LLM HELPER
# --------------------
def make_llm():
    """Create the ChatOpenAI client."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )


def build_cluster_prompt(cluster_id: int, df_cluster: pd.DataFrame) -> str:
    """
    Build a prompt summarizing a single cluster:
    - area_of_law distribution
    - remedy_type distribution
    - a few example cases with short text snippets
    """
    # Basic stats
    total = len(df_cluster)

    # Area of law distribution
    area_counts = (
        df_cluster["area_of_law"]
        .fillna("None")
        .value_counts()
        .to_dict()
    )

    # Remedy type distribution
    # remedy_type is a list-ish string, so we normalize a bit
    remedy_flat = []
    for v in df_cluster["remedy_type"].dropna():
        if isinstance(v, str):
            # Likely something like "['injunctive relief', 'damages']"
            # We'll just extract words between quotes
            # or split on comma as a cheap approximation
            cleaned = (
                v.replace("[", "")
                 .replace("]", "")
                 .replace("'", "")
                 .strip()
            )
            if cleaned:
                remedy_flat.extend([x.strip() for x in cleaned.split(",") if x.strip()])
        elif isinstance(v, list):
            remedy_flat.extend(v)

    remedy_counts = {}
    for r in remedy_flat:
        remedy_counts[r] = remedy_counts.get(r, 0) + 1

    # Sample a few example cases
    # random.sample can't handle smaller than population, so we min()
    n_examples = min(N_EXAMPLES_PER_CLUSTER, total)
    examples = df_cluster.sample(n=n_examples, random_state=42)

    # Build examples text
    case_col = CASE_COL if CASE_COL in df_cluster.columns else "id"
    example_blocks = []
    for _, row in examples.iterrows():
        case_name = str(row.get(case_col, "Unknown Case"))
        text_full = str(row.get(TEXT_COL, ""))
        snippet = text_full[:MAX_SNIPPET_CHARS].replace("\n", " ")
        example_blocks.append(
            f"Case: {case_name}\nSnippet: {snippet}"
        )

    examples_text = "\n\n".join(example_blocks)

    # Format stats
    area_str = ", ".join([f"{k}: {v}" for k, v in area_counts.items()])
    remedy_str = ", ".join([f"{k}: {v}" for k, v in remedy_counts.items()]) or "None detected"

    prompt = f"""
You are helping label a cluster of legal appellate cases from U.S. state courts.

Each cluster groups together cases that are semantically similar.

Your task:
1. Give this cluster a SHORT label (3–7 words), like:
   - "Property – Prescriptive Easements & Access"
   - "Criminal Appeals – Sentencing Issues"
   - "Employment – Unemployment Benefits Review"

2. Give ONE short descriptive sentence (max 25 words) explaining what kinds of cases appear in this cluster.

3. Focus on *substantive theme*, not jurisdiction.

Information about this cluster:

- Cluster ID: {cluster_id}
- Number of cases: {total}

- area_of_law distribution:
  {area_str}

- remedy_type distribution:
  {remedy_str}

Here are some example cases from this cluster:

{examples_text}

Return your answer in this EXACT JSON format (no extra text):

{{
  "label": "...",
  "description": "..."
}}
    """
    # Dedent for nicer logs, but LLM doesn't care
    return textwrap.dedent(prompt).strip()


def ask_llm_for_label(llm: ChatOpenAI, cluster_id: int, df_cluster: pd.DataFrame) -> dict:
    """Call the LLM to get a label + description for a cluster."""
    prompt = build_cluster_prompt(cluster_id, df_cluster)
    messages = [
        SystemMessage(content="You are an expert legal data analyst."),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    content = resp.content

    # Very simple JSON extraction; for a class project, this is fine.
    # We assume the model follows instructions reasonably well.
    import json

    try:
        parsed = json.loads(content)
        label = parsed.get("label", "").strip()
        description = parsed.get("description", "").strip()
    except Exception:
        # Fallback: try to salvage something, or mark as unknown
        label = f"Cluster {cluster_id} (unparsed)"
        description = content.strip()[:200]

    return {
        "cluster_id": cluster_id,
        "label": label,
        "description": description,
    }


# --------------------
# MAIN
# --------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    print(f"Loading clustered data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if CLUSTER_COL not in df.columns:
        raise ValueError(f"Column '{CLUSTER_COL}' not found in CSV.")

    cluster_ids = sorted(df[CLUSTER_COL].unique().tolist())
    print(f"Found {len(cluster_ids)} clusters: {cluster_ids}")

    llm = make_llm()

    label_rows = []

    for cid in cluster_ids:
        df_cluster = df[df[CLUSTER_COL] == cid]
        print(f"\nLabeling cluster {cid} with {len(df_cluster)} cases...")
        info = ask_llm_for_label(llm, cid, df_cluster)
        print(f"  → Label: {info['label']}")
        print(f"  → Description: {info['description']}")
        label_rows.append(info)

    labels_df = pd.DataFrame(label_rows)

    print(f"\nSaving cluster labels to: {OUTPUT_LABELS_CSV}")
    labels_df.to_csv(OUTPUT_LABELS_CSV, index=False)

    # Merge back labels into full dataset
    df = df.merge(labels_df, how="left", left_on=CLUSTER_COL, right_on="cluster_id")

    # Avoid duplicate column name
    df = df.rename(columns={
        "label": "cluster_label",
        "description": "cluster_description",
    })

    # Optionally drop the duplicate cluster_id column from labels_df
    # (since df already has CLUSTER_COL)
    if "cluster_id_y" in df.columns:
        df = df.drop(columns=["cluster_id_y"])
    if "cluster_id_x" in df.columns:
        df = df.rename(columns={"cluster_id_x": CLUSTER_COL})

    print(f"Saving full labeled dataset to: {OUTPUT_FULL_CSV}")
    df.to_csv(OUTPUT_FULL_CSV, index=False)

    print("Done labeling clusters.")


if __name__ == "__main__":
    main()
