# agents/cluster_agent.py

from pathlib import Path
from typing import Optional

import pandas as pd

from agents.legal_tools import explore_topic

# --------------------------------------
# CONFIG
# --------------------------------------
DATA_DIR = Path("./data")
CASES_CSV = DATA_DIR / "summarization_with_metadata_clusters.csv"

ID_COL = "id"          # case id column
CLUSTER_COL = "cluster_id"
CASE_NAME_COL = "case_name"  # optional, if present


# --------------------------------------
# CORE LOGIC
# --------------------------------------
def load_cases_df() -> pd.DataFrame:
    """Load the clustered cases CSV."""
    if not CASES_CSV.exists():
        raise FileNotFoundError(
            f"Cases CSV not found at {CASES_CSV}. "
            "Make sure you ran the clustering pipeline first."
        )
    df = pd.read_csv(CASES_CSV)
    if ID_COL not in df.columns or CLUSTER_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{ID_COL}' and '{CLUSTER_COL}' in {CASES_CSV}, "
            f"but got: {list(df.columns)}"
        )
    return df


def find_cluster_id_for_case(df: pd.DataFrame, case_id: str) -> tuple[int, Optional[str]]:
    """
    Given a case_id, return (cluster_id, case_name).

    case_name may be None if the column doesn't exist.
    """
    # Try exact match on string
    rows = df[df[ID_COL].astype(str) == str(case_id)]

    if rows.empty:
        raise ValueError(f"No case found with {ID_COL} = '{case_id}' in {CASES_CSV}.")

    row = rows.iloc[0]

    cluster_id = int(row[CLUSTER_COL])

    case_name = None
    if CASE_NAME_COL in df.columns:
        val = row.get(CASE_NAME_COL)
        case_name = str(val) if pd.notna(val) else None

    return cluster_id, case_name


def explain_case_cluster(case_id: str) -> str:
    """
    1. Look up this case_id in the clustered CSV.
    2. Find its cluster_id.
    3. Call explore_topic({"cluster_id": cluster_id}).
    4. Return a friendly natural language explanation.
    """
    df = load_cases_df()
    cluster_id, case_name = find_cluster_id_for_case(df, case_id)

    # 🔧 FIX: wrap in "arguments"
    topic_info = explore_topic.invoke({
        "arguments": {
            "cluster_id": cluster_id
        }
    })

    # If explore_topic returned a plain string (error message), just surface it.
    if isinstance(topic_info, str):
        return (
            f"Case '{case_id}' is in cluster {cluster_id}, "
            f"but explore_topic returned an issue: {topic_info}"
        )

    label = topic_info.get("label", f"Cluster {cluster_id}")
    description = topic_info.get("description", "").strip()
    sample_case = topic_info.get("sample_case")

    # Build a friendly explanation
    lines = []

    if case_name:
        lines.append(f"Case **{case_name}** (id: `{case_id}`)")
    else:
        lines.append(f"Case id: `{case_id}`")

    lines.append(f"- Assigned to **Cluster {cluster_id} – {label}**")

    if description:
        lines.append(f"- This cluster is described as: _{description}_")

    if sample_case and (not case_name or sample_case != case_name):
        lines.append(f"- Example case from this cluster: **{sample_case}**")

    return "\n".join(lines)

# --------------------------------------
# DEMO / CLI ENTRY
# --------------------------------------
if __name__ == "__main__":
    # 🔧 Change this to a real case id from your CSV to test
    demo_case_id = "case_10670882_opinion_11137469"  # e.g., "case_123_opinion_1"

    try:
        explanation = explain_case_cluster(demo_case_id)
        print("\n========== CLUSTER EXPLANATION ==========\n")
        print(explanation)
    except Exception as e:
        print(f"Error: {e}")
