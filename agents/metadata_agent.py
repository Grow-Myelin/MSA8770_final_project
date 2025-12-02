# agents/metadata_agent.py

from pathlib import Path
from typing import Optional

import pandas as pd

from agents.legal_tools import extract_metadata

# --------------------------------------
# CONFIG
# --------------------------------------
DATA_DIR = Path("./data")
CASES_CSV = DATA_DIR / "summarization_with_metadata_clusters.csv"

ID_COL = "id"             # case id column
TEXT_COL = "text_clean"   # column that holds the opinion text
CASE_NAME_COL = "case_name"  # optional, if present


# --------------------------------------
# CORE HELPERS
# --------------------------------------
def load_cases_df() -> pd.DataFrame:
    """Load the clustered cases CSV (or any CSV with id + text_clean)."""
    if not CASES_CSV.exists():
        raise FileNotFoundError(
            f"Cases CSV not found at {CASES_CSV}. "
            "Make sure you ran the preprocessing/clustering pipeline first."
        )
    df = pd.read_csv(CASES_CSV)
    missing = [col for col in [ID_COL, TEXT_COL] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} in {CASES_CSV}, "
            f"but got: {list(df.columns)}"
        )
    return df


def get_case_row(df: pd.DataFrame, case_id: str) -> pd.Series:
    """Return the row for a given case_id, or raise if not found."""
    rows = df[df[ID_COL].astype(str) == str(case_id)]
    if rows.empty:
        raise ValueError(f"No case found with {ID_COL} = '{case_id}' in {CASES_CSV}.")
    return rows.iloc[0]


def explain_case_metadata(case_id: str) -> str:
    """
    For a given case_id:

    1. Look up the case in the CSV.
    2. Get its text_clean.
    3. Call extract_metadata({"text": ...}).
    4. Return a friendly explanation of orgs, places, area_of_law, remedy_type.
    """
    df = load_cases_df()
    row = get_case_row(df, case_id)

    text = row.get(TEXT_COL, "")
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Optional: grab case_name if available
    case_name: Optional[str] = None
    if CASE_NAME_COL in df.columns:
        val = row.get(CASE_NAME_COL)
        if pd.notna(val):
            case_name = str(val)

    # Tool expects {"arguments": {...}} because signature is `arguments: dict`
    meta = extract_metadata.invoke({
        "arguments": {
            "text": text
        }
    })

    # If the tool returns an error string, just surface it
    if isinstance(meta, str):
        header = f"Case id: `{case_id}`"
        if case_name:
            header = f"Case **{case_name}** (id: `{case_id}`)"
        return header + f"\n\nMetadata tool returned: {meta}"

    orgs = meta.get("organizations", [])
    places = meta.get("places", [])
    area = meta.get("area_of_law", None)
    remedy = meta.get("remedy_type", [])

    # Build explanation
    lines = []

    if case_name:
        lines.append(f"Case **{case_name}** (id: `{case_id}`)")
    else:
        lines.append(f"Case id: `{case_id}`")

    if area:
        lines.append(f"- **Area of law:** {area}")
    else:
        lines.append("- **Area of law:** (not clearly identified)")

    if remedy:
        remedies_str = ", ".join(remedy)
        lines.append(f"- **Remedy type(s):** {remedies_str}")
    else:
        lines.append("- **Remedy type(s):** none detected")

    if orgs:
        seen = set()
        orgs_clean = []
        for o in orgs:
            if o not in seen:
                orgs_clean.append(o)
                seen.add(o)
        lines.append(f"- **Organizations mentioned:** {', '.join(orgs_clean)}")

    if places:
        seen = set()
        places_clean = []
        for p in places:
            if p not in seen:
                places_clean.append(p)
                seen.add(p)
        lines.append(f"- **Places mentioned:** {', '.join(places_clean)}")

    if not orgs and not places:
        lines.append("- **Organizations / places:** none clearly extracted from the header section.")

    return "\n".join(lines)


# --------------------------------------
# SIMPLE TEST ENTRY (no CLI)
# --------------------------------------
if __name__ == "__main__":
    # 🔧 Just change this to whatever case id you want to test
    demo_case_id = "case_10670882_opinion_11137469"

    try:
        explanation = explain_case_metadata(demo_case_id)
        print("\n========== CASE METADATA ==========\n")
        print(explanation)
    except Exception as e:
        print(f"Error: {e}")
