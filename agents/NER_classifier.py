
import os
import pandas as pd
import spacy
from pathlib import Path

# ---------
# CONFIG
# ---------
DATA_DIR = Path("./data")
INPUT_CSV = DATA_DIR / "summarization_extract_clean.csv"
OUTPUT_CSV = DATA_DIR / "summarization_with_metadata.csv"

TEXT_COL = "text_clean"   # your existing column
ID_COL = "id"             # case id column (e.g. case_xxx_opinion_xxx)


# ---------
# LOAD SPACY MODEL
# ---------
print("Loading spaCy model en_core_web_sm ...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")


# ---------
# NER HELPER
# ---------
def run_spacy(text: str):
    """
    Run spaCy NER on the first part of the opinion text.

    Returns:
        orgs: list of ORG entity strings
        places: list of GPE/LOC entity strings
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Use only the first N characters (header + intro is usually enough)
    header = text[:2000]
    doc = nlp(header)

    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]

    return orgs, places


# ---------
# AREA OF LAW CLASSIFIER (RULE-BASED)
# ---------
def classify_area_of_law(text: str, orgs: list[str]) -> str | None:
    """
    Very rough heuristic classifier for area of law.
    You can refine this once you inspect more cases.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text.lower()

    orgs_lower = [o.lower() for o in orgs]

    # Property law
    if ("prescriptive easement" in t
        or "easement by prescription" in t
        or "easement" in t and "landlocked" in t
        or "quiet title" in t):
        return "property"

    # Employment / Unemployment / Labor
    if ("unemployment benefits" in t
        or "employment security" in t
        or "wrongful termination" in t
        or any("employment security" in o for o in orgs_lower)):
        return "employment"

    # Administrative / agency review
    if ("administrative review" in t
        or "board of review" in t
        or "agency decision" in t
        or "administrative agency" in t):
        return "administrative"

    # Criminal
    if ("defendant was convicted" in t
        or "sentenced to" in t
        or "indictment" in t
        or "felony" in t):
        return "criminal"

    # Family law
    if ("dissolution of marriage" in t
        or "custody" in t
        or "child support" in t
        or "visitation" in t):
        return "family"

    # Torts
    if ("negligence" in t
        or "duty of care" in t
        or "personal injury" in t
        or "tort" in t):
        return "torts"

    # Contracts
    if ("breach of contract" in t
        or "contract dispute" in t
        or "lease agreement" in t
        or "promissory" in t):
        return "contracts"

    # Catch-all if nothing matched
    return None


# ---------
# REMEDY TYPE CLASSIFIER (RULE-BASED)
# ---------
def classify_remedy_type(text: str) -> list[str]:
    """
    Heuristic detection of remedy / relief types.
    Returns a list (can be empty).
    """
    remedies: list[str] = []

    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text.lower()

    # Injunctive relief
    if ("preliminary injunction" in t
        or "permanent injunction" in t
        or "temporary restraining order" in t
        or "tro" in t):
        remedies.append("injunctive relief")

    # Declaratory judgment / relief
    if ("declaratory judgment" in t
        or "declaratory relief" in t):
        remedies.append("declaratory judgment")

    # Damages
    if "damages" in t:
        remedies.append("damages")

    # Benefits (e.g. unemployment, disability)
    if "unemployment benefits" in t \
         or "benefits under" in t \
         or "eligibility for benefits" in t:
          remedies.append("benefits")


    # Administrative review / appeal of agency decision
    if "administrative review" in t or "board of review" in t:
        remedies.append("administrative review")

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in remedies:
        if r not in seen:
            uniq.append(r)
            seen.add(r)

    return uniq


# ---------
# MAIN PIPELINE
# ---------
def main():
    print(f"Loading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Prepare new columns
    orgs_col = []
    places_col = []
    area_col = []
    remedy_col = []

    total = len(df)
    print(f"Processing {total} rows for metadata extraction...")

    for i, row in df.iterrows():
        text = row.get(TEXT_COL, "")

        orgs, places = run_spacy(text)
        area = classify_area_of_law(text, orgs)
        remedies = classify_remedy_type(text)

        orgs_col.append(orgs)
        places_col.append(places)
        area_col.append(area)
        remedy_col.append(remedies)

        if i % 50 == 0:
            print(f"  Processed {i}/{total} cases...")

    # Attach new columns
    df["orgs"] = orgs_col           # will be stored as Python list repr in CSV
    df["places"] = places_col
    df["area_of_law"] = area_col
    df["remedy_type"] = remedy_col

    print(f"Saving enriched CSV to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
