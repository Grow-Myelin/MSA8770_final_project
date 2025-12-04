
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
    Heuristic classifier for area of law based on keywords.
    Returns the most likely area of law or None if uncertain.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text.lower()

    orgs_lower = [o.lower() for o in orgs]

    # Criminal law (check early - common category)
    if any(kw in t for kw in [
        "defendant was convicted", "criminal conviction", "sentenced to",
        "indictment", "felony", "misdemeanor", "criminal defendant",
        "plea agreement", "plea bargain", "guilty verdict", "not guilty",
        "prison", "incarceration", "probation", "parole",
        "murder", "manslaughter", "robbery", "burglary", "assault",
        "drug offense", "possession of", "trafficking"
    ]):
        return "criminal"

    # Constitutional law
    if any(kw in t for kw in [
        "first amendment", "second amendment", "fourth amendment",
        "fifth amendment", "fourteenth amendment", "due process",
        "equal protection", "constitutional right", "civil rights",
        "free speech", "freedom of religion", "establishment clause",
        "search and seizure", "miranda", "habeas corpus"
    ]):
        return "constitutional"

    # Environmental law
    if any(kw in t for kw in [
        "environmental protection", "clean air act", "clean water act",
        "epa", "environmental impact", "pollution", "hazardous waste",
        "endangered species", "nepa", "environmental review"
    ]) or "environmental protection agency" in t:
        return "environmental"

    # Immigration law
    if any(kw in t for kw in [
        "immigration", "deportation", "asylum", "refugee",
        "naturalization", "visa", "uscis", "removal proceedings",
        "alien", "lawful permanent resident", "green card"
    ]):
        return "immigration"

    # Regulatory / Administrative law
    if any(kw in t for kw in [
        "administrative review", "board of review", "agency decision",
        "administrative agency", "rulemaking", "regulatory",
        "nuclear regulatory", "nrc", "fcc", "sec", "ftc",
        "administrative procedure act", "arbitrary and capricious",
        "chevron deference", "agency interpretation"
    ]):
        return "administrative"

    # Labor / Employment law
    if any(kw in t for kw in [
        "unemployment benefits", "employment security", "wrongful termination",
        "workplace discrimination", "title vii", "ada", "fmla",
        "labor relations", "nlrb", "collective bargaining", "union",
        "wage and hour", "flsa", "osha", "workplace safety",
        "employment discrimination", "hostile work environment"
    ]) or any("employment security" in o for o in orgs_lower):
        return "employment"

    # Property law
    if any(kw in t for kw in [
        "prescriptive easement", "easement by prescription", "quiet title",
        "adverse possession", "eminent domain", "condemnation",
        "zoning", "land use", "property rights", "trespass",
        "landlord", "tenant", "lease", "eviction", "foreclosure",
        "mortgage", "deed", "title insurance", "real property"
    ]):
        return "property"

    # Family law
    if any(kw in t for kw in [
        "dissolution of marriage", "divorce", "custody", "child support",
        "visitation", "alimony", "spousal support", "adoption",
        "paternity", "parental rights", "child welfare", "dcfs",
        "guardianship", "domestic relations"
    ]):
        return "family"

    # Torts
    if any(kw in t for kw in [
        "negligence", "duty of care", "personal injury", "tort",
        "malpractice", "medical malpractice", "legal malpractice",
        "products liability", "defamation", "libel", "slander",
        "intentional infliction", "wrongful death", "premises liability",
        "negligent", "proximate cause", "damages"
    ]):
        return "torts"

    # Contracts
    if any(kw in t for kw in [
        "breach of contract", "contract dispute", "lease agreement",
        "promissory", "contractual obligation", "specific performance",
        "breach of warranty", "unjust enrichment", "quasi-contract",
        "contract interpretation", "parol evidence"
    ]):
        return "contracts"

    # Intellectual Property
    if any(kw in t for kw in [
        "patent", "trademark", "copyright", "trade secret",
        "infringement", "intellectual property", "licensing agreement"
    ]):
        return "intellectual_property"

    # Tax law
    if any(kw in t for kw in [
        "tax court", "irs", "income tax", "tax liability",
        "tax refund", "tax assessment", "internal revenue"
    ]):
        return "tax"

    # Bankruptcy
    if any(kw in t for kw in [
        "bankruptcy", "chapter 7", "chapter 11", "chapter 13",
        "debtor", "creditor", "discharge", "reorganization"
    ]):
        return "bankruptcy"

    # Civil procedure (catch-all for procedural matters)
    if any(kw in t for kw in [
        "summary judgment", "motion to dismiss", "class action",
        "jurisdiction", "standing", "statute of limitations",
        "res judicata", "collateral estoppel"
    ]):
        return "civil_procedure"

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
