#!/usr/bin/env python3
"""
convert_to_csv.py - Convert sampler JSON output to CSV format for pipeline.

This script converts the JSON output from courtlistener_sampler.py
to the CSV format expected by the agents pipeline.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    """Clean HTML and normalize whitespace in opinion text."""
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def convert_json_to_csv(input_path: Path, output_path: Path):
    """
    Convert sampler JSON output to CSV format.

    Expected input: Either comprehensive_cases.json or summarization_extract.json
    Output: summarization_extract_clean.csv with columns:
        - id: unique case-opinion identifier
        - case_name: name of the case
        - text_clean: cleaned opinion text
    """
    print(f"Loading JSON from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # Handle different JSON formats
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]

        # Format 1: summarization_extract.json format
        if "case_id" in first_item and "text" in first_item:
            print("Detected summarization_extract.json format")
            for item in data:
                records.append({
                    "id": item.get("case_id", ""),
                    "case_name": item.get("case_name", ""),
                    "text_clean": clean_text(item.get("text", "")),
                })

        # Format 2: comprehensive_cases.json format
        elif "docket" in first_item and "cluster" in first_item:
            print("Detected comprehensive_cases.json format")
            for case in data:
                cluster = case.get("cluster", {})
                docket = case.get("docket", {})
                opinions = case.get("opinions", [])

                case_name = docket.get("case_name", "") or cluster.get("case_name", "")
                cluster_id = cluster.get("id", 0)

                for opinion in opinions:
                    opinion_id = opinion.get("id", 0)

                    # Get the best available text
                    text = (
                        opinion.get("html_with_citations") or
                        opinion.get("plain_text") or
                        opinion.get("html") or
                        ""
                    )

                    case_id = f"case_{cluster_id}_opinion_{opinion_id}"

                    records.append({
                        "id": case_id,
                        "case_name": case_name,
                        "text_clean": clean_text(text),
                    })

    if not records:
        print("ERROR: No records found in JSON file")
        sys.exit(1)

    # Create DataFrame and save
    df = pd.DataFrame(records)

    # Filter out empty texts
    original_count = len(df)
    df = df[df["text_clean"].str.len() > 100]  # Minimum 100 chars
    filtered_count = len(df)

    print(f"Converted {original_count} records, kept {filtered_count} with sufficient text")

    print(f"Saving CSV to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_csv.py <input.json> <output.csv>")
        print("Example: python convert_to_csv.py corpus/comprehensive_cases.json data/summarization_extract_clean.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_json_to_csv(input_path, output_path)


if __name__ == "__main__":
    main()
