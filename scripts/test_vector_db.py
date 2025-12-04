#!/usr/bin/env python3
"""
test_vector_db.py - Test vector database queries directly

Usage: python scripts/test_vector_db.py [query] [--limit N]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.legal_tools import get_points, build_context


def test_query(query: str, limit: int = 5, show_full: bool = False):
    """Test a query against the vector database."""
    print(f"Query: {query}")
    print(f"Limit: {limit}")
    print("=" * 60)

    points = get_points(query, limit=limit)

    print(f"\nFound {len(points)} results:\n")

    for i, p in enumerate(points):
        case_name = p.payload.get("case_name", "Unknown")
        case_id = p.payload.get("case_id", "N/A")
        chunk_type = p.payload.get("chunk_type", "full")
        chunk_index = p.payload.get("chunk_index", 0)
        score = p.score
        text = p.payload.get("text", "") or p.payload.get("text_clean", "")

        print(f"{i+1}. {case_name}")
        print(f"   ID: {case_id}")
        print(f"   Chunk: #{chunk_index} ({chunk_type})")
        print(f"   Score: {score:.4f}")
        print(f"   Text length: {len(text)} chars")

        if show_full:
            print(f"\n   --- FULL TEXT ---")
            print(text)
            print(f"   --- END ---\n")
        else:
            preview = text[:200].replace("\n", " ") if text else "No text"
            print(f"   Preview: {preview}...")
        print()

    if not show_full:
        print("=" * 60)
        print("\nBuilt context preview (max 6000 chars):")
        print("-" * 40)
        context = build_context(points, max_total=6000)
        print(context[:2000] + "..." if len(context) > 2000 else context)


def main():
    parser = argparse.ArgumentParser(description="Test vector database queries")
    parser.add_argument("query", nargs="?", default="contract breach damages",
                        help="Query string to search (default: 'contract breach damages')")
    parser.add_argument("--limit", "-l", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--full", "-f", action="store_true",
                        help="Show full text instead of preview")

    args = parser.parse_args()

    test_query(args.query, args.limit, args.full)


if __name__ == "__main__":
    main()
