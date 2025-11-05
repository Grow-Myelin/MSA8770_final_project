#!/usr/bin/env python3
"""
Example usage of the CourtListener sampler as a library.

This shows how to use the CourtListenerSampler class programmatically
rather than via the command-line interface.
"""

from courtlistener_sampler import CourtListenerSampler, Opinion
import os


def example_basic_usage():
    """Basic sequential sampling example."""
    # Get API token from environment variable
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if not api_token:
        print("Please set COURTLISTENER_TOKEN environment variable")
        return

    # Create sampler
    sampler = CourtListenerSampler(api_token=api_token, target_count=100)

    # Sample documents
    print("Sampling 100 documents sequentially...")
    opinions = sampler.sequential_sample()

    # Save corpus
    sampler.save_corpus(opinions, output_dir="example_corpus")

    print(f"Sampled {len(opinions)} documents")


def example_custom_processing():
    """Example with custom processing of sampled documents."""
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if not api_token:
        print("Please set COURTLISTENER_TOKEN environment variable")
        return

    sampler = CourtListenerSampler(api_token=api_token, target_count=50)

    # Sample documents
    opinions = sampler.diverse_sample()

    # Custom processing
    print("\nDocument Statistics:")
    print(f"Total documents: {len(opinions)}")

    # Count by court
    court_counts = {}
    for opinion in opinions:
        court_counts[opinion.court] = court_counts.get(opinion.court, 0) + 1

    print("\nDocuments by court:")
    for court, count in sorted(court_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count}")

    # Average text length
    text_lengths = [len(opinion.text) for opinion in opinions if opinion.text]
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        print(f"\nAverage text length: {avg_length:.0f} characters")

    # Date range
    dates = [opinion.date_filed for opinion in opinions if opinion.date_filed]
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")


def example_filtered_sampling():
    """Example of sampling with custom filters."""
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if not api_token:
        print("Please set COURTLISTENER_TOKEN environment variable")
        return

    sampler = CourtListenerSampler(api_token=api_token, target_count=50)

    # Sample only Supreme Court cases
    print("Sampling only Supreme Court cases...")
    params = {
        "type": "o",
        "court": "scotus",
        "order_by": "score desc",
        "q": "*",
    }

    opinions = sampler._sample_with_params(params, target=50)

    print(f"Sampled {len(opinions)} Supreme Court opinions")

    # Display some case names
    print("\nSample case names:")
    for opinion in opinions[:5]:
        print(f"  - {opinion.case_name} ({opinion.date_filed})")


def example_date_range_sampling():
    """Example of sampling from a specific date range."""
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if not api_token:
        print("Please set COURTLISTENER_TOKEN environment variable")
        return

    sampler = CourtListenerSampler(api_token=api_token, target_count=100)

    # Sample from recent years (2020-2025)
    print("Sampling from 2020-2025...")
    opinions = sampler._sample_by_date_range(
        start_date="2020-01-01",
        end_date="2025-12-31",
        target=100
    )

    print(f"Sampled {len(opinions)} opinions from 2020-2025")

    # Show year distribution
    year_counts = {}
    for opinion in opinions:
        if opinion.date_filed:
            year = opinion.date_filed[:4]
            year_counts[year] = year_counts.get(year, 0) + 1

    print("\nDistribution by year:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")


def example_progressive_sampling():
    """Example showing progress during sampling."""
    api_token = os.environ.get("COURTLISTENER_TOKEN")
    if not api_token:
        print("Please set COURTLISTENER_TOKEN environment variable")
        return

    print("Starting progressive sampling...")
    sampler = CourtListenerSampler(api_token=api_token, target_count=200)

    # Sample with progress updates
    opinions = []
    targets = [50, 100, 150, 200]

    for i, target in enumerate(targets):
        print(f"\nPhase {i+1}: Sampling to reach {target} documents...")

        if i == 0:
            batch = sampler._sample_with_params(
                {"type": "o", "q": "*", "order_by": "score desc"},
                target=50
            )
        else:
            # Continue sampling
            remaining = target - len(opinions)
            batch = sampler._sample_with_params(
                {"type": "o", "q": "*", "order_by": "score desc"},
                target=remaining
            )

        opinions.extend(batch)
        print(f"Total collected: {len(opinions)}")

    sampler.save_corpus(opinions, output_dir="progressive_corpus")


if __name__ == "__main__":
    print("CourtListener Sampler - Example Usage\n")

    # Check for API token
    if not os.environ.get("COURTLISTENER_TOKEN"):
        print("ERROR: Please set the COURTLISTENER_TOKEN environment variable")
        print("\nExamples:")
        print("  export COURTLISTENER_TOKEN='your-token-here'  # Linux/Mac")
        print("  set COURTLISTENER_TOKEN=your-token-here       # Windows")
        exit(1)

    # Run examples
    print("=" * 60)
    print("Example 1: Basic Sequential Sampling")
    print("=" * 60)
    example_basic_usage()

    print("\n" + "=" * 60)
    print("Example 2: Custom Processing")
    print("=" * 60)
    example_custom_processing()

    print("\n" + "=" * 60)
    print("Example 3: Filtered Sampling (Supreme Court)")
    print("=" * 60)
    example_filtered_sampling()

    print("\n" + "=" * 60)
    print("Example 4: Date Range Sampling")
    print("=" * 60)
    example_date_range_sampling()

    print("\n" + "=" * 60)
    print("Example 5: Progressive Sampling")
    print("=" * 60)
    example_progressive_sampling()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
