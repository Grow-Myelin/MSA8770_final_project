#!/usr/bin/env python3
"""
CourtListener Corpus Sampler

This script samples a corpus of 1000 documents from the CourtListener.com API.
Supports multiple sampling strategies for diversity.

Usage:
    python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy sequential
    python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy stratified
    python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy diverse
"""

import argparse
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass, asdict


@dataclass
class Opinion:
    """Represents a court opinion document."""
    id: int
    cluster_id: int
    court: str
    date_filed: str
    case_name: str
    text: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CourtListenerSampler:
    """Samples documents from CourtListener API."""

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"

    def __init__(self, api_token: str, target_count: int = 1000):
        """
        Initialize the sampler.

        Args:
            api_token: CourtListener API token
            target_count: Number of documents to sample (default: 1000)
        """
        self.api_token = api_token
        self.target_count = target_count
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.request_count = 0
        self.rate_limit = 5000  # 5000 requests per hour

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an API request with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response
        """
        # Simple rate limiting (5000 requests/hour = ~1.4 per second)
        time.sleep(0.8)

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.request_count += 1

            if self.request_count % 100 == 0:
                print(f"Made {self.request_count} API requests...")

            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    def sequential_sample(self) -> List[Opinion]:
        """
        Sample documents sequentially using pagination.
        Simple approach that gets the first N documents.

        Returns:
            List of Opinion objects
        """
        print(f"Starting sequential sampling for {self.target_count} documents...")
        opinions = []

        # Use search endpoint with broad query
        url = f"{self.BASE_URL}/search/"
        params = {
            "type": "o",  # opinions
            "order_by": "score desc",
            "q": "*",  # Match all
        }

        cursor = None
        while len(opinions) < self.target_count:
            if cursor:
                params["cursor"] = cursor

            data = self._make_request(url, params)

            if not data.get("results"):
                print("No more results available")
                break

            for result in data["results"]:
                if len(opinions) >= self.target_count:
                    break

                opinion = self._parse_opinion(result)
                if opinion:
                    opinions.append(opinion)

            cursor = data.get("next")
            if not cursor:
                break

            print(f"Collected {len(opinions)}/{self.target_count} documents...")

        return opinions

    def stratified_sample(self, years_range: int = 20) -> List[Opinion]:
        """
        Sample documents across different time periods for temporal diversity.

        Args:
            years_range: Number of years to sample from (default: 20)

        Returns:
            List of Opinion objects
        """
        print(f"Starting stratified sampling for {self.target_count} documents across {years_range} years...")
        opinions = []

        # Calculate time periods
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_range * 365)

        # Divide into periods (e.g., yearly)
        docs_per_period = self.target_count // years_range
        extra_docs = self.target_count % years_range

        for year_offset in range(years_range):
            period_start = start_date + timedelta(days=year_offset * 365)
            period_end = period_start + timedelta(days=365)

            target_for_period = docs_per_period + (1 if year_offset < extra_docs else 0)

            print(f"Sampling {target_for_period} docs from {period_start.date()} to {period_end.date()}...")

            period_opinions = self._sample_by_date_range(
                period_start.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
                target_for_period
            )

            opinions.extend(period_opinions)
            print(f"Collected {len(opinions)}/{self.target_count} documents...")

        return opinions

    def diverse_sample(self) -> List[Opinion]:
        """
        Sample documents from multiple courts for jurisdictional diversity.

        Returns:
            List of Opinion objects
        """
        print(f"Starting diverse sampling for {self.target_count} documents...")
        opinions = []

        # Sample from different court types
        court_types = [
            ("scotus", 100),  # Supreme Court
            ("ca1,ca2,ca3,ca4,ca5,ca6,ca7,ca8,ca9,ca10,ca11,cadc,cafc", 600),  # Circuit Courts
            ("",  300),  # All other courts
        ]

        for court_filter, target in court_types:
            print(f"Sampling {target} documents from courts: {court_filter or 'all'}...")

            params = {
                "type": "o",
                "order_by": "score desc",
                "q": "*",
            }

            if court_filter:
                params["court"] = court_filter

            court_opinions = self._sample_with_params(params, target)
            opinions.extend(court_opinions)

            print(f"Collected {len(opinions)}/{self.target_count} documents...")

            if len(opinions) >= self.target_count:
                break

        return opinions[:self.target_count]

    def _sample_by_date_range(self, start_date: str, end_date: str, target: int) -> List[Opinion]:
        """Sample documents within a specific date range."""
        params = {
            "type": "o",
            "filed_after": start_date,
            "filed_before": end_date,
            "order_by": "score desc",
            "q": "*",
        }

        return self._sample_with_params(params, target)

    def _sample_with_params(self, params: Dict, target: int) -> List[Opinion]:
        """Sample documents with specific query parameters."""
        opinions = []
        url = f"{self.BASE_URL}/search/"
        cursor = None

        while len(opinions) < target:
            if cursor:
                params["cursor"] = cursor

            try:
                data = self._make_request(url, params)
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

            if not data.get("results"):
                break

            for result in data["results"]:
                if len(opinions) >= target:
                    break

                opinion = self._parse_opinion(result)
                if opinion:
                    opinions.append(opinion)

            cursor = data.get("next")
            if not cursor or "cursor=" not in cursor:
                break

        return opinions

    def _parse_opinion(self, result: Dict[str, Any]) -> Optional[Opinion]:
        """
        Parse an opinion from API result.

        Args:
            result: API result dictionary

        Returns:
            Opinion object or None if parsing fails
        """
        try:
            # Extract the opinion text (may be in different fields)
            text = result.get("text", "") or result.get("snippet", "")

            return Opinion(
                id=result.get("id", 0),
                cluster_id=result.get("cluster_id", 0),
                court=result.get("court", ""),
                date_filed=result.get("dateFiled", "") or result.get("date_filed", ""),
                case_name=result.get("caseName", "") or result.get("case_name", ""),
                text=text,
                url=result.get("absolute_url", "")
            )
        except Exception as e:
            print(f"Error parsing opinion: {e}")
            return None

    def save_corpus(self, opinions: List[Opinion], output_dir: str = "corpus"):
        """
        Save the corpus to disk.

        Args:
            opinions: List of Opinion objects
            output_dir: Directory to save the corpus
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save as JSONL (one JSON object per line)
        jsonl_file = output_path / "opinions.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for opinion in opinions:
                f.write(json.dumps(opinion.to_dict(), ensure_ascii=False) + "\n")

        # Save as single JSON array
        json_file = output_path / "opinions.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([op.to_dict() for op in opinions], f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "total_documents": len(opinions),
            "date_collected": datetime.now().isoformat(),
            "courts": list(set(op.court for op in opinions)),
            "date_range": {
                "earliest": min((op.date_filed for op in opinions if op.date_filed), default=None),
                "latest": max((op.date_filed for op in opinions if op.date_filed), default=None)
            }
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nCorpus saved successfully!")
        print(f"  - Documents: {len(opinions)}")
        print(f"  - JSONL: {jsonl_file}")
        print(f"  - JSON: {json_file}")
        print(f"  - Metadata: {metadata_file}")
        print(f"  - Total API requests: {self.request_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample a corpus of documents from CourtListener.com API"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="CourtListener API token"
    )
    parser.add_argument(
        "--strategy",
        choices=["sequential", "stratified", "diverse"],
        default="sequential",
        help="Sampling strategy (default: sequential)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of documents to sample (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="corpus",
        help="Output directory (default: corpus)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Years to sample from for stratified strategy (default: 20)"
    )

    args = parser.parse_args()

    # Initialize sampler
    sampler = CourtListenerSampler(api_token=args.token, target_count=args.count)

    # Sample based on strategy
    print(f"\nUsing {args.strategy} sampling strategy...")

    if args.strategy == "sequential":
        opinions = sampler.sequential_sample()
    elif args.strategy == "stratified":
        opinions = sampler.stratified_sample(years_range=args.years)
    elif args.strategy == "diverse":
        opinions = sampler.diverse_sample()

    # Save corpus
    if opinions:
        sampler.save_corpus(opinions, output_dir=args.output)
        print(f"\n✓ Successfully sampled {len(opinions)} documents!")
    else:
        print("\n✗ No documents were sampled.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
