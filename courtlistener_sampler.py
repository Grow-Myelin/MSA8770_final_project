#!/usr/bin/env python3
"""
Enhanced CourtListener Corpus Sampler for Multi-Agent RAG System (Incremental Save)

This script samples a comprehensive corpus from the CourtListener.com API,
capturing all necessary metadata, full text, citation relationships, and structured data
needed for the multi-agent system (summarization, NER, clustering, and RAG).

Features incremental saving and resume capability.

Usage:
    python enhanced_courtlistener_sampler.py --token YOUR_API_TOKEN --strategy sequential
    python enhanced_courtlistener_sampler.py --token YOUR_API_TOKEN --strategy stratified
    python enhanced_courtlistener_sampler.py --token YOUR_API_TOKEN --strategy diverse
    python enhanced_courtlistener_sampler.py --resume --output existing_corpus_dir
"""

import argparse
import json
import time
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import requests
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlparse, parse_qs
import pickle
import fcntl  # For file locking on Unix systems
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CourtInfo:
    """Court metadata."""
    id: str
    full_name: str
    short_name: str
    jurisdiction: str
    citation_string: str


@dataclass
class JudgeInfo:
    """Judge information."""
    id: Optional[int]
    name: str
    name_full: Optional[str] = None


@dataclass
class Citation:
    """Citation information."""
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    type: Optional[str] = None
    federal_cite_one: Optional[str] = None
    federal_cite_two: Optional[str] = None
    federal_cite_three: Optional[str] = None
    state_cite_one: Optional[str] = None
    state_cite_two: Optional[str] = None
    state_cite_three: Optional[str] = None
    specialty_cite_one: Optional[str] = None
    scotus_early_cite: Optional[str] = None
    lexis_cite: Optional[str] = None
    westlaw_cite: Optional[str] = None
    neutral_cite: Optional[str] = None


@dataclass
class OpinionCluster:
    """Opinion cluster metadata (case-level information)."""
    id: int
    docket_id: int
    judges: List[JudgeInfo]
    date_filed: str
    date_filed_is_approximate: bool
    slug: str
    case_name: str
    case_name_short: str
    case_name_full: str
    scdb_id: str
    scdb_decision_direction: Optional[int]
    scdb_votes_majority: Optional[int]
    scdb_votes_minority: Optional[int]
    source: str
    procedural_history: str
    attorneys: str
    nature_of_suit: str
    posture: str
    syllabus: str
    headnotes: str
    summary: str
    disposition: str
    history: str
    other_dates: str
    cross_reference: str
    correction: str
    citation_count: int
    precedential_status: str
    date_blocked: Optional[str]
    blocked: bool
    citations: List[Citation]
    sub_opinions: List[int]  # List of opinion IDs


@dataclass
class Opinion:
    """Individual opinion within a cluster."""
    id: int
    cluster_id: int
    date_created: str
    date_modified: str
    author: Optional[JudgeInfo]
    joined_by: List[JudgeInfo]
    type: str
    sha1: str
    page_count: Optional[int]
    download_url: Optional[str]
    local_path: Optional[str]
    plain_text: str
    html: str
    html_lawbox: str
    html_columbia: str
    html_anon_2020: str
    xml_harvard: str
    html_with_citations: str  # Primary text field
    extracted_by_ocr: bool
    opinions_cited: List[int]  # IDs of opinions this opinion cites


@dataclass
class Docket:
    """Docket information."""
    id: int
    court: CourtInfo
    appeal_from_str: str
    assigned_to: Optional[JudgeInfo]
    referred_to: Optional[JudgeInfo]
    date_created: str
    date_modified: str
    source: int
    date_cert_granted: Optional[str]
    date_cert_denied: Optional[str]
    date_argued: Optional[str]
    date_reargued: Optional[str]
    date_reargument_denied: Optional[str]
    date_filed: Optional[str]
    date_terminated: Optional[str]
    date_last_filing: Optional[str]
    case_name: str
    case_name_short: str
    case_name_full: str
    slug: str
    docket_number: str
    docket_number_core: str
    pacer_case_id: Optional[str]
    cause: str
    nature_of_suit: str
    jury_demand: str
    jurisdiction_type: str
    appellate_fee_status: str
    appellate_case_type_information: str
    mdl_status: str
    filepath_local: str
    filepath_ia: str
    filepath_ia_json: str
    ia_upload_failure_count: Optional[int]
    ia_needs_upload: Optional[bool]
    ia_date_first_change: Optional[str]
    view_count: int
    date_last_index: Optional[str]
    appeal_from: Optional[int]
    parties: List[str]  # Party information


@dataclass
class EnhancedCourtCase:
    """Complete court case with all related information."""
    docket: Docket
    cluster: OpinionCluster
    opinions: List[Opinion]
    citing_opinions: List[int]  # IDs of opinions that cite this case
    cited_opinions: List[int]   # IDs of opinions this case cites
    citation_depth: Dict[int, int]  # Maps opinion_id to citation depth
    absolute_url: str


@dataclass
class SamplingState:
    """State for resuming sampling operations."""
    strategy: str
    target_count: int
    collected_cluster_ids: Set[int]
    completed_cases: int
    last_cursor: Optional[str]
    current_phase: str  # e.g., "scotus", "circuit", "all"
    phase_progress: Dict[str, Any]
    start_time: str
    last_save_time: str


class IncrementalSaver:
    """Handles incremental saving of cases and state."""
    
    def __init__(self, output_dir: str):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # File paths
        self.cases_file = self.output_path / "cases_incremental.jsonl"
        self.state_file = self.output_path / "sampling_state.json"
        self.progress_file = self.output_path / "progress.json"
        self.lock_file = self.output_path / ".sampling.lock"
        
        # In-memory tracking
        self.saved_count = 0
        self.last_save_time = datetime.now()
        
    def save_case(self, case: EnhancedCourtCase):
        """Save a single case incrementally."""
        case_dict = {
            "docket": asdict(case.docket),
            "cluster": asdict(case.cluster),
            "opinions": [asdict(op) for op in case.opinions],
            "citing_opinions": case.citing_opinions,
            "cited_opinions": case.cited_opinions,
            "citation_depth": case.citation_depth,
            "absolute_url": case.absolute_url,
            "saved_at": datetime.now().isoformat()
        }
        
        # Append to JSONL file with file locking
        with open(self.cases_file, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(case_dict, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        self.saved_count += 1
        self.last_save_time = datetime.now()
        
    def save_state(self, state: SamplingState):
        """Save current sampling state."""
        state_dict = asdict(state)
        state_dict["collected_cluster_ids"] = list(state.collected_cluster_ids)
        state_dict["last_updated"] = datetime.now().isoformat()
        
        # Use temporary file + rename for atomic write
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        
        temp_file.replace(self.state_file)
        
    def update_progress(self, progress_info: Dict[str, Any]):
        """Update progress information."""
        progress_info["last_updated"] = datetime.now().isoformat()
        progress_info["cases_saved"] = self.saved_count
        
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_info, f, indent=2, default=str)
    
    def load_state(self) -> Optional[SamplingState]:
        """Load previous sampling state."""
        if not self.state_file.exists():
            return None
            
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            
            state = SamplingState(
                strategy=state_dict["strategy"],
                target_count=state_dict["target_count"],
                collected_cluster_ids=set(state_dict["collected_cluster_ids"]),
                completed_cases=state_dict["completed_cases"],
                last_cursor=state_dict.get("last_cursor"),
                current_phase=state_dict["current_phase"],
                phase_progress=state_dict["phase_progress"],
                start_time=state_dict["start_time"],
                last_save_time=state_dict["last_save_time"]
            )
            
            # Update saved count from file
            self.saved_count = self.count_saved_cases()
            
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def count_saved_cases(self) -> int:
        """Count how many cases have been saved."""
        if not self.cases_file.exists():
            return 0
            
        count = 0
        try:
            with open(self.cases_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.warning(f"Error counting saved cases: {e}")
            
        return count
    
    def load_saved_cases(self) -> List[EnhancedCourtCase]:
        """Load all previously saved cases."""
        cases = []
        if not self.cases_file.exists():
            return cases
            
        try:
            with open(self.cases_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        case_dict = json.loads(line)
                        # Convert back to EnhancedCourtCase (simplified version for final processing)
                        cases.append(case_dict)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading saved cases: {e}")
            
        return cases
    
    def create_lock(self) -> bool:
        """Create a lock file to prevent concurrent runs."""
        try:
            if self.lock_file.exists():
                # Check if lock is stale (older than 1 hour)
                lock_time = datetime.fromtimestamp(self.lock_file.stat().st_mtime)
                if datetime.now() - lock_time > timedelta(hours=1):
                    logger.warning("Removing stale lock file")
                    self.lock_file.unlink()
                else:
                    return False
            
            with open(self.lock_file, "w") as f:
                f.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            return True
        except Exception as e:
            logger.error(f"Failed to create lock: {e}")
            return False
    
    def remove_lock(self):
        """Remove the lock file."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove lock: {e}")


class EnhancedCourtListenerSampler:
    """Enhanced sampler with incremental saving and resume capability."""

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    SAVE_INTERVAL = 10  # Save every N cases

    def __init__(self, api_token: str, target_count: int = 1000, output_dir: str = "enhanced_corpus"):
        """
        Initialize the enhanced sampler.

        Args:
            api_token: CourtListener API token
            target_count: Number of documents to sample (default: 1000)
            output_dir: Output directory for incremental saves
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
        self.collected_cluster_ids: Set[int] = set()
        
        # Incremental saving
        self.saver = IncrementalSaver(output_dir)
        self.cases_processed_since_save = 0
        
        # Cache for API data to avoid duplicate requests
        self.opinion_cache: Dict[int, Dict] = {}
        self.cluster_cache: Dict[int, Dict] = {}
        self.docket_cache: Dict[int, Dict] = {}
        self.court_cache: Dict[str, Dict] = {}
        self.judge_cache: Dict[int, Dict] = {}

    def _make_request(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Make an API request with rate limiting and retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters
            retries: Number of retry attempts

        Returns:
            JSON response or None if failed
        """
        # Rate limiting (5000 requests/hour = ~1.4 per second)
        time.sleep(0.8)

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                self.request_count += 1

                if self.request_count % 100 == 0:
                    logger.info(f"Made {self.request_count} API requests...")

                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    logger.debug(f"404 Not Found for {url} - resource may not exist")
                    return None
                logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def _get_court_info(self, court_id: str) -> Optional[CourtInfo]:
        """Fetch court information."""
        if court_id in self.court_cache:
            court_data = self.court_cache[court_id]
        else:
            url = f"{self.BASE_URL}/courts/{court_id}/"
            court_data = self._make_request(url)
            if not court_data:
                # Return basic info if API call fails
                return CourtInfo(
                    id=court_id,
                    full_name=court_id,
                    short_name=court_id,
                    jurisdiction="",
                    citation_string=""
                )
            self.court_cache[court_id] = court_data

        return CourtInfo(
            id=court_data.get("id", court_id),
            full_name=court_data.get("full_name", court_id),
            short_name=court_data.get("short_name", court_id),
            jurisdiction=court_data.get("jurisdiction", ""),
            citation_string=court_data.get("citation_string", "")
        )

    def _get_judge_info(self, judge_id: int) -> Optional[JudgeInfo]:
        """Fetch judge information."""
        if judge_id in self.judge_cache:
            judge_data = self.judge_cache[judge_id]
        else:
            url = f"{self.BASE_URL}/people/{judge_id}/"
            judge_data = self._make_request(url)
            if not judge_data:
                return JudgeInfo(id=judge_id, name=f"Judge {judge_id}")
            self.judge_cache[judge_id] = judge_data

        name_parts = []
        if judge_data.get("name_first"):
            name_parts.append(judge_data["name_first"])
        if judge_data.get("name_middle"):
            name_parts.append(judge_data["name_middle"])
        if judge_data.get("name_last"):
            name_parts.append(judge_data["name_last"])
        
        name = " ".join(name_parts) if name_parts else f"Judge {judge_id}"

        return JudgeInfo(
            id=judge_data.get("id"),
            name=name,
            name_full=judge_data.get("name_full")
        )

    def _parse_judges_from_string(self, judge_string: str) -> List[JudgeInfo]:
        """Parse judge names from string format."""
        if not judge_string:
            return []
        
        # Simple parsing - could be enhanced with more sophisticated name parsing
        judge_names = [name.strip() for name in judge_string.split(',')]
        return [JudgeInfo(id=None, name=name) for name in judge_names if name]

    def _get_citations_from_cluster_data(self, cluster_data: Dict) -> List[Citation]:
        """Extract citations from cluster data (they're embedded in the cluster response)."""
        citations = []
        
        # Citations are usually in the cluster data itself
        citation_fields = [
            'federal_cite_one', 'federal_cite_two', 'federal_cite_three',
            'state_cite_one', 'state_cite_two', 'state_cite_three',
            'specialty_cite_one', 'scotus_early_cite', 'lexis_cite',
            'westlaw_cite', 'neutral_cite'
        ]
        
        citation = Citation()
        has_citation = False
        
        for field in citation_fields:
            value = cluster_data.get(field)
            if value:
                setattr(citation, field, value)
                has_citation = True
        
        if has_citation:
            citations.append(citation)
        
        return citations

    def _get_opinion_citations(self, opinion_id: int) -> tuple[List[int], List[int], Dict[int, int]]:
        """
        Get citation relationships for an opinion.
        
        Returns:
            (citing_opinions, cited_opinions, citation_depth_map)
        """
        citing_opinions = []
        cited_opinions = []
        citation_depth = {}

        # Get opinions that cite this one (forward citations)
        url = f"{self.BASE_URL}/opinions-cited/"
        params = {"cited_opinion": opinion_id}
        data = self._make_request(url, params)
        
        if data and data.get("results"):
            for citation in data["results"]:
                citing_id = self._extract_id_from_url(citation.get("citing_opinion", ""))
                if citing_id:
                    citing_opinions.append(citing_id)
                    citation_depth[citing_id] = citation.get("depth", 1)

        # Get opinions this one cites (backward citations)
        params = {"citing_opinion": opinion_id}
        data = self._make_request(url, params)
        
        if data and data.get("results"):
            for citation in data["results"]:
                cited_id = self._extract_id_from_url(citation.get("cited_opinion", ""))
                if cited_id:
                    cited_opinions.append(cited_id)
                    if cited_id not in citation_depth:
                        citation_depth[cited_id] = citation.get("depth", 1)

        return citing_opinions, cited_opinions, citation_depth

    def _extract_id_from_url(self, url: str) -> Optional[int]:
        """Extract ID from API URL."""
        if not url:
            return None
        
        try:
            # URLs are like: https://www.courtlistener.com/api/rest/v4/opinions/123456/
            parts = url.rstrip('/').split('/')
            return int(parts[-1])
        except (ValueError, IndexError):
            return None

    def _get_comprehensive_case_data(self, cluster_id: int) -> Optional[EnhancedCourtCase]:
        """
        Fetch comprehensive data for a case including all related information.
        
        Args:
            cluster_id: The cluster ID from search results
            
        Returns:
            EnhancedCourtCase with all related data
        """
        try:
            # 1. Get cluster data
            if cluster_id in self.cluster_cache:
                cluster_data = self.cluster_cache[cluster_id]
            else:
                cluster_url = f"{self.BASE_URL}/clusters/{cluster_id}/"
                cluster_data = self._make_request(cluster_url)
                if not cluster_data:
                    logger.warning(f"Failed to fetch cluster {cluster_id}")
                    return None
                self.cluster_cache[cluster_id] = cluster_data

            # 2. Get docket data
            docket_url = cluster_data.get("docket")
            if not docket_url:
                logger.warning(f"No docket URL for cluster {cluster_id}")
                return None
                
            docket_id = self._extract_id_from_url(docket_url)
            if not docket_id:
                logger.warning(f"Could not extract docket ID from URL: {docket_url}")
                return None
                
            if docket_id in self.docket_cache:
                docket_data = self.docket_cache[docket_id]
            else:
                docket_data = self._make_request(docket_url)
                if not docket_data:
                    logger.warning(f"Failed to fetch docket for cluster {cluster_id}")
                    return None
                self.docket_cache[docket_id] = docket_data

            # 3. Get court information
            court_id = docket_data.get("court_id", "")
            court_info = self._get_court_info(court_id)
            if not court_info:
                logger.warning(f"Failed to fetch court info for {court_id}")
                return None

            # 4. Get opinions data
            opinion_urls = cluster_data.get("sub_opinions", [])
            opinions = []
            all_citing_opinions = []
            all_cited_opinions = []
            all_citation_depth = {}

            for opinion_url in opinion_urls:
                opinion_id = self._extract_id_from_url(opinion_url)
                if not opinion_id:
                    continue

                if opinion_id in self.opinion_cache:
                    opinion_data = self.opinion_cache[opinion_id]
                else:
                    opinion_data = self._make_request(opinion_url)
                    if not opinion_data:
                        logger.debug(f"Failed to fetch opinion {opinion_id}, skipping")
                        continue
                    self.opinion_cache[opinion_id] = opinion_data

                # Get citation relationships (with error handling)
                try:
                    citing_ops, cited_ops, depth_map = self._get_opinion_citations(opinion_id)
                    all_citing_opinions.extend(citing_ops)
                    all_cited_opinions.extend(cited_ops)
                    all_citation_depth.update(depth_map)
                except Exception as e:
                    logger.debug(f"Failed to get citations for opinion {opinion_id}: {e}")
                    citing_ops, cited_ops = [], []

                # Parse author and joined_by judges
                author = None
                if opinion_data.get("author"):
                    author_id = self._extract_id_from_url(opinion_data["author"])
                    if author_id:
                        author = self._get_judge_info(author_id)

                joined_by = []
                for judge_url in opinion_data.get("joined_by", []):
                    judge_id = self._extract_id_from_url(judge_url)
                    if judge_id:
                        judge_info = self._get_judge_info(judge_id)
                        if judge_info:
                            joined_by.append(judge_info)

                opinion = Opinion(
                    id=opinion_data.get("id", 0),
                    cluster_id=cluster_id,
                    date_created=opinion_data.get("date_created", ""),
                    date_modified=opinion_data.get("date_modified", ""),
                    author=author,
                    joined_by=joined_by,
                    type=opinion_data.get("type", ""),
                    sha1=opinion_data.get("sha1", ""),
                    page_count=opinion_data.get("page_count"),
                    download_url=opinion_data.get("download_url"),
                    local_path=opinion_data.get("local_path"),
                    plain_text=opinion_data.get("plain_text", ""),
                    html=opinion_data.get("html", ""),
                    html_lawbox=opinion_data.get("html_lawbox", ""),
                    html_columbia=opinion_data.get("html_columbia", ""),
                    html_anon_2020=opinion_data.get("html_anon_2020", ""),
                    xml_harvard=opinion_data.get("xml_harvard", ""),
                    html_with_citations=opinion_data.get("html_with_citations", ""),
                    extracted_by_ocr=opinion_data.get("extracted_by_ocr", False),
                    opinions_cited=cited_ops
                )
                opinions.append(opinion)

            # Skip cases with no opinions
            if not opinions:
                logger.warning(f"No opinions found for cluster {cluster_id}, skipping")
                return None

            # 5. Parse judges from cluster
            judges = []
            
            # Handle panel IDs (specific judge references)
            for judge_url in cluster_data.get("panel", []):
                judge_id = self._extract_id_from_url(judge_url)
                if judge_id:
                    judge_info = self._get_judge_info(judge_id)
                    if judge_info:
                        judges.append(judge_info)
            
            # Handle judge strings (when no specific ID available)
            judge_string = cluster_data.get("judges", "")
            if judge_string and not judges:
                judges = self._parse_judges_from_string(judge_string)

            # 6. Get citations from cluster data (embedded)
            citations = self._get_citations_from_cluster_data(cluster_data)

            # 7. Build comprehensive data structures
            assigned_to = None
            if docket_data.get("assigned_to"):
                assigned_to_id = self._extract_id_from_url(docket_data["assigned_to"])
                if assigned_to_id:
                    assigned_to = self._get_judge_info(assigned_to_id)

            referred_to = None
            if docket_data.get("referred_to"):
                referred_to_id = self._extract_id_from_url(docket_data["referred_to"])
                if referred_to_id:
                    referred_to = self._get_judge_info(referred_to_id)

            # Build docket
            docket = Docket(
                id=docket_data.get("id", 0),
                court=court_info,
                appeal_from_str=docket_data.get("appeal_from_str", ""),
                assigned_to=assigned_to,
                referred_to=referred_to,
                date_created=docket_data.get("date_created", ""),
                date_modified=docket_data.get("date_modified", ""),
                source=docket_data.get("source", 0),
                date_cert_granted=docket_data.get("date_cert_granted"),
                date_cert_denied=docket_data.get("date_cert_denied"),
                date_argued=docket_data.get("date_argued"),
                date_reargued=docket_data.get("date_reargued"),
                date_reargument_denied=docket_data.get("date_reargument_denied"),
                date_filed=docket_data.get("date_filed"),
                date_terminated=docket_data.get("date_terminated"),
                date_last_filing=docket_data.get("date_last_filing"),
                case_name=docket_data.get("case_name", ""),
                case_name_short=docket_data.get("case_name_short", ""),
                case_name_full=docket_data.get("case_name_full", ""),
                slug=docket_data.get("slug", ""),
                docket_number=docket_data.get("docket_number", ""),
                docket_number_core=docket_data.get("docket_number_core", ""),
                pacer_case_id=docket_data.get("pacer_case_id"),
                cause=docket_data.get("cause", ""),
                nature_of_suit=docket_data.get("nature_of_suit", ""),
                jury_demand=docket_data.get("jury_demand", ""),
                jurisdiction_type=docket_data.get("jurisdiction_type", ""),
                appellate_fee_status=docket_data.get("appellate_fee_status", ""),
                appellate_case_type_information=docket_data.get("appellate_case_type_information", ""),
                mdl_status=docket_data.get("mdl_status", ""),
                filepath_local=docket_data.get("filepath_local", ""),
                filepath_ia=docket_data.get("filepath_ia", ""),
                filepath_ia_json=docket_data.get("filepath_ia_json", ""),
                ia_upload_failure_count=docket_data.get("ia_upload_failure_count"),
                ia_needs_upload=docket_data.get("ia_needs_upload"),
                ia_date_first_change=docket_data.get("ia_date_first_change"),
                view_count=docket_data.get("view_count", 0),
                date_last_index=docket_data.get("date_last_index"),
                appeal_from=docket_data.get("appeal_from"),
                parties=[]  # Would need separate API call to get party details
            )

            # Build cluster
            cluster = OpinionCluster(
                id=cluster_data.get("id", 0),
                docket_id=docket.id,
                judges=judges,
                date_filed=cluster_data.get("date_filed", ""),
                date_filed_is_approximate=cluster_data.get("date_filed_is_approximate", False),
                slug=cluster_data.get("slug", ""),
                case_name=cluster_data.get("case_name", ""),
                case_name_short=cluster_data.get("case_name_short", ""),
                case_name_full=cluster_data.get("case_name_full", ""),
                scdb_id=cluster_data.get("scdb_id", ""),
                scdb_decision_direction=cluster_data.get("scdb_decision_direction"),
                scdb_votes_majority=cluster_data.get("scdb_votes_majority"),
                scdb_votes_minority=cluster_data.get("scdb_votes_minority"),
                source=cluster_data.get("source", ""),
                procedural_history=cluster_data.get("procedural_history", ""),
                attorneys=cluster_data.get("attorneys", ""),
                nature_of_suit=cluster_data.get("nature_of_suit", ""),
                posture=cluster_data.get("posture", ""),
                syllabus=cluster_data.get("syllabus", ""),
                headnotes=cluster_data.get("headnotes", ""),
                summary=cluster_data.get("summary", ""),
                disposition=cluster_data.get("disposition", ""),
                history=cluster_data.get("history", ""),
                other_dates=cluster_data.get("other_dates", ""),
                cross_reference=cluster_data.get("cross_reference", ""),
                correction=cluster_data.get("correction", ""),
                citation_count=cluster_data.get("citation_count", 0),
                precedential_status=cluster_data.get("precedential_status", ""),
                date_blocked=cluster_data.get("date_blocked"),
                blocked=cluster_data.get("blocked", False),
                citations=citations,
                sub_opinions=[op.id for op in opinions]
            )

            # Build comprehensive case
            enhanced_case = EnhancedCourtCase(
                docket=docket,
                cluster=cluster,
                opinions=opinions,
                citing_opinions=list(set(all_citing_opinions)),
                cited_opinions=list(set(all_cited_opinions)),
                citation_depth=all_citation_depth,
                absolute_url=cluster_data.get("absolute_url", "")
            )

            return enhanced_case

        except Exception as e:
            logger.error(f"Error processing cluster {cluster_id}: {e}")
            return None

    def _save_case_incrementally(self, case: EnhancedCourtCase, state: SamplingState):
        """Save a case incrementally and update state."""
        self.saver.save_case(case)
        self.cases_processed_since_save += 1
        state.completed_cases += 1
        
        # Update progress every few cases
        if self.cases_processed_since_save >= self.SAVE_INTERVAL:
            self.saver.save_state(state)
            self.saver.update_progress({
                "strategy": state.strategy,
                "target_count": state.target_count,
                "completed_cases": state.completed_cases,
                "current_phase": state.current_phase,
                "api_requests": self.request_count,
                "completion_percentage": (state.completed_cases / state.target_count) * 100
            })
            self.cases_processed_since_save = 0
            logger.info(f"Incremental save: {state.completed_cases}/{state.target_count} cases")

    def resume_sampling(self, output_dir: str) -> List[EnhancedCourtCase]:
        """Resume sampling from previous state."""
        self.saver = IncrementalSaver(output_dir)
        
        # Load previous state
        state = self.saver.load_state()
        if not state:
            logger.error("No previous state found to resume from")
            return []
        
        logger.info(f"Resuming {state.strategy} sampling from {state.completed_cases}/{state.target_count} cases")
        self.collected_cluster_ids = state.collected_cluster_ids
        self.target_count = state.target_count
        
        # Continue sampling based on strategy
        if state.strategy == "sequential":
            return self._resume_sequential_sample(state)
        elif state.strategy == "stratified":
            return self._resume_stratified_sample(state)
        elif state.strategy == "diverse":
            return self._resume_diverse_sample(state)
        else:
            logger.error(f"Unknown strategy: {state.strategy}")
            return []

    def sequential_sample(self) -> List[EnhancedCourtCase]:
        """Sample documents sequentially with incremental saving."""
        logger.info(f"Starting enhanced sequential sampling for {self.target_count} documents...")
        
        state = SamplingState(
            strategy="sequential",
            target_count=self.target_count,
            collected_cluster_ids=set(),
            completed_cases=0,
            last_cursor=None,
            current_phase="sequential",
            phase_progress={},
            start_time=datetime.now().isoformat(),
            last_save_time=datetime.now().isoformat()
        )
        
        return self._do_sequential_sample(state)

    def _do_sequential_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Internal sequential sampling with state tracking."""
        url = f"{self.BASE_URL}/search/"
        params = {
            "type": "o",  # opinions
            "order_by": "score desc",
            "q": "*",  # Match all
            "status": "Precedential"  # Focus on precedential cases for better quality
        }

        cursor = state.last_cursor
        processed_count = 0

        while state.completed_cases < self.target_count:
            if cursor:
                params["cursor"] = cursor

            data = self._make_request(url, params)
            if not data or not data.get("results"):
                logger.warning("No more results available")
                break

            for result in data["results"]:
                if state.completed_cases >= self.target_count:
                    break

                cluster_id = result.get("cluster_id")
                if not cluster_id or cluster_id in self.collected_cluster_ids:
                    continue

                # Get comprehensive case data
                enhanced_case = self._get_comprehensive_case_data(cluster_id)
                if enhanced_case:
                    self._save_case_incrementally(enhanced_case, state)
                    self.collected_cluster_ids.add(cluster_id)
                    state.collected_cluster_ids.add(cluster_id)
                    logger.info(f"Collected case {state.completed_cases}/{self.target_count}: {enhanced_case.cluster.case_name}")
                else:
                    logger.debug(f"Skipped cluster {cluster_id} - could not process")

                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count} search results, collected {state.completed_cases} complete cases")

            cursor_url = data.get("next")
            if cursor_url:
                parsed_url = urlparse(cursor_url)
                cursor_params = parse_qs(parsed_url.query)
                cursor = cursor_params.get("cursor", [None])[0]
                state.last_cursor = cursor
            else:
                cursor = None
                state.last_cursor = None

            if not cursor:
                logger.warning("No more pages available")
                break

        # Final save
        self.saver.save_state(state)
        logger.info(f"Enhanced sequential sampling completed: {state.completed_cases} cases collected")
        return self.saver.load_saved_cases()

    def _resume_sequential_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Resume sequential sampling from previous state."""
        return self._do_sequential_sample(state)

    def diverse_sample(self) -> List[EnhancedCourtCase]:
        """Sample documents from multiple courts with incremental saving."""
        logger.info(f"Starting enhanced diverse sampling for {self.target_count} documents...")
        
        state = SamplingState(
            strategy="diverse",
            target_count=self.target_count,
            collected_cluster_ids=set(),
            completed_cases=0,
            last_cursor=None,
            current_phase="scotus",
            phase_progress={
                "scotus_target": min(50, self.target_count // 10),
                "circuit_target": min(250, self.target_count * 5 // 10),
                "all_target": self.target_count,
                "scotus_completed": 0,
                "circuit_completed": 0,
                "all_completed": 0
            },
            start_time=datetime.now().isoformat(),
            last_save_time=datetime.now().isoformat()
        )
        
        return self._do_diverse_sample(state)

    def _do_diverse_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Internal diverse sampling with state tracking."""
        court_phases = [
            ("scotus", "scotus", state.phase_progress["scotus_target"]),
            ("circuit", "ca1,ca2,ca3,ca4,ca5,ca6,ca7,ca8,ca9,ca10,ca11,cadc,cafc", state.phase_progress["circuit_target"]),
            ("all", "", state.phase_progress["all_target"]),
        ]

        # Find current phase
        start_phase = 0
        if state.current_phase == "circuit":
            start_phase = 1
        elif state.current_phase == "all":
            start_phase = 2

        for phase_idx in range(start_phase, len(court_phases)):
            phase_name, court_filter, total_target = court_phases[phase_idx]
            
            if state.completed_cases >= self.target_count:
                break

            phase_completed_key = f"{phase_name}_completed"
            already_completed = state.phase_progress.get(phase_completed_key, 0)
            remaining_for_phase = min(total_target - already_completed, self.target_count - state.completed_cases)
            
            if remaining_for_phase <= 0:
                continue

            state.current_phase = phase_name
            logger.info(f"Sampling {remaining_for_phase} cases from phase '{phase_name}' (courts: {court_filter or 'all'})")

            params = {
                "type": "o",
                "order_by": "score desc",
                "q": "*",
                "status": "Precedential"
            }

            if court_filter:
                params["court"] = court_filter

            phase_cases = self._sample_with_params_incremental(params, remaining_for_phase, state)
            state.phase_progress[phase_completed_key] += len(phase_cases)
            
            logger.info(f"Completed phase '{phase_name}': {len(phase_cases)} cases")

        # Final save
        self.saver.save_state(state)
        logger.info(f"Enhanced diverse sampling completed: {state.completed_cases} cases collected")
        return self.saver.load_saved_cases()

    def _resume_diverse_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Resume diverse sampling from previous state."""
        return self._do_diverse_sample(state)

    def _sample_with_params_incremental(self, params: Dict, target: int, state: SamplingState) -> List[Dict]:
        """Sample with specific parameters and incremental saving."""
        cases = []
        url = f"{self.BASE_URL}/search/"
        cursor = state.last_cursor if state.last_cursor else None
        processed_count = 0

        while len(cases) < target and state.completed_cases < self.target_count:
            if cursor:
                params["cursor"] = cursor

            try:
                data = self._make_request(url, params)
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

            if not data or not data.get("results"):
                break

            for result in data["results"]:
                if len(cases) >= target or state.completed_cases >= self.target_count:
                    break

                cluster_id = result.get("cluster_id")
                if not cluster_id or cluster_id in self.collected_cluster_ids:
                    continue

                enhanced_case = self._get_comprehensive_case_data(cluster_id)
                if enhanced_case:
                    self._save_case_incrementally(enhanced_case, state)
                    cases.append(asdict(enhanced_case))
                    self.collected_cluster_ids.add(cluster_id)
                    state.collected_cluster_ids.add(cluster_id)

                processed_count += 1

            cursor_url = data.get("next")
            if cursor_url:
                parsed_url = urlparse(cursor_url)
                cursor_params = parse_qs(parsed_url.query)
                cursor = cursor_params.get("cursor", [None])[0]
                state.last_cursor = cursor
            else:
                cursor = None
                state.last_cursor = None

            if not cursor:
                break

        return cases

    def stratified_sample(self, years_range: int = 20) -> List[EnhancedCourtCase]:
        """Sample documents across different time periods with incremental saving."""
        logger.info(f"Starting enhanced stratified sampling for {self.target_count} documents across {years_range} years...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_range * 365)
        docs_per_period = self.target_count // years_range
        extra_docs = self.target_count % years_range

        state = SamplingState(
            strategy="stratified",
            target_count=self.target_count,
            collected_cluster_ids=set(),
            completed_cases=0,
            last_cursor=None,
            current_phase="period_0",
            phase_progress={
                "years_range": years_range,
                "docs_per_period": docs_per_period,
                "extra_docs": extra_docs,
                "current_year_offset": 0,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            start_time=datetime.now().isoformat(),
            last_save_time=datetime.now().isoformat()
        )
        
        return self._do_stratified_sample(state)

    def _do_stratified_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Internal stratified sampling with state tracking."""
        years_range = state.phase_progress["years_range"]
        docs_per_period = state.phase_progress["docs_per_period"]
        extra_docs = state.phase_progress["extra_docs"]
        start_date = datetime.fromisoformat(state.phase_progress["start_date"])
        current_year_offset = state.phase_progress.get("current_year_offset", 0)

        for year_offset in range(current_year_offset, years_range):
            if state.completed_cases >= self.target_count:
                break

            period_start = start_date + timedelta(days=year_offset * 365)
            period_end = period_start + timedelta(days=365)
            target_for_period = docs_per_period + (1 if year_offset < extra_docs else 0)
            remaining_target = min(target_for_period, self.target_count - state.completed_cases)

            state.current_phase = f"period_{year_offset}"
            state.phase_progress["current_year_offset"] = year_offset
            
            logger.info(f"Sampling {remaining_target} cases from {period_start.date()} to {period_end.date()}...")

            period_cases = self._sample_by_date_range_incremental(
                period_start.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
                remaining_target,
                state
            )

            logger.info(f"Collected {len(period_cases)} cases for period {year_offset + 1}/{years_range}")

        # Final save
        self.saver.save_state(state)
        logger.info(f"Enhanced stratified sampling completed: {state.completed_cases} cases collected")
        return self.saver.load_saved_cases()

    def _resume_stratified_sample(self, state: SamplingState) -> List[EnhancedCourtCase]:
        """Resume stratified sampling from previous state."""
        return self._do_stratified_sample(state)

    def _sample_by_date_range_incremental(self, start_date: str, end_date: str, target: int, state: SamplingState) -> List[Dict]:
        """Sample within date range with incremental saving."""
        params = {
            "type": "o",
            "filed_after": start_date,
            "filed_before": end_date,
            "order_by": "score desc",
            "q": "*",
            "status": "Precedential"
        }

        return self._sample_with_params_incremental(params, target, state)

    def finalize_corpus(self, output_dir: str):
        """Create final corpus files from incremental saves."""
        logger.info("Finalizing corpus from incremental saves...")
        
        # Load all saved cases
        saved_cases = self.saver.load_saved_cases()
        if not saved_cases:
            logger.warning("No saved cases found")
            return

        output_path = Path(output_dir)
        
        # Save final comprehensive file
        comprehensive_file = output_path / "comprehensive_cases.json"
        with open(comprehensive_file, "w", encoding="utf-8") as f:
            json.dump(saved_cases, f, indent=2, ensure_ascii=False, default=str)

        # Create agent extracts from saved data
        self._create_final_agent_extracts(saved_cases, output_path)

        # Create final metadata
        metadata = {
            "total_cases": len(saved_cases),
            "date_collected": datetime.now().isoformat(),
            "api_requests_made": self.request_count,
            "finalized_from_incremental": True
        }

        metadata_file = output_path / "final_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Finalized corpus: {len(saved_cases)} cases")
        logger.info(f"Comprehensive JSON: {comprehensive_file}")
        logger.info(f"Metadata: {metadata_file}")

    def _create_final_agent_extracts(self, cases_data: List[Dict], output_path: Path):
        """Create agent extracts from saved case data."""
        # These would be similar to the previous agent extract methods
        # but working with the dictionary format from saved data
        
        summarization_data = []
        ner_data = []
        clustering_data = []
        rag_data = []
        
        for case_dict in cases_data:
            cluster = case_dict.get("cluster", {})
            docket = case_dict.get("docket", {})
            opinions = case_dict.get("opinions", [])
            
            case_name = cluster.get("case_name", "")
            cluster_id = cluster.get("id", 0)
            
            # Extract for each agent type
            for opinion in opinions:
                opinion_id = opinion.get("id", 0)
                text_content = (
                    opinion.get("html_with_citations") or 
                    opinion.get("html") or 
                    opinion.get("plain_text") or
                    ""
                )
                
                if text_content.strip():
                    # Summarization extract
                    summarization_data.append({
                        "id": f"case_{cluster_id}_opinion_{opinion_id}",
                        "case_name": case_name,
                        "court": docket.get("court", {}).get("full_name", ""),
                        "date_filed": cluster.get("date_filed", ""),
                        "opinion_type": opinion.get("type", ""),
                        "text": text_content,
                        "metadata": {
                            "cluster_id": cluster_id,
                            "opinion_id": opinion_id,
                            "precedential_status": cluster.get("precedential_status", "")
                        }
                    })
                    
                    # Add to other extracts as needed...

        # Save extracts
        if summarization_data:
            with open(output_path / "summarization_extract.json", "w", encoding="utf-8") as f:
                json.dump(summarization_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created final agent extracts: {len(summarization_data)} items")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CourtListener sampler with incremental saving and resume capability"
    )
    parser.add_argument(
        "--token",
        help="CourtListener API token (or set COURTLISTENER_TOKEN env var)"
    )
    parser.add_argument(
        "--config",
        help="Path to config file (JSON with 'api_token' field)"
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
        default="enhanced_corpus",
        help="Output directory (default: enhanced_corpus)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Years to sample from for stratified strategy (default: 20)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous sampling session"
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only create final corpus from incremental saves"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Handle finalize-only mode
    if args.finalize_only:
        sampler = EnhancedCourtListenerSampler("dummy", output_dir=args.output)
        sampler.finalize_corpus(args.output)
        return 0

    # Get API token
    api_token = args.token

    if not api_token and args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                api_token = config.get('api_token')
                logger.info(f"Loaded API token from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    if not api_token:
        api_token = os.environ.get('COURTLISTENER_TOKEN')
        if api_token:
            logger.info("Using API token from COURTLISTENER_TOKEN environment variable")

    if not api_token:
        logger.error("Error: No API token provided!")
        logger.error("Please provide token via:")
        logger.error("  1. --token argument")
        logger.error("  2. --config config.json (with 'api_token' field)")
        logger.error("  3. COURTLISTENER_TOKEN environment variable")
        return 1

    # Initialize enhanced sampler
    sampler = EnhancedCourtListenerSampler(
        api_token=api_token, 
        target_count=args.count,
        output_dir=args.output
    )

    # Check for lock file
    if not sampler.saver.create_lock():
        logger.error("Another sampling process is already running or a stale lock exists")
        logger.error("Use --force to override or remove the lock file manually")
        return 1

    try:
        # Handle resume mode
        if args.resume:
            logger.info("Resuming sampling from previous session...")
            cases = sampler.resume_sampling(args.output)
        else:
            # Sample based on strategy
            logger.info(f"Using {args.strategy} sampling strategy for enhanced data collection...")
            
            if args.strategy == "sequential":
                cases = sampler.sequential_sample()
            elif args.strategy == "stratified":
                cases = sampler.stratified_sample(years_range=args.years)
            elif args.strategy == "diverse":
                cases = sampler.diverse_sample()

        # Finalize corpus
        sampler.finalize_corpus(args.output)
        
        if isinstance(cases, list) and cases:
            logger.info(f"\n✓ Successfully sampled {len(cases)} comprehensive cases!")
            logger.info(f"  - API requests made: {sampler.request_count}")
        else:
            logger.warning("\n⚠ Sampling completed but no final case list available (check incremental saves)")
            
    except KeyboardInterrupt:
        logger.warning("\nSampling interrupted by user")
        logger.info("Progress has been saved incrementally. Use --resume to continue.")
        return 1
    except Exception as e:
        logger.error(f"\nError during sampling: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    finally:
        # Always remove lock
        sampler.saver.remove_lock()

    return 0


if __name__ == "__main__":
    exit(main())