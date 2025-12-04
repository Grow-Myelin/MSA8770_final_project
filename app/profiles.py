"""
profiles.py - User Profile Management for Long-Term Memory

This module provides persistent user profile storage for tracking:
- Research preferences (areas of law, jurisdictions)
- Query history (distilled summaries)
- Auto-detected interests based on usage patterns
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter

from app.logger import get_logger

logger = get_logger("app.profiles")

# Profile storage directory
PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
MAX_QUERY_HISTORY = 20
MAX_AREAS_OF_LAW = 5
MAX_JURISDICTIONS = 5


def _get_profile_path(profile_id: str) -> Path:
    """Get the file path for a profile."""
    return PROFILES_DIR / f"{profile_id}.json"


def _generate_profile_id() -> str:
    """Generate a unique profile ID."""
    return str(uuid.uuid4())[:8]


def create_profile(name: str) -> dict:
    """
    Create a new user profile.

    Args:
        name: Display name for the profile

    Returns:
        The created profile dictionary
    """
    profile_id = _generate_profile_id()
    now = datetime.now().isoformat()

    profile = {
        "profile_id": profile_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "preferences": {
            "areas_of_law": [],
            "jurisdictions": [],
        },
        "topics_summary": "",
        "query_history": []
    }

    # Save to file
    profile_path = _get_profile_path(profile_id)
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Created profile: {profile_id} ({name})")
    return profile


def get_profile(profile_id: str) -> Optional[dict]:
    """
    Get a profile by ID.

    Args:
        profile_id: The profile ID to retrieve

    Returns:
        Profile dictionary or None if not found
    """
    profile_path = _get_profile_path(profile_id)

    if not profile_path.exists():
        return None

    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading profile {profile_id}: {e}")
        return None


def update_profile(profile_id: str, updates: dict) -> Optional[dict]:
    """
    Update a profile with new data.

    Args:
        profile_id: The profile ID to update
        updates: Dictionary of fields to update (name, preferences, topics_summary)

    Returns:
        Updated profile dictionary or None if not found
    """
    profile = get_profile(profile_id)
    if not profile:
        return None

    # Apply allowed updates
    allowed_fields = ["name", "preferences", "topics_summary"]
    for field in allowed_fields:
        if field in updates:
            if field == "preferences" and isinstance(updates[field], dict):
                # Merge preferences
                profile["preferences"].update(updates[field])
            else:
                profile[field] = updates[field]

    profile["updated_at"] = datetime.now().isoformat()

    # Save
    profile_path = _get_profile_path(profile_id)
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Updated profile: {profile_id}")
    return profile


def delete_profile(profile_id: str) -> bool:
    """
    Delete a profile.

    Args:
        profile_id: The profile ID to delete

    Returns:
        True if deleted, False if not found
    """
    profile_path = _get_profile_path(profile_id)

    if not profile_path.exists():
        return False

    try:
        profile_path.unlink()
        logger.info(f"Deleted profile: {profile_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting profile {profile_id}: {e}")
        return False


def list_profiles() -> List[dict]:
    """
    List all profiles.

    Returns:
        List of profile summary dictionaries
    """
    profiles = []

    for profile_file in PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_file, 'r') as f:
                profile = json.load(f)
                profiles.append({
                    "profile_id": profile.get("profile_id"),
                    "name": profile.get("name"),
                    "created_at": profile.get("created_at"),
                    "updated_at": profile.get("updated_at"),
                    "query_count": len(profile.get("query_history", [])),
                    "areas_of_law": profile.get("preferences", {}).get("areas_of_law", [])
                })
        except Exception as e:
            logger.error(f"Error reading profile {profile_file}: {e}")
            continue

    # Sort by most recently updated
    profiles.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return profiles


def add_query_to_history(
    profile_id: str,
    question: str,
    area_of_law: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    answer_preview: Optional[str] = None,
    session_log_file: Optional[str] = None
) -> Optional[dict]:
    """
    Add a query to the profile's history and auto-update preferences.

    Args:
        profile_id: The profile ID
        question: The question that was asked
        area_of_law: Detected area of law (optional)
        jurisdiction: Detected jurisdiction (optional)
        answer_preview: First 200 chars of answer (optional)
        session_log_file: Filename of the session log for replay (optional)

    Returns:
        Updated profile or None if not found
    """
    profile = get_profile(profile_id)
    if not profile:
        return None

    # Create query entry
    query_entry = {
        "question": question[:500],  # Truncate long questions
        "timestamp": datetime.now().isoformat(),
        "area_of_law": area_of_law,
        "jurisdiction": jurisdiction,
        "answer_preview": answer_preview[:200] if answer_preview else None,
        "session_log_file": session_log_file
    }

    # Add to history
    profile["query_history"].append(query_entry)

    # Keep only last N queries
    if len(profile["query_history"]) > MAX_QUERY_HISTORY:
        profile["query_history"] = profile["query_history"][-MAX_QUERY_HISTORY:]

    # Auto-update preferences based on query patterns
    profile = _update_preferences_from_history(profile)

    profile["updated_at"] = datetime.now().isoformat()

    # Save
    profile_path = _get_profile_path(profile_id)
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Added query to profile {profile_id} history")
    return profile


def _update_preferences_from_history(profile: dict) -> dict:
    """
    Auto-detect and update preferences based on query history.

    Analyzes the query history to identify:
    - Most frequently researched areas of law
    - Most frequently mentioned jurisdictions
    """
    history = profile.get("query_history", [])

    if not history:
        return profile

    # Count areas of law
    area_counts = Counter()
    jurisdiction_counts = Counter()

    for query in history:
        if query.get("area_of_law"):
            area_counts[query["area_of_law"]] += 1
        if query.get("jurisdiction"):
            jurisdiction_counts[query["jurisdiction"]] += 1

    # Update preferences with most common values
    if area_counts:
        profile["preferences"]["areas_of_law"] = [
            area for area, _ in area_counts.most_common(MAX_AREAS_OF_LAW)
        ]

    if jurisdiction_counts:
        profile["preferences"]["jurisdictions"] = [
            jur for jur, _ in jurisdiction_counts.most_common(MAX_JURISDICTIONS)
        ]

    # Generate topics summary
    if len(history) >= 3:
        recent_topics = set()
        for query in history[-10:]:  # Last 10 queries
            if query.get("area_of_law"):
                recent_topics.add(query["area_of_law"])

        if recent_topics:
            profile["topics_summary"] = f"Recent research focus: {', '.join(sorted(recent_topics))}"

    return profile


def get_profile_context(profile_id: str) -> str:
    """
    Get a formatted context string for including in LLM prompts.

    This provides the LLM with user preferences and history context
    to improve response relevance.

    Args:
        profile_id: The profile ID

    Returns:
        Formatted context string
    """
    profile = get_profile(profile_id)
    if not profile:
        return ""

    context_parts = []

    # Add preference context
    prefs = profile.get("preferences", {})
    if prefs.get("areas_of_law"):
        context_parts.append(
            f"User frequently researches: {', '.join(prefs['areas_of_law'])}"
        )

    if prefs.get("jurisdictions"):
        context_parts.append(
            f"User focuses on jurisdictions: {', '.join(prefs['jurisdictions'])}"
        )

    # Add recent query context
    history = profile.get("query_history", [])
    if history:
        recent = history[-3:]  # Last 3 queries
        recent_questions = [q.get("question", "")[:100] for q in recent]
        context_parts.append(
            f"Recent questions: {'; '.join(recent_questions)}"
        )

    # Add topics summary
    if profile.get("topics_summary"):
        context_parts.append(profile["topics_summary"])

    if not context_parts:
        return ""

    return "User Profile Context:\n" + "\n".join(f"- {part}" for part in context_parts)
