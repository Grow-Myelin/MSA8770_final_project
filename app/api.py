"""
api.py - API routes for the Legal Demo Dashboard

This module provides REST API endpoints that wrap the agent tools,
including Server-Sent Events (SSE) for real-time agent streaming.
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.logger import get_logger, SessionLogger, get_request_id

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

router = APIRouter()
logger = get_logger("app.api")

# ============================================
# Conversation Memory Store (Short-Term)
# ============================================
# In-memory store for conversation history keyed by conversation_id
# Format: {conversation_id: [{"question": str, "answer": str, "timestamp": str}, ...]}
_conversation_store: dict = {}
MAX_HISTORY_LENGTH = 10  # Keep last 10 exchanges per conversation


def get_conversation_history(conversation_id: str) -> list:
    """Get conversation history for a conversation ID."""
    return _conversation_store.get(conversation_id, [])


def add_to_conversation(conversation_id: str, question: str, answer: str) -> None:
    """Add a Q&A pair to conversation history."""
    if conversation_id not in _conversation_store:
        _conversation_store[conversation_id] = []

    _conversation_store[conversation_id].append({
        "question": question,
        "answer": answer[:1000],  # Truncate long answers
        "timestamp": datetime.now().isoformat()
    })

    # Keep only last N exchanges
    if len(_conversation_store[conversation_id]) > MAX_HISTORY_LENGTH:
        _conversation_store[conversation_id] = _conversation_store[conversation_id][-MAX_HISTORY_LENGTH:]


def clear_conversation(conversation_id: str) -> None:
    """Clear conversation history for a conversation ID."""
    if conversation_id in _conversation_store:
        del _conversation_store[conversation_id]


# ============================================
# Pydantic Models
# ============================================

class AskRequest(BaseModel):
    question: str
    mode: str = "overview"  # overview, outcome, rules, reasoning
    k: int = 5


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    sources: List[dict]


class CaseSummary(BaseModel):
    id: str
    case_name: str
    cluster_id: Optional[int] = None
    area_of_law: Optional[str] = None


class CaseDetail(BaseModel):
    id: str
    case_name: str
    text_preview: str
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    cluster_description: Optional[str] = None
    area_of_law: Optional[str] = None
    remedy_type: Optional[List[str]] = None
    organizations: Optional[List[str]] = None
    places: Optional[List[str]] = None


class CaseFullText(BaseModel):
    """Full case text for detailed view."""
    id: str
    case_name: str
    text_full: str
    text_length: int
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    cluster_description: Optional[str] = None
    area_of_law: Optional[str] = None
    remedy_type: Optional[List[str]] = None
    organizations: Optional[List[str]] = None
    places: Optional[List[str]] = None


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    description: str
    case_count: int


class ClusterDetail(BaseModel):
    cluster_id: int
    label: str
    description: str
    sample_case: Optional[str] = None
    cases: List[CaseSummary]


# ============================================
# Data Loading Helpers
# ============================================

def load_cases_df() -> pd.DataFrame:
    """Load the main cases CSV with clusters and metadata."""
    csv_path = DATA_DIR / "summarization_with_metadata_clusters.csv"
    if not csv_path.exists():
        # Try alternative path
        csv_path = DATA_DIR / "summarization_with_metadata_clusters_labeled.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_cluster_labels() -> pd.DataFrame:
    """Load cluster labels CSV."""
    csv_path = DATA_DIR / "cluster_labels.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_cluster_stats() -> pd.DataFrame:
    """Load cluster stats CSV."""
    csv_path = DATA_DIR / "cluster_stats.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def parse_list_field(value) -> List[str]:
    """Parse a list field that might be stored as string."""
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Handle string representation of list
        cleaned = value.strip("[]").replace("'", "").replace('"', "")
        if not cleaned:
            return []
        return [x.strip() for x in cleaned.split(",") if x.strip()]
    return []


def extract_relevant_snippet(text: str, query: str, snippet_length: int = 400) -> str:
    """
    Extract the most relevant snippet from text based on query keywords.
    Skips boilerplate header text and finds passages with query terms.
    """
    if not text or not query:
        return text[:snippet_length] if text else ""

    # Skip common boilerplate patterns at the start
    skip_patterns = [
        r"^\(Slip Opinion\).*?NOTE:.*?(?=\n\n|\. [A-Z])",
        r"^.*?Syllabus.*?(?=\n\n)",
        r"^.*?SUPREME COURT OF THE UNITED STATES.*?(?=\n\n)",
    ]

    clean_text = text
    for pattern in skip_patterns:
        clean_text = re.sub(pattern, "", clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = clean_text.strip()

    # Extract keywords from query (simple approach)
    query_words = set(
        word.lower() for word in re.findall(r'\b\w{4,}\b', query)
        if word.lower() not in {'what', 'which', 'where', 'when', 'does', 'have', 'been', 'this', 'that', 'with', 'from', 'they', 'their', 'would', 'could', 'should'}
    )

    if not query_words:
        # No meaningful keywords, return from after boilerplate
        return clean_text[:snippet_length] + "..." if len(clean_text) > snippet_length else clean_text

    # Split into sentences and score them
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)

    best_start = 0
    best_score = 0

    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word in sentence_lower)
        if score > best_score:
            best_score = score
            best_start = i

    # Build snippet from best matching sentence and surrounding context
    start_idx = max(0, best_start - 1)
    snippet_sentences = []
    current_length = 0

    for sentence in sentences[start_idx:]:
        if current_length + len(sentence) > snippet_length:
            break
        snippet_sentences.append(sentence)
        current_length += len(sentence) + 1

    snippet = " ".join(snippet_sentences)

    if not snippet or len(snippet) < 50:
        # Fallback: return text after skipping first 500 chars of boilerplate
        fallback_start = min(500, len(clean_text) // 4)
        return clean_text[fallback_start:fallback_start + snippet_length] + "..."

    return snippet + ("..." if len(clean_text) > len(snippet) else "")


# ============================================
# API Endpoints
# ============================================

@router.get("/cases", response_model=List[CaseSummary])
async def list_cases(
    limit: int = 100,
    offset: int = 0,
    search: Optional[str] = Query(None, description="Search term for case name"),
    area_of_law: Optional[str] = Query(None, description="Filter by area of law"),
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
):
    """List all cases with basic info, with optional search and filters."""
    df = load_cases_df()
    if df.empty:
        return []

    # Apply search filter
    if search:
        search_lower = search.lower()
        df = df[df["case_name"].fillna("").str.lower().str.contains(search_lower, regex=False)]

    # Apply area of law filter
    if area_of_law:
        df = df[df["area_of_law"].fillna("").str.lower() == area_of_law.lower()]

    # Apply cluster filter
    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id]

    # Apply pagination
    total = len(df)
    df_page = df.iloc[offset:offset + limit]

    cases = []
    for _, row in df_page.iterrows():
        cases.append(CaseSummary(
            id=str(row.get("id", "")),
            case_name=str(row.get("case_name", "Unknown")),
            cluster_id=int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else None,
            area_of_law=str(row["area_of_law"]) if pd.notna(row.get("area_of_law")) else None,
        ))

    return cases


@router.get("/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str):
    """Get detailed information about a specific case."""
    df = load_cases_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    # Find the case
    case_row = df[df["id"] == case_id]
    if case_row.empty:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    row = case_row.iloc[0]

    # Get cluster info
    cluster_id = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else None
    cluster_label = None
    cluster_description = None

    if cluster_id is not None:
        labels_df = load_cluster_labels()
        if not labels_df.empty:
            cluster_row = labels_df[labels_df["cluster_id"] == cluster_id]
            if not cluster_row.empty:
                cluster_label = str(cluster_row.iloc[0].get("label", ""))
                cluster_description = str(cluster_row.iloc[0].get("description", ""))

    # Get text preview
    text = str(row.get("text_clean", ""))
    text_preview = text[:1000] + "..." if len(text) > 1000 else text

    return CaseDetail(
        id=str(row.get("id", "")),
        case_name=str(row.get("case_name", "Unknown")),
        text_preview=text_preview,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        cluster_description=cluster_description,
        area_of_law=str(row["area_of_law"]) if pd.notna(row.get("area_of_law")) else None,
        remedy_type=parse_list_field(row.get("remedy_type")),
        organizations=parse_list_field(row.get("orgs")),
        places=parse_list_field(row.get("places")),
    )


@router.get("/cases/{case_id}/full", response_model=CaseFullText)
async def get_case_full_text(case_id: str):
    """Get full case text for detailed side panel view."""
    df = load_cases_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    # Find the case
    case_row = df[df["id"] == case_id]
    if case_row.empty:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    row = case_row.iloc[0]

    # Get cluster info
    cluster_id = int(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else None
    cluster_label = None
    cluster_description = None

    if cluster_id is not None:
        labels_df = load_cluster_labels()
        if not labels_df.empty:
            cluster_row = labels_df[labels_df["cluster_id"] == cluster_id]
            if not cluster_row.empty:
                cluster_label = str(cluster_row.iloc[0].get("label", ""))
                cluster_description = str(cluster_row.iloc[0].get("description", ""))

    # Get FULL text (not truncated)
    text_full = str(row.get("text_clean", ""))

    return CaseFullText(
        id=str(row.get("id", "")),
        case_name=str(row.get("case_name", "Unknown")),
        text_full=text_full,
        text_length=len(text_full),
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        cluster_description=cluster_description,
        area_of_law=str(row["area_of_law"]) if pd.notna(row.get("area_of_law")) else None,
        remedy_type=parse_list_field(row.get("remedy_type")),
        organizations=parse_list_field(row.get("orgs")),
        places=parse_list_field(row.get("places")),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a legal question using RAG."""
    try:
        # Import the agent tools (lazy import to avoid startup issues)
        from agents.legal_tools import legal_rag, get_points, build_context

        # Call the RAG tool
        answer = legal_rag.invoke({
            "arguments": {
                "question": request.question,
                "mode": request.mode,
                "k": request.k,
            }
        })

        # Get source documents for citation
        points = get_points(request.question, limit=request.k)

        # Deduplicate by case_id and extract relevant snippets
        seen_cases = set()
        sources = []

        for p in points:
            case_id = p.payload.get("case_id", "")

            # Skip duplicates
            if case_id in seen_cases:
                continue
            seen_cases.add(case_id)

            # Get the full text
            full_text = p.payload.get("text", "") or p.payload.get("text_clean", "")

            # Extract relevant snippet based on the question
            snippet = extract_relevant_snippet(full_text, request.question, snippet_length=350)

            sources.append({
                "case_id": case_id,
                "case_name": p.payload.get("case_name", ""),
                "snippet": snippet,
            })

        return AskResponse(
            question=request.question,
            mode=request.mode,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.get("/clusters", response_model=List[ClusterSummary])
async def list_clusters():
    """List all clusters with labels."""
    labels_df = load_cluster_labels()
    stats_df = load_cluster_stats()

    if labels_df.empty:
        return []

    clusters = []
    for _, row in labels_df.iterrows():
        cluster_id = int(row["cluster_id"])

        # Get case count from stats
        case_count = 0
        if not stats_df.empty:
            stat_row = stats_df[stats_df["cluster_id"] == cluster_id]
            if not stat_row.empty:
                case_count = int(stat_row.iloc[0].get("n_cases", 0))

        clusters.append(ClusterSummary(
            cluster_id=cluster_id,
            label=str(row.get("label", f"Cluster {cluster_id}")),
            description=str(row.get("description", "")),
            case_count=case_count,
        ))

    return clusters


@router.get("/clusters/{cluster_id}", response_model=ClusterDetail)
async def get_cluster(cluster_id: int):
    """Get detailed information about a cluster including its cases."""
    labels_df = load_cluster_labels()
    stats_df = load_cluster_stats()
    cases_df = load_cases_df()

    if labels_df.empty:
        raise HTTPException(status_code=404, detail="No cluster data available")

    # Find the cluster
    cluster_row = labels_df[labels_df["cluster_id"] == cluster_id]
    if cluster_row.empty:
        raise HTTPException(status_code=404, detail=f"Cluster not found: {cluster_id}")

    row = cluster_row.iloc[0]

    # Get sample case from stats
    sample_case = None
    if not stats_df.empty:
        stat_row = stats_df[stats_df["cluster_id"] == cluster_id]
        if not stat_row.empty:
            sample_case = str(stat_row.iloc[0].get("sample_case", ""))

    # Get cases in this cluster
    cluster_cases = []
    if not cases_df.empty:
        cases_in_cluster = cases_df[cases_df["cluster_id"] == cluster_id]
        for _, case_row in cases_in_cluster.head(20).iterrows():  # Limit to 20 cases
            cluster_cases.append(CaseSummary(
                id=str(case_row.get("id", "")),
                case_name=str(case_row.get("case_name", "Unknown")),
                cluster_id=cluster_id,
                area_of_law=str(case_row["area_of_law"]) if pd.notna(case_row.get("area_of_law")) else None,
            ))

    return ClusterDetail(
        cluster_id=cluster_id,
        label=str(row.get("label", f"Cluster {cluster_id}")),
        description=str(row.get("description", "")),
        sample_case=sample_case,
        cases=cluster_cases,
    )


@router.post("/metadata/{case_id}")
async def extract_case_metadata(case_id: str):
    """Extract metadata for a specific case using the metadata agent."""
    df = load_cases_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    case_row = df[df["id"] == case_id]
    if case_row.empty:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    row = case_row.iloc[0]
    text = str(row.get("text_clean", ""))

    if not text:
        raise HTTPException(status_code=400, detail="No text available for this case")

    try:
        from agents.legal_tools import extract_metadata

        result = extract_metadata.invoke({
            "arguments": {"text": text}
        })

        return {
            "case_id": case_id,
            "metadata": result,
        }

    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {str(e)}")


# ============================================
# Agent Streaming Endpoints (SSE)
# ============================================

class AgentAnalyzeRequest(BaseModel):
    question: str
    case_id: Optional[str] = None
    cluster_id: Optional[int] = None


@router.get("/agent/stream")
async def agent_stream(
    question: str,
    case_id: Optional[str] = Query(None),
    cluster_id: Optional[int] = Query(None),
    conversation_id: Optional[str] = Query(None, description="Conversation ID for maintaining context across queries"),
    profile_id: Optional[str] = Query(None, description="User profile ID for long-term memory")
):
    """
    Server-Sent Events endpoint for real-time multi-agent analysis.

    Streams agent events as they occur, enabling transparent visibility
    into the reasoning process.

    Supports conversation memory: pass conversation_id to maintain context
    across multiple queries in the same session.

    Supports profile memory: pass profile_id to track query history and
    auto-detect user preferences over time.

    Session logs are automatically saved to logs/sessions/ for debugging.
    """
    # Create session logger for this query
    session = SessionLogger(question)
    session.set_metadata("case_id", case_id)
    session.set_metadata("cluster_id", cluster_id)
    session.set_metadata("conversation_id", conversation_id)
    session.set_metadata("profile_id", profile_id)
    session.set_metadata("request_id", get_request_id())

    # Get conversation history if conversation_id provided
    conversation_history = []
    if conversation_id:
        conversation_history = get_conversation_history(conversation_id)
        session.set_metadata("history_length", len(conversation_history))

    # Get profile context if profile_id provided
    profile_context = ""
    if profile_id:
        from app.profiles import get_profile_context
        profile_context = get_profile_context(profile_id)

    logger.info(
        f"Agent stream started: {question[:100]}",
        extra={"extra_data": {
            "session_id": session.session_id,
            "case_id": case_id,
            "cluster_id": cluster_id,
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "history_length": len(conversation_history)
        }}
    )

    async def event_generator():
        final_answer = None
        detected_area_of_law = None
        try:
            from agents.orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator()

            async for event in orchestrator.run_with_streaming(
                question=question,
                case_id=case_id,
                cluster_id=cluster_id,
                conversation_history=conversation_history,
                profile_context=profile_context
            ):
                # Log event to session
                session.log_event(event)

                # Capture final answer for session metadata
                if event.get("event") == "agent_complete" and event.get("agent") == "supervisor":
                    final_answer = event.get("data", {}).get("final_answer", "")

                # Capture detected area of law from metadata specialist
                if event.get("event") == "extracted_entities":
                    detected_area_of_law = event.get("data", {}).get("area_of_law")

                yield f"data: {json.dumps(event)}\n\n"

            # Store Q&A pair in conversation history
            if conversation_id and final_answer:
                add_to_conversation(conversation_id, question, final_answer)

            # Save session log first so we have the filename
            session.set_metadata("final_answer_preview", final_answer[:500] if final_answer else None)
            session.set_metadata("status", "success")
            log_file = session.save()

            # Update user profile with query (after session save to include log filename)
            if profile_id and final_answer:
                from app.profiles import add_query_to_history
                add_query_to_history(
                    profile_id=profile_id,
                    question=question,
                    area_of_law=detected_area_of_law,
                    answer_preview=final_answer[:200] if final_answer else None,
                    session_log_file=session.filename
                )

            # Send completion signal with session info
            completion_event = {
                'event': 'stream_end',
                'message': 'Analysis complete',
                'session_id': session.session_id,
                'log_file': session.filename,
                'conversation_id': conversation_id,
                'profile_id': profile_id
            }
            yield f"data: {json.dumps(completion_event)}\n\n"

            logger.info(
                f"Agent stream completed",
                extra={"extra_data": session.get_summary()}
            )

        except Exception as e:
            logger.error(f"Agent stream error: {str(e)}", exc_info=True)

            session.set_metadata("status", "error")
            session.set_metadata("error", str(e))
            session.save()

            error_event = {
                "event": "error",
                "agent": "system",
                "message": f"Stream error: {str(e)}",
                "session_id": session.session_id
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/agent/analyze", response_model=dict)
async def agent_analyze(request: AgentAnalyzeRequest):
    """
    Full multi-agent analysis endpoint (non-streaming).

    Returns the complete analysis with all events and final answer.
    Session logs are automatically saved.
    """
    # Create session logger
    session = SessionLogger(request.question)
    session.set_metadata("case_id", request.case_id)
    session.set_metadata("cluster_id", request.cluster_id)
    session.set_metadata("request_id", get_request_id())
    session.set_metadata("endpoint", "analyze")

    logger.info(
        f"Agent analyze started: {request.question[:100]}",
        extra={"extra_data": {"session_id": session.session_id}}
    )

    try:
        from agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Collect all events
        events = []
        final_answer = None

        async for event in orchestrator.run_with_streaming(
            question=request.question,
            case_id=request.case_id,
            cluster_id=request.cluster_id
        ):
            events.append(event)
            session.log_event(event)

            if event.get("event") == "agent_complete" and event.get("agent") == "supervisor":
                final_answer = event.get("data", {}).get("final_answer")

        # Save session
        session.set_metadata("status", "success")
        session.set_metadata("final_answer_preview", final_answer[:500] if final_answer else None)
        log_file = session.save()

        logger.info(
            f"Agent analyze completed",
            extra={"extra_data": session.get_summary()}
        )

        return {
            "question": request.question,
            "events": events,
            "final_answer": final_answer,
            "event_count": len(events),
            "session_id": session.session_id,
            "log_file": session.filename
        }

    except Exception as e:
        logger.error(f"Agent analysis failed: {str(e)}", exc_info=True)
        session.set_metadata("status", "error")
        session.set_metadata("error", str(e))
        session.save()
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")


# ============================================
# Session Log Endpoints
# ============================================

@router.get("/logs/sessions")
async def list_session_logs(limit: int = 50):
    """
    List recent session logs.

    Returns metadata about each session without the full event data.
    """
    from app.logger import SESSION_LOG_DIR

    logs = []
    log_files = sorted(SESSION_LOG_DIR.glob("*.json"), reverse=True)[:limit]

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
                logs.append({
                    "filename": log_file.name,
                    "session_id": data.get("session_id"),
                    "question": data.get("question", "")[:100],
                    "start_time": data.get("start_time"),
                    "duration_ms": data.get("duration_ms"),
                    "event_count": data.get("event_count"),
                    "status": data.get("metadata", {}).get("status", "unknown")
                })
        except Exception:
            continue

    return {"sessions": logs, "total": len(logs)}


@router.get("/logs/sessions/{filename}")
async def get_session_log(filename: str):
    """
    Get full session log by filename.

    Returns the complete session data including all events.
    """
    from app.logger import SESSION_LOG_DIR

    log_path = SESSION_LOG_DIR / filename

    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Session log not found: {filename}")

    if not log_path.suffix == ".json":
        raise HTTPException(status_code=400, detail="Invalid log file format")

    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log: {str(e)}")


# ============================================
# Conversation Memory Endpoints
# ============================================

@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history for a given conversation ID.

    Returns the list of Q&A pairs in the conversation.
    """
    history = get_conversation_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "history": history,
        "message_count": len(history)
    }


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Clear conversation history for a given conversation ID.

    Use this to start a fresh conversation thread.
    """
    clear_conversation(conversation_id)
    logger.info(f"Conversation cleared: {conversation_id}")
    return {
        "conversation_id": conversation_id,
        "status": "cleared"
    }


# ============================================
# User Profile Endpoints (Long-Term Memory)
# ============================================

class CreateProfileRequest(BaseModel):
    name: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    preferences: Optional[dict] = None
    topics_summary: Optional[str] = None


@router.get("/profiles")
async def list_all_profiles():
    """
    List all user profiles.

    Returns summary info for each profile including name, query count,
    and detected areas of law.
    """
    from app.profiles import list_profiles
    profiles = list_profiles()
    return {"profiles": profiles, "total": len(profiles)}


@router.post("/profiles")
async def create_new_profile(request: CreateProfileRequest):
    """
    Create a new user profile.

    Profiles enable long-term memory across sessions, tracking:
    - Research preferences (auto-detected from queries)
    - Query history (last 20 questions)
    - Topic interests
    """
    from app.profiles import create_profile

    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Profile name is required")

    profile = create_profile(request.name.strip())
    logger.info(f"Created profile: {profile['profile_id']}")
    return profile


@router.get("/profiles/{profile_id}")
async def get_profile_detail(profile_id: str):
    """
    Get detailed information about a profile.

    Returns full profile data including preferences, query history,
    and topics summary.
    """
    from app.profiles import get_profile

    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    return profile


@router.put("/profiles/{profile_id}")
async def update_profile_data(profile_id: str, request: UpdateProfileRequest):
    """
    Update a profile's name, preferences, or topics summary.
    """
    from app.profiles import update_profile

    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.preferences is not None:
        updates["preferences"] = request.preferences
    if request.topics_summary is not None:
        updates["topics_summary"] = request.topics_summary

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    profile = update_profile(profile_id, updates)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    return profile


@router.delete("/profiles/{profile_id}")
async def delete_user_profile(profile_id: str):
    """
    Delete a user profile.
    """
    from app.profiles import delete_profile

    success = delete_profile(profile_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    return {"profile_id": profile_id, "status": "deleted"}


@router.get("/profiles/{profile_id}/context")
async def get_profile_context_text(profile_id: str):
    """
    Get the formatted context string for a profile.

    This is the context that gets included in LLM prompts to personalize
    responses based on user preferences and history.
    """
    from app.profiles import get_profile_context

    context = get_profile_context(profile_id)
    return {"profile_id": profile_id, "context": context}
