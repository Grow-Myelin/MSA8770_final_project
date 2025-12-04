"""
orchestrator.py - Multi-Agent Legal Analysis System

LangGraph-based supervisor with specialist agents for transparent,
real-time legal case analysis with streaming events.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, TypedDict, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.legal_tools import legal_rag, extract_metadata, explore_topic, get_points, build_context

# Logger for orchestrator
logger = logging.getLogger("orchestrator")


# ============================================
# Configuration
# ============================================

MODEL_NAME = "gpt-5-mini"


# ============================================
# Event Types for Streaming
# ============================================

@dataclass
class AgentEvent:
    """Event emitted by agents for real-time streaming."""
    event: str  # agent_start, agent_thinking, tool_call, tool_result, agent_complete, error, llm_prompt
    agent: str  # supervisor, rag_specialist, cluster_specialist, metadata_specialist
    message: str
    data: Optional[dict] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "agent": self.agent,
            "message": self.message,
            "data": self.data or {},
            "timestamp": self.timestamp or datetime.now().isoformat()
        }


def _format_messages_for_event(messages) -> dict:
    """Format LangChain messages for llm_prompt event data."""
    formatted = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "human"
        else:
            role = "unknown"
        formatted.append({
            "role": role,
            "content": m.content
        })
    return {
        "messages": formatted,
        "total_length": sum(len(m.content) for m in messages)
    }


# ============================================
# Agent State Definition
# ============================================

class OrchestratorState(TypedDict):
    """State passed between agents in the graph."""
    question: str
    case_id: Optional[str]
    conversation_history: Optional[list]  # Previous Q&A pairs for context
    profile_context: Optional[str]  # User profile preferences/history context
    routing_decision: Optional[dict]
    rag_result: Optional[str]
    rag_sources: Optional[list]  # Grounded sources from corpus
    cluster_result: Optional[dict]
    metadata_result: Optional[dict]
    final_answer: Optional[str]
    events: list  # Accumulated events for streaming


# ============================================
# System Prompts
# ============================================

SUPERVISOR_SYSTEM_PROMPT = """You are a legal analysis supervisor coordinating specialist agents.

Given a user question about legal cases, analyze the question and decide which specialists to invoke:
- RAG Specialist: For doctrinal questions, legal rules, standards of review, case law interpretation
- Cluster Specialist: For topic analysis, related cases, thematic context (requires a cluster_id)
- Metadata Specialist: For entity extraction, area of law classification, remedy types

You can invoke multiple specialists if the question requires comprehensive analysis.

Respond with a JSON object indicating your routing decision:
{
    "reasoning": "Your analysis of what the question requires",
    "invoke_rag": true/false,
    "rag_mode": "overview|outcome|rules|reasoning",
    "invoke_metadata": true/false,
    "invoke_cluster": true/false,
    "cluster_id": null or integer
}

Always explain your reasoning before making the routing decision."""

RAG_SPECIALIST_PROMPT = """You are a legal RAG specialist with access to a corpus of U.S. appellate opinions.

Use the legal_rag tool to answer doctrinal questions. Choose the appropriate mode:
- "overview": General case summary
- "outcome": Court's decision and relief
- "rules": Key legal rules and tests (most common)
- "reasoning": Court's reasoning and argument handling

When calling legal_rag, pass arguments as: {"question": "...", "mode": "...", "k": 5}

Always provide substantive legal analysis based on the retrieved cases.
If the context is insufficient, clearly state what information is missing."""

METADATA_SPECIALIST_PROMPT = """You are a metadata extraction specialist for legal cases.

Use the extract_metadata tool to identify:
- Organizations mentioned (courts, agencies, companies)
- Places mentioned (jurisdictions, locations)
- Area of law (criminal, civil, administrative, etc.)
- Remedy types (damages, injunction, declaratory, etc.)

When calling extract_metadata, pass: {"question": "...", "k": 5} to retrieve and analyze relevant cases.

Provide structured information about the case's key entities and classifications."""

CLUSTER_SPECIALIST_PROMPT = """You are a topic analysis specialist for legal case clustering.

Use the explore_topic tool to analyze case clusters. When given a cluster_id:
1. Identify what types of cases are in that cluster
2. Explain the common themes and legal issues
3. Provide context about how cases in this cluster relate

When calling explore_topic, pass: {"cluster_id": <integer>}

Help users understand where cases fit in the broader legal landscape."""

SYNTHESIS_PROMPT = """You are synthesizing results from multiple specialist agents into a comprehensive answer.

Based on the specialist results provided, create a unified response that:
1. Directly answers the user's question
2. Integrates insights from all available specialist analyses
3. Highlights key legal principles, entities, and classifications
4. Notes any limitations or areas needing further research

Be concise but thorough. Structure your response clearly."""


# ============================================
# Agent Orchestrator Class
# ============================================

class AgentOrchestrator:
    """
    Multi-agent orchestrator for legal analysis with real-time streaming.

    Uses a supervisor pattern where a central agent routes to specialists
    and synthesizes their results.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        self.events_queue: asyncio.Queue = asyncio.Queue()

    async def _emit_event(self, event: AgentEvent):
        """Add event to queue for streaming and log it."""
        await self.events_queue.put(event)

        # Log event for debugging
        log_level = logging.DEBUG if event.event in ("agent_thinking",) else logging.INFO
        if event.event == "error":
            log_level = logging.ERROR

        logger.log(
            log_level,
            f"[{event.agent}] {event.event}: {event.message[:100]}",
            extra={"extra_data": {"event_type": event.event, "agent": event.agent}}
        )

    async def _supervisor_route(self, state: OrchestratorState) -> OrchestratorState:
        """Supervisor analyzes the question and decides routing."""
        await self._emit_event(AgentEvent(
            event="agent_start",
            agent="supervisor",
            message=f"Analyzing question: \"{state['question'][:100]}...\""
        ))

        # Build context including conversation history
        history = state.get("conversation_history", [])
        history_context = ""
        if history:
            history_lines = []
            for h in history[-3:]:  # Last 3 exchanges for context
                history_lines.append(f"User: {h.get('question', '')[:200]}")
                history_lines.append(f"Assistant: {h.get('answer', '')[:300]}...")
            history_context = "\n\nConversation History:\n" + "\n".join(history_lines)

        # Include user profile context if available
        profile_context = state.get("profile_context", "")
        profile_section = f"\n\n{profile_context}" if profile_context else ""

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {state['question']}\nCase ID (if provided): {state.get('case_id', 'None')}{history_context}{profile_section}")
        ]

        context_notes = []
        if history:
            context_notes.append("conversation context")
        if profile_context:
            context_notes.append("user profile")
        context_note = f" (with {', '.join(context_notes)})" if context_notes else ""

        await self._emit_event(AgentEvent(
            event="agent_thinking",
            agent="supervisor",
            message=f"Determining which specialists to invoke...{context_note}"
        ))

        # Emit llm_prompt event with full prompt details
        await self._emit_event(AgentEvent(
            event="llm_prompt",
            agent="supervisor",
            message="Sending prompt to LLM for routing decision",
            data=_format_messages_for_event(messages)
        ))

        response = await asyncio.to_thread(self.llm.invoke, messages)

        # Parse routing decision from response
        try:
            # Try to extract JSON from response
            content = response.content
            # Find JSON in response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                routing = json.loads(json_match.group())
            else:
                # Default routing if no JSON found
                routing = {
                    "reasoning": content,
                    "invoke_rag": True,
                    "rag_mode": "rules",
                    "invoke_metadata": True,
                    "invoke_cluster": False,
                    "cluster_id": None
                }
        except json.JSONDecodeError:
            routing = {
                "reasoning": "Could not parse routing decision, using defaults",
                "invoke_rag": True,
                "rag_mode": "rules",
                "invoke_metadata": True,
                "invoke_cluster": False,
                "cluster_id": None
            }

        await self._emit_event(AgentEvent(
            event="agent_thinking",
            agent="supervisor",
            message=f"Routing decision: {routing.get('reasoning', 'Analyzing...')[:200]}",
            data={
                "invoke_rag": routing.get("invoke_rag", False),
                "invoke_metadata": routing.get("invoke_metadata", False),
                "invoke_cluster": routing.get("invoke_cluster", False)
            }
        ))

        # Merge with any pre-existing routing (e.g., cluster_id from user)
        existing = state.get("routing_decision") or {}
        if existing.get("invoke_cluster") and existing.get("cluster_id"):
            routing["invoke_cluster"] = True
            routing["cluster_id"] = existing["cluster_id"]

        state["routing_decision"] = routing
        return state

    async def _run_rag_specialist(self, state: OrchestratorState) -> OrchestratorState:
        """RAG specialist retrieves and analyzes legal documents."""
        routing = state.get("routing_decision", {})
        if not routing.get("invoke_rag", False):
            return state

        await self._emit_event(AgentEvent(
            event="agent_start",
            agent="rag_specialist",
            message="Starting legal document retrieval..."
        ))

        mode = routing.get("rag_mode", "rules")

        # Step 1: Retrieve raw passages from vector DB (GROUNDED DATA)
        # Retrieve more chunks for better coverage (chunked collection has smaller pieces)
        k_retrieve = 10  # Retrieve more, build_context will limit by budget
        await self._emit_event(AgentEvent(
            event="tool_call",
            agent="rag_specialist",
            message="Querying Qdrant vector database for relevant case passages...",
            data={"tool": "get_points", "query": state["question"][:100] + "...", "k": k_retrieve}
        ))

        try:
            # Get raw passages from Qdrant (may be chunks or full docs)
            points = await asyncio.to_thread(get_points, state["question"], k_retrieve)

            # Extract source information (include chunk metadata if available)
            sources = []
            for i, p in enumerate(points):
                case_name = p.payload.get("case_name", "Unknown Case")
                case_id = p.payload.get("case_id", "")
                chunk_type = p.payload.get("chunk_type", "")
                text = p.payload.get("text", "") or p.payload.get("text_clean", "")
                # Store full chunk text for expandable view (chunks are typically <2000 chars)
                # Frontend will show preview and allow expansion to full text
                snippet = text[:2000] + "..." if len(text) > 2000 else text
                sources.append({
                    "rank": i + 1,
                    "case_name": case_name,
                    "case_id": case_id,
                    "chunk_type": chunk_type,
                    "snippet": snippet,
                    "text_length": len(text),
                    "score": getattr(p, 'score', None)
                })

            # Emit GROUNDED DATA event - this is directly from the corpus
            await self._emit_event(AgentEvent(
                event="retrieved_sources",
                agent="rag_specialist",
                message=f"Retrieved {len(sources)} relevant passages from case law corpus",
                data={
                    "source_type": "GROUNDED - Direct from Corpus",
                    "sources": sources
                }
            ))

            # Build context from retrieved passages
            context = await asyncio.to_thread(build_context, points)

            await self._emit_event(AgentEvent(
                event="tool_call",
                agent="rag_specialist",
                message=f"Sending context to LLM for synthesis (mode=\"{mode}\")",
                data={
                    "tool": "legal_rag",
                    "mode": mode,
                    "context_length": len(context),
                    "input_preview": context[:300] + "..." if len(context) > 300 else context
                }
            ))

            # Step 2: Call LLM to synthesize answer (LLM GENERATED)
            result = await asyncio.to_thread(
                legal_rag.invoke,
                {"arguments": {"question": state["question"], "mode": mode, "k": 5}}
            )

            # Emit LLM GENERATED result
            await self._emit_event(AgentEvent(
                event="llm_generated",
                agent="rag_specialist",
                message="LLM synthesized answer from retrieved passages",
                data={
                    "source_type": "LLM GENERATED - Synthesized from corpus",
                    "answer_preview": str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                }
            ))

            # Store both grounded sources and generated answer
            state["rag_result"] = result
            state["rag_sources"] = sources

            await self._emit_event(AgentEvent(
                event="agent_complete",
                agent="rag_specialist",
                message=f"RAG complete: {len(sources)} sources retrieved, answer synthesized"
            ))

        except Exception as e:
            await self._emit_event(AgentEvent(
                event="error",
                agent="rag_specialist",
                message=f"Error in RAG: {str(e)}"
            ))
            state["rag_result"] = f"Error: {str(e)}"

        return state

    async def _run_metadata_specialist(self, state: OrchestratorState) -> OrchestratorState:
        """Metadata specialist extracts entities and classifications."""
        routing = state.get("routing_decision", {})
        if not routing.get("invoke_metadata", False):
            return state

        await self._emit_event(AgentEvent(
            event="agent_start",
            agent="metadata_specialist",
            message="Starting metadata extraction using NER and rule-based classifiers..."
        ))

        # Show input data
        await self._emit_event(AgentEvent(
            event="tool_call",
            agent="metadata_specialist",
            message="Retrieving case text from corpus for entity extraction...",
            data={
                "tool": "extract_metadata",
                "input": {
                    "question": state["question"][:100] + "...",
                    "k": 5,
                    "method": "spaCy NER + keyword classification"
                }
            }
        ))

        try:
            result = await asyncio.to_thread(
                extract_metadata.invoke,
                {"arguments": {"question": state["question"], "k": 5}}
            )

            # Show grounded extraction results
            if isinstance(result, dict) and "error" not in result:
                await self._emit_event(AgentEvent(
                    event="extracted_entities",
                    agent="metadata_specialist",
                    message="Extracted entities and classifications from corpus text",
                    data={
                        "source_type": "GROUNDED - Extracted from corpus via NER",
                        "organizations": result.get("organizations", [])[:10],
                        "places": result.get("places", [])[:10],
                        "area_of_law": result.get("area_of_law"),
                        "remedy_types": result.get("remedy_type", []),
                        "extraction_method": "spaCy en_core_web_sm + rule-based keyword matching"
                    }
                ))

            state["metadata_result"] = result

            await self._emit_event(AgentEvent(
                event="agent_complete",
                agent="metadata_specialist",
                message=f"Metadata extraction complete: area={result.get('area_of_law', 'N/A')}, {len(result.get('organizations', []))} orgs, {len(result.get('places', []))} places"
            ))

        except Exception as e:
            await self._emit_event(AgentEvent(
                event="error",
                agent="metadata_specialist",
                message=f"Error in metadata extraction: {str(e)}"
            ))
            state["metadata_result"] = {"error": str(e)}

        return state

    async def _run_cluster_specialist(self, state: OrchestratorState) -> OrchestratorState:
        """Cluster specialist analyzes topic groupings."""
        routing = state.get("routing_decision", {})
        if not routing.get("invoke_cluster", False):
            return state

        cluster_id = routing.get("cluster_id") or state.get("case_id")
        if not cluster_id:
            await self._emit_event(AgentEvent(
                event="agent_thinking",
                agent="cluster_specialist",
                message="No cluster_id provided, skipping cluster analysis"
            ))
            return state

        await self._emit_event(AgentEvent(
            event="agent_start",
            agent="cluster_specialist",
            message=f"Analyzing cluster {cluster_id}..."
        ))

        await self._emit_event(AgentEvent(
            event="tool_call",
            agent="cluster_specialist",
            message=f"Calling explore_topic(cluster_id={cluster_id})",
            data={"tool": "explore_topic", "cluster_id": cluster_id}
        ))

        try:
            result = await asyncio.to_thread(
                explore_topic.invoke,
                {"arguments": {"cluster_id": int(cluster_id)}}
            )

            await self._emit_event(AgentEvent(
                event="tool_result",
                agent="cluster_specialist",
                message="Retrieved cluster information",
                data=result if isinstance(result, dict) else {"result": str(result)}
            ))

            state["cluster_result"] = result

            await self._emit_event(AgentEvent(
                event="agent_complete",
                agent="cluster_specialist",
                message=f"Cluster: {result.get('label', 'Unknown')}" if isinstance(result, dict) else "Analysis complete"
            ))

        except Exception as e:
            await self._emit_event(AgentEvent(
                event="error",
                agent="cluster_specialist",
                message=f"Error in cluster analysis: {str(e)}"
            ))
            state["cluster_result"] = {"error": str(e)}

        return state

    async def _synthesize_results(self, state: OrchestratorState) -> OrchestratorState:
        """Supervisor synthesizes all specialist results into final answer."""
        await self._emit_event(AgentEvent(
            event="agent_start",
            agent="supervisor",
            message="Synthesizing specialist results..."
        ))

        # Build synthesis context
        context_parts = [f"Original Question: {state['question']}"]

        if state.get("rag_result"):
            context_parts.append(f"\n--- RAG Analysis ---\n{state['rag_result']}")

        if state.get("metadata_result"):
            meta = state["metadata_result"]
            if isinstance(meta, dict) and "error" not in meta:
                context_parts.append(f"\n--- Metadata ---\nArea of Law: {meta.get('area_of_law', 'N/A')}\nRemedy Types: {meta.get('remedy_type', [])}\nOrganizations: {meta.get('organizations', [])[:5]}\nPlaces: {meta.get('places', [])[:5]}")

        if state.get("cluster_result"):
            cluster = state["cluster_result"]
            if isinstance(cluster, dict) and "error" not in cluster:
                context_parts.append(f"\n--- Cluster Analysis ---\nCluster: {cluster.get('label', 'N/A')}\nDescription: {cluster.get('description', 'N/A')}")

        synthesis_context = "\n".join(context_parts)

        await self._emit_event(AgentEvent(
            event="agent_thinking",
            agent="supervisor",
            message="Integrating insights from all specialists..."
        ))

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=synthesis_context)
        ]

        # Emit llm_prompt event with full prompt details for synthesis
        await self._emit_event(AgentEvent(
            event="llm_prompt",
            agent="supervisor",
            message="Sending prompt to LLM for final synthesis",
            data=_format_messages_for_event(messages)
        ))

        response = await asyncio.to_thread(self.llm.invoke, messages)

        # Build structured final result with sources
        sources = state.get("rag_sources", [])
        metadata = state.get("metadata_result", {})

        final_result = {
            "answer": response.content,
            "sources": sources,
            "metadata": {
                "area_of_law": metadata.get("area_of_law") if isinstance(metadata, dict) else None,
                "remedy_types": metadata.get("remedy_type", []) if isinstance(metadata, dict) else [],
                "organizations": metadata.get("organizations", [])[:5] if isinstance(metadata, dict) else [],
                "places": metadata.get("places", [])[:5] if isinstance(metadata, dict) else []
            }
        }

        state["final_answer"] = response.content
        state["final_result"] = final_result

        await self._emit_event(AgentEvent(
            event="agent_complete",
            agent="supervisor",
            message="Analysis complete",
            data={
                "final_answer": response.content,
                "sources_used": len(sources),
                "grounded_sources": sources,
                "grounded_metadata": final_result["metadata"]
            }
        ))

        return state

    async def run_with_streaming(
        self,
        question: str,
        case_id: Optional[str] = None,
        cluster_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
        profile_context: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Run the multi-agent analysis with real-time event streaming.

        Yields events as they occur for real-time UI updates.

        Args:
            question: The legal question to analyze
            case_id: Optional case ID for case-specific analysis
            cluster_id: Optional cluster ID to analyze a specific topic cluster
            conversation_history: Previous Q&A pairs for context continuity
            profile_context: User profile context for personalized responses
        """
        start_time = time.time()
        logger.info(
            f"Starting orchestrator pipeline: {question[:100]}",
            extra={"extra_data": {"case_id": case_id, "cluster_id": cluster_id, "history_len": len(conversation_history or []), "has_profile": bool(profile_context)}}
        )

        # Initialize state
        state: OrchestratorState = {
            "question": question,
            "case_id": case_id,
            "conversation_history": conversation_history or [],
            "profile_context": profile_context or "",
            "routing_decision": None,
            "rag_result": None,
            "rag_sources": None,
            "cluster_result": None,
            "metadata_result": None,
            "final_answer": None,
            "events": []
        }

        # If cluster_id provided, pre-set routing to include cluster analysis
        if cluster_id is not None:
            state["routing_decision"] = {
                "invoke_cluster": True,
                "cluster_id": cluster_id
            }

        # Create fresh queue for this run
        self.events_queue = asyncio.Queue()

        # Task to collect events
        async def run_pipeline():
            try:
                # Phase 1: Supervisor routes
                state_after_route = await self._supervisor_route(state)

                # Phase 2: Run specialists in parallel where possible
                routing = state_after_route.get("routing_decision", {})

                tasks = []
                if routing.get("invoke_rag"):
                    tasks.append(self._run_rag_specialist(state_after_route))
                if routing.get("invoke_metadata"):
                    tasks.append(self._run_metadata_specialist(state_after_route))
                if routing.get("invoke_cluster"):
                    tasks.append(self._run_cluster_specialist(state_after_route))

                if tasks:
                    # Run specialists in parallel
                    await asyncio.gather(*tasks)

                # Phase 3: Synthesize results
                await self._synthesize_results(state_after_route)

                # Signal completion
                await self.events_queue.put(None)

            except Exception as e:
                await self._emit_event(AgentEvent(
                    event="error",
                    agent="supervisor",
                    message=f"Pipeline error: {str(e)}"
                ))
                await self.events_queue.put(None)

        # Start pipeline in background
        pipeline_task = asyncio.create_task(run_pipeline())

        # Yield events as they come
        event_count = 0
        while True:
            event = await self.events_queue.get()
            if event is None:
                break
            event_count += 1
            yield event.to_dict()

        # Ensure pipeline completes
        await pipeline_task

        # Log completion
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Orchestrator pipeline completed",
            extra={"extra_data": {"duration_ms": duration_ms, "event_count": event_count}}
        )

    def run_sync(
        self,
        question: str,
        case_id: Optional[str] = None,
        cluster_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
        profile_context: Optional[str] = None
    ) -> dict:
        """
        Synchronous version that collects all events and returns final result.

        Returns dict with 'events' list and 'final_answer'.
        """
        async def collect_results():
            events = []
            final_answer = None
            async for event in self.run_with_streaming(question, case_id, cluster_id, conversation_history, profile_context):
                events.append(event)
                if event.get("event") == "agent_complete" and event.get("agent") == "supervisor":
                    final_answer = event.get("data", {}).get("final_answer")
            return {"events": events, "final_answer": final_answer}

        return asyncio.run(collect_results())


# ============================================
# CLI Demo
# ============================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Multi-Agent Legal Analysis Orchestrator")
    print("=" * 60)

    orchestrator = AgentOrchestrator()

    # Demo question
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What standard of review applies in unemployment-benefit administrative appeals?"

    print(f"\nQuestion: {question}\n")
    print("-" * 60)

    result = orchestrator.run_sync(question)

    print("\n--- Events ---")
    for event in result["events"]:
        agent = event.get("agent", "unknown")
        event_type = event.get("event", "unknown")
        message = event.get("message", "")

        icons = {
            "supervisor": "🎯",
            "rag_specialist": "📚",
            "metadata_specialist": "🏷️",
            "cluster_specialist": "🔗"
        }
        icon = icons.get(agent, "•")

        print(f"{icon} [{agent}] {event_type}: {message[:100]}")

    print("\n--- Final Answer ---")
    print(result.get("final_answer", "No answer generated"))
