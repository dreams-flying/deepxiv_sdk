"""
State definitions for the RAG ReAct agent.

Mirrors DeepXiv's PaperInfo / AgentState pattern,
adapted for internal document retrieval.
"""

import operator
from typing import TypedDict, List, Dict, Annotated


class ChunkInfo(TypedDict):
    """Lightweight metadata for a single document chunk — no full text."""
    chunk_id:      str   # ES _id, pass to read_chunk()
    section_title: str   # Section heading detected at index time (may be empty)
    description:   str   # First 60 chars of chunk content — cheap relevance signal
    page_num:      int
    token_count:   int   # Character count of the chunk


class DocInfo(TypedDict):
    """All information known about a loaded document."""
    file_id:       str
    file_name:     str
    file_urls:     str
    total_tokens:  int
    chunks:        Dict[str, ChunkInfo]  # chunk_id -> ChunkInfo
    loaded_chunks: Dict[str, str]        # chunk_id -> full content (read cache)


class AgentState(TypedDict):
    """Complete state for the RAG ReAct agent."""

    # Session context (used for ES filtering)
    session_id: str

    # Documents being tracked this session (file_id -> DocInfo)
    docs: Dict[str, DocInfo]

    # Conversation history — accumulated across rounds via operator.add
    messages: Annotated[List[Dict], operator.add]

    # Current question and last LLM response text
    question: str
    response: str

    # Status trail: ["planning", "tool_call", "tool_response", "answer", ...]
    status: List[str]

    # Loop control
    round:                   int
    num_llm_calls_available: int
    start_time:              float

    # Final output fields
    prediction:  str
    termination: str

    # Caches — avoid redundant ES round-trips
    chunk_cache:          Dict[str, str]   # chunk_id -> full content
    search_results_cache: List[Dict]
