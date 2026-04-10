"""
State definitions for the RAG ReAct agent.

Mirrors DeepXiv's PaperInfo / AgentState pattern,
adapted for internal document retrieval.
"""

import operator
from typing import TypedDict, List, Dict, Optional, Annotated


class ChunkInfo(TypedDict):
    """Metadata for a single document chunk (no full content)."""
    chunk_id: str        # ES _id — pass to read_chunk()
    chunk_index: int     # Position within document
    section_title: str   # Detected section heading (may be empty)
    description: str     # TLDR of this chunk (first ~150 chars)
    page_num: int
    token_count: int


class DocInfo(TypedDict):
    """Information about a loaded document."""
    file_id: str
    file_name: str
    title: str
    file_urls: str
    total_tokens: int
    chunks: Dict[str, ChunkInfo]   # chunk_id -> ChunkInfo
    loaded_chunks: Dict[str, str]  # chunk_id -> full content (cache)


class AgentState(TypedDict):
    """Full state for the RAG ReAct agent."""

    # User context
    user_id: str

    # Documents being tracked (file_id -> DocInfo)
    docs: Dict[str, DocInfo]

    # Conversation history (accumulated via operator.add)
    messages: Annotated[List[Dict], operator.add]

    # Current question and last LLM response
    question: str
    response: str

    # Status history: ["planning", "tool_call", "answer", ...]
    status: List[str]

    # Loop counters
    round: int
    num_llm_calls_available: int
    start_time: float

    # Final output
    prediction: str
    termination: str

    # Caches to avoid redundant ES fetches
    chunk_cache: Dict[str, str]    # chunk_id -> full content
    search_results_cache: List[Dict]
