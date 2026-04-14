"""
Main Agent class for RAG document retrieval.

Usage:
    from elasticsearch import Elasticsearch
    from deeprag_sdk import Reader
    from deeprag_sdk.agent import Agent

    es     = Elasticsearch("http://localhost:9200")
    reader = Reader(es, index_name="my_docs")

    agent = Agent(
        api_key="sk-...",
        reader=reader,
        model="gpt-4o",
        print_process=True,
    )

    answer = agent.query(
        question="2023年平安银行营业收入是多少？",
        session_id="user_001",
    )
    print(answer)
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI

from ..reader import Reader
from .graph import create_react_graph, create_initial_state
from .tools import ToolExecutor


class Agent:
    """ReAct agent for internal document retrieval.

    Progressive disclosure strategy enforced through tool ordering:
      1. search_docs   — scan document-level summaries (lowest cost)
      2. quick_preview — batch concurrent document previews
      3. load_doc      — inspect chunk structure + 60-char TLDRs
      4. read_chunk    — read targeted chunks (medium cost)
      5. get_full_doc  — read entire document (highest cost, last resort)

    Document structure loaded in one query is cached and reused in
    subsequent queries within the same session.
    """

    def __init__(
        self,
        api_key: str,
        reader: Reader,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        max_llm_calls: int = 20,
        max_time_seconds: int = 300,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        print_process: bool = False,
        stream: bool = False,
    ):
        """
        Args:
            api_key:          OpenAI-compatible API key.
            reader:           Reader instance connected to your ES index.
            model:            LLM model name (e.g. "gpt-4o", "deepseek-chat").
            base_url:         Override API base URL for alternative providers
                              (e.g. DeepSeek: "https://api.deepseek.com/v1").
            max_llm_calls:    Maximum LLM calls per query (default 20).
            max_time_seconds: Hard timeout per query in seconds (default 300).
            max_tokens:       Max tokens per LLM response (default 4096).
            temperature:      Sampling temperature (default 0.7).
            print_process:    Print tool calls and responses to stdout.
            stream:           Use streaming mode for LLM output.
        """
        self.reader           = reader
        self.model            = model
        self.max_llm_calls    = max_llm_calls
        self.max_time_seconds = max_time_seconds
        self.max_tokens       = max_tokens
        self.temperature      = temperature
        self.print_process    = print_process
        self.stream           = stream

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        self._graph = create_react_graph()

        # Persistent doc cache: session_id -> {file_id -> DocInfo}
        self._persistent_docs: Dict[str, Dict] = {}

    def query(
        self,
        question: str,
        session_id: str = "",
        reset_docs: bool = False,
        file_ids: Optional[List[str]] = None,
    ) -> str:
        """Answer a question using the document retrieval agent.

        Args:
            question:   Natural language question.
            session_id: Session / user identifier for ES filtering and cache scoping.
            reset_docs: Clear the document cache for this session before querying.
            file_ids:   Hard-restrict search to these file IDs only (optional).
                        Useful when the caller knows which documents are in scope.

        Returns:
            The agent's final answer as a string.
        """
        prediction, _ = self._run(
            question=question,
            session_id=session_id,
            reset_docs=reset_docs,
            file_ids=file_ids,
        )
        return prediction

    def query_with_sources(
        self,
        question: str,
        session_id: str = "",
        reset_docs: bool = False,
        file_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict]]:
        """Like query(), but also returns the source chunks the agent read.

        Returns:
            (answer, sources) where sources is a list of dicts:
              {chunk_id, file_id, file_name, page_num, section_title, content}
            The list is ordered by read time (first read first).
        """
        return self._run(
            question=question,
            session_id=session_id,
            reset_docs=reset_docs,
            file_ids=file_ids,
        )

    def _run(
        self,
        question: str,
        session_id: str = "",
        reset_docs: bool = False,
        file_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict]]:
        """Internal runner shared by query() and query_with_sources()."""
        if reset_docs:
            self._persistent_docs.pop(session_id, None)

        persistent = self._persistent_docs.setdefault(session_id, {})
        state = create_initial_state(
            session_id=session_id,
            docs=dict(persistent),
            file_ids=file_ids,
        )
        state["question"] = question

        tool_executor = ToolExecutor(reader=self.reader)

        # recursion_limit counts every node visit, not just LLM calls.
        # Each round visits ~4 nodes (planning -> tool_call -> check_limits -> planning),
        # so budget = max_llm_calls * 4 + safety buffer.
        recursion_limit = self.max_llm_calls * 4 + 10

        config = {
            "recursion_limit": recursion_limit,
            "configurable": {
                "client":           self.client,
                "model_name":       self.model,
                "max_llm_calls":    self.max_llm_calls,
                "max_time_seconds": self.max_time_seconds,
                "max_tokens":       self.max_tokens,
                "temperature":      self.temperature,
                "print_process":    self.print_process,
                "stream":           self.stream,
                "tool_executor":    tool_executor,
            },
        }

        if self.print_process:
            print(f"\n{'='*60}")
            print(f"[Agent] Question : {question}")
            print(f"[Agent] Session  : {session_id or '(none)'}")
            if file_ids:
                print(f"[Agent] FileIDs  : {file_ids}")
            print(f"{'='*60}\n")

        final_state = self._graph.invoke(state, config=config)

        # Persist newly loaded documents for reuse in follow-up queries
        self._persistent_docs[session_id].update(final_state.get("docs", {}))

        prediction  = final_state.get("prediction", "")
        termination = final_state.get("termination", "unknown")
        sources     = final_state.get("read_chunk_results", [])

        if self.print_process:
            print(f"\n[Agent] Termination : {termination}")
            print(f"[Agent] Rounds      : {final_state.get('round', 0)}")
            print(f"[Agent] Sources     : {len(sources)} chunks read")

        return prediction, sources

    def get_loaded_docs(self, session_id: str = "") -> Dict:
        """Return cached document structures for a session."""
        return dict(self._persistent_docs.get(session_id, {}))

    def reset_docs(self, session_id: str = "") -> None:
        """Clear the document cache for a session."""
        self._persistent_docs.pop(session_id, None)

    def add_doc(self, file_id: str, session_id: str = "") -> bool:
        """Pre-load a document's chunk structure into the cache.

        Useful for pinning reference documents that should always be available
        regardless of what search_docs returns.

        Returns True if the document was successfully loaded.
        """
        try:
            head = self.reader.head(file_id=file_id, session_id=session_id)
            if not head or not head.get("chunks"):
                return False

            chunks_dict = {
                c["chunk_id"]: {
                    "chunk_id":      c["chunk_id"],
                    "section_title": c["section_title"],
                    "description":   c["description"],
                    "page_num":      c["page_num"],
                    "token_count":   c["token_count"],
                }
                for c in head["chunks"]
            }

            self._persistent_docs.setdefault(session_id, {})[file_id] = {
                "file_id":       file_id,
                "file_name":     head.get("file_name", ""),
                "file_urls":     head.get("file_urls", ""),
                "total_tokens":  head.get("total_tokens", 0),
                "chunks":        chunks_dict,
                "loaded_chunks": {},
            }
            return True
        except Exception:
            return False
