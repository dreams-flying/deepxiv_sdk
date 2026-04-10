"""
Main Agent class for RAG document retrieval.

Usage:
    from elasticsearch import Elasticsearch
    from FlagEmbedding import BGEM3FlagModel
    from deeprag_sdk import Reader
    from deeprag_sdk.agent import Agent

    es = Elasticsearch("http://localhost:9200")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    reader = Reader(es, index_name="my_docs", embed_model=model)

    agent = Agent(
        api_key="sk-...",
        reader=reader,
        model="gpt-4o",
        print_process=True,
    )

    answer = agent.query(
        question="2023年平安银行营业收入是多少？",
        user_id="user_001",
    )
    print(answer)
"""

import time
from typing import Dict, Optional, Any
from openai import OpenAI

from ..reader import Reader
from .graph import create_react_graph, create_initial_state
from .tools import ToolExecutor


class Agent:
    """
    ReAct agent for internal document retrieval.

    The agent follows a progressive disclosure strategy:
      1. search_docs  → scan descriptions (low cost)
      2. load_doc     → inspect chunk structure (low cost)
      3. read_chunk   → read targeted chunks (medium cost)
      4. get_full_doc → read entire document (high cost, last resort)

    Supports multi-turn queries within the same session.
    Documents loaded in one query are cached and reused in subsequent queries.
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
            reader:           Reader instance (connected to your ES index).
            model:            LLM model name (e.g. "gpt-4o", "deepseek-chat").
            base_url:         Override API base URL for alternative providers
                              (e.g. DeepSeek: "https://api.deepseek.com/v1").
            max_llm_calls:    Maximum LLM calls per query (default 20).
            max_time_seconds: Hard timeout per query in seconds (default 300).
            max_tokens:       Max tokens per LLM response (default 4096).
            temperature:      Sampling temperature (default 0.7).
            print_process:    Print tool calls and LLM responses to stdout.
            stream:           Use streaming mode for LLM output.
        """
        self.reader = reader
        self.model = model
        self.max_llm_calls = max_llm_calls
        self.max_time_seconds = max_time_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.print_process = print_process
        self.stream = stream

        # OpenAI-compatible client
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # Compile the LangGraph ReAct workflow
        self._graph = create_react_graph()

        # Persistent document cache across queries (per-user if multi-user)
        # Maps user_id -> {file_id -> DocInfo}
        self._persistent_docs: Dict[str, Dict] = {}

    def query(
        self,
        question: str,
        user_id: str = "",
        reset_docs: bool = False,
    ) -> str:
        """
        Answer a question using the document retrieval agent.

        Args:
            question:   The user's question in natural language.
            user_id:    User identifier for multi-tenant ES filtering.
                        Also used to scope the persistent document cache.
            reset_docs: Clear the document cache for this user before querying.

        Returns:
            The agent's final answer as a string.
        """
        if reset_docs:
            self._persistent_docs.pop(user_id, None)

        # Retrieve cached docs for this user
        persistent = self._persistent_docs.setdefault(user_id, {})

        # Build initial state
        state = create_initial_state(user_id=user_id, docs=dict(persistent))
        state["question"] = question

        # Create a fresh tool executor bound to this reader
        tool_executor = ToolExecutor(reader=self.reader)

        # LangGraph runnable config
        config = {
            "configurable": {
                "client": self.client,
                "model_name": self.model,
                "max_llm_calls": self.max_llm_calls,
                "max_time_seconds": self.max_time_seconds,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "print_process": self.print_process,
                "stream": self.stream,
                "tool_executor": tool_executor,
            }
        }

        if self.print_process:
            print(f"\n{'='*60}")
            print(f"[Agent] 问题：{question}")
            print(f"[Agent] 用户：{user_id or '(未设置)'}")
            print(f"{'='*60}\n")

        # Run the graph
        final_state = self._graph.invoke(state, config=config)

        # Persist newly loaded documents back to the cache
        self._persistent_docs[user_id].update(final_state.get("docs", {}))

        prediction = final_state.get("prediction", "")
        termination = final_state.get("termination", "unknown")

        if self.print_process:
            print(f"\n[Agent] 终止原因：{termination}")
            print(f"[Agent] 轮次：{final_state.get('round', 0)}")

        return prediction

    def get_loaded_docs(self, user_id: str = "") -> Dict:
        """Return currently cached documents for a user."""
        return dict(self._persistent_docs.get(user_id, {}))

    def reset_docs(self, user_id: str = "") -> None:
        """Clear the document cache for a user."""
        self._persistent_docs.pop(user_id, None)

    def add_doc(self, file_id: str, user_id: str = "") -> bool:
        """
        Pre-load a document's structure into the cache.
        Useful for pinning reference documents across multiple queries.

        Returns True if successfully loaded.
        """
        try:
            head = self.reader.head(file_id=file_id, user_id=user_id)
            if not head or not head.get("chunks"):
                return False

            chunks_dict = {
                c["chunk_id"]: {
                    "chunk_id": c["chunk_id"],
                    "chunk_index": c["chunk_index"],
                    "section_title": c["section_title"],
                    "description": c["description"],
                    "page_num": c["page_num"],
                    "token_count": c["token_count"],
                }
                for c in head["chunks"]
            }

            self._persistent_docs.setdefault(user_id, {})[file_id] = {
                "file_id": file_id,
                "file_name": head.get("file_name", ""),
                "title": head.get("title", ""),
                "file_urls": head.get("file_urls", ""),
                "total_tokens": head.get("total_tokens", 0),
                "chunks": chunks_dict,
                "loaded_chunks": {},
            }
            return True
        except Exception:
            return False
