# encoding: utf-8
"""
DeepRAGRetrievalTool
====================
Drop-in replacement for the legacy RAGRetrievalTool that uses the deeprag_sdk
progressive-reading Agent internally instead of a one-shot HTTP API call.

Key differences from the original RAGRetrievalTool:
  - Inner retrieval uses a ReAct agent: search → TLDR scan → targeted chunk reads
  - Returns actual source segments (chunk text + metadata) rather than a
    synthesised answer, so the outer LLM can do its own reasoning.
  - Supports cross-knowledge-base queries by merging file_ids from multiple
    robot_ids and passing them as a hard scope restriction to the agent.

Usage example
-------------
    from elasticsearch import Elasticsearch
    from deeprag_sdk import Reader
    from deeprag_sdk.agent import Agent
    from deeprag_sdk.rag_tool import DeepRAGRetrievalTool

    es     = Elasticsearch("http://localhost:9200")
    reader = Reader(es, index_name="my_docs")
    agent  = Agent(api_key="sk-...", reader=reader, model="gpt-4o")

    robot_id_file_id_dict = {
        "robot_abc": ["file1", "file2"],
        "robot_def": [],          # empty list = all files for this robot
    }
    knowledge_base_info = {
        "robot_abc": {"description": "2023 annual reports"},
        "robot_def": {"description": "Internal HR policies"},
    }

    tool = DeepRAGRetrievalTool(
        agent=agent,
        robot_id_file_id_dict=robot_id_file_id_dict,
        knowledge_base_info=knowledge_base_info,
    )

    import asyncio
    result = asyncio.run(tool.execute("平安银行2023年营业收入"))
    print(result)
"""

import json
import traceback
from typing import Dict, List, Optional

from scienceclaw.agent.tools.base import Tool

from .agent.agent import Agent


class DeepRAGRetrievalTool(Tool):
    """Search knowledge bases using progressive deep reading.

    Instead of a single-pass embedding lookup, uses a ReAct agent that
    progressively narrows from document summaries down to targeted chunk reads,
    following the same token-budget discipline as DeepXiv.

    The tool returns a JSON list of precise source segments so the outer LLM
    receives raw evidence rather than a pre-synthesised string.
    """

    name     = "rag_retrieval"
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    # Dynamic description — generated from knowledge_base_info at call time.
    @property
    def description(self) -> str:
        base = (
            "Search user-uploaded documents in the knowledge base to retrieve "
            "relevant content using progressive deep reading. "
        )
        if getattr(self, "knowledge_base_info", None):
            base += "\n\nAvailable knowledge bases:\n"
            for kb_id, info in self.knowledge_base_info.items():
                base += f"- {kb_id}: {info.get('description', 'No description')}\n"
            base += (
                "\nSpecify 'robot_ids' to search specific knowledge bases, "
                "or leave empty to search all."
            )
        return base

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type":        "string",
                "description": "Search query to find information in uploaded documents",
            },
            "robot_ids": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": (
                    "Optional: specific knowledge-base IDs to search. "
                    "Leave empty to search all available bases."
                ),
                "default": None,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        agent: Agent,
        robot_id_file_id_dict: Dict[str, List[str]],
        knowledge_base_info: Optional[Dict[str, Dict]] = None,
        request_id: str = "",
        lang: str = "Chinese",
        tenantid: Optional[str] = None,
    ):
        """
        Args:
            agent:
                A configured deeprag_sdk.Agent instance (holds ES reader + LLM client).
            robot_id_file_id_dict:
                Maps robot_id (= ES session_id) to the list of file_ids that belong
                to that knowledge base.  An empty list means "all files for this robot".
            knowledge_base_info:
                Optional metadata for the dynamic tool description.
                Keyed by robot_id; each value is a dict with at least a 'description' key.
            request_id:
                Caller-supplied trace ID for logging.
            lang:
                Language hint (currently informational only).
            tenantid:
                Tenant identifier (currently informational only).
        """
        self.agent                 = agent
        self.robot_id_file_id_dict = robot_id_file_id_dict
        self.knowledge_base_info   = knowledge_base_info or {}
        self.request_id            = request_id
        self.lang                  = lang
        self.tenantid              = tenantid

    async def execute(self, query: str, robot_ids: Optional[List[str]] = None) -> str:
        """Search knowledge bases with the progressive-reading agent.

        Args:
            query:     Natural language search query.
            robot_ids: Optional list of robot_ids (knowledge-base IDs) to restrict
                       the search.  Pass None or [] to search all configured bases.

        Returns:
            JSON string: a list of source-segment objects, each with:
              - file_name:     document filename
              - file_id:       ES file identifier
              - page_num:      page number within the source document
              - section_title: section heading (may be empty)
              - chunk:         the retrieved text segment

            If no chunks were read but the agent produced a synthesised answer,
            returns a single-item list with a "summary" key instead.

            On error, returns a JSON object with an "error" key.
        """
        # ------------------------------------------------------------------
        # 1. Determine which robot_ids (knowledge bases) to search
        # ------------------------------------------------------------------
        if robot_ids:
            selected = {k: v for k, v in self.robot_id_file_id_dict.items() if k in robot_ids}
            if not selected:
                return json.dumps(
                    {
                        "error": f"No valid knowledge bases found for IDs: {robot_ids}",
                        "available_bases": list(self.robot_id_file_id_dict.keys()),
                    },
                    ensure_ascii=False,
                )
        else:
            selected = self.robot_id_file_id_dict

        # ------------------------------------------------------------------
        # 2. Map robot_ids → session_id + allowed_file_ids
        #
        # robot_id IS the ES session_id (confirmed by caller).
        #
        # Single robot_id:
        #   Use it directly as session_id so ES can filter efficiently.
        #   If it has explicit file_ids, also restrict by those.
        #
        # Multiple robot_ids:
        #   We can't filter by a single session_id, so we rely solely on
        #   file_ids.  If every selected robot has explicit file_ids we can
        #   merge them; if any robot's list is empty (= "all files"), we
        #   cannot safely restrict by file_id and leave the filter open.
        # ------------------------------------------------------------------
        selected_robots = list(selected.keys())

        if len(selected_robots) == 1:
            session_id   = selected_robots[0]
            explicit_ids = selected[session_id]
            file_ids     = explicit_ids if explicit_ids else None
        else:
            session_id   = ""                           # no single-session filter
            merged: List[str] = []
            has_open_robot    = False
            for robot_id, fids in selected.items():
                if fids:
                    merged.extend(fids)
                else:
                    has_open_robot = True   # this robot has no explicit file restriction

            # If every robot has explicit file_ids, use the merged list.
            # If any robot is unrestricted, we cannot safely limit by file_id.
            file_ids = merged if (merged and not has_open_robot) else None

        # ------------------------------------------------------------------
        # 3. Run the progressive-reading agent
        # ------------------------------------------------------------------
        try:
            answer, sources = self.agent.query_with_sources(
                question=query,
                session_id=session_id,
                file_ids=file_ids,
            )
        except Exception:
            return json.dumps(
                {
                    "error": "Agent retrieval failed",
                    "detail": traceback.format_exc(),
                    "request_id": self.request_id,
                },
                ensure_ascii=False,
            )

        # ------------------------------------------------------------------
        # 4. Format source segments for the outer LLM
        # ------------------------------------------------------------------
        results = [
            {
                "file_name":     src.get("file_name", ""),
                "file_id":       src.get("file_id", ""),
                "page_num":      src.get("page_num", 0),
                "section_title": src.get("section_title", ""),
                "chunk":         src.get("content", ""),
            }
            for src in sources
            if src.get("content")   # skip empty reads
        ]

        # Deduplicate by chunk_id while preserving first-seen order
        seen: set = set()
        deduped = []
        for src, res in zip(sources, results):
            cid = src.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                deduped.append(res)

        # If no chunks were read but the agent produced a narrative answer,
        # surface it as a fallback so the outer LLM still gets something.
        if not deduped and answer:
            deduped.append({"summary": answer})

        return json.dumps(deduped, ensure_ascii=False)
