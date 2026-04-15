# encoding: utf-8
"""
Progressive RAG retrieval tools for the scienceclaw agent loop.

Place these two files at:
    scienceclaw/agent/tools/custom/deep_rag_reader.py      (Reader class)
    scienceclaw/agent/tools/custom/deep_rag_retrieval.py   (this file)

Provides 6 Tool subclasses that mirror deeprag_sdk's progressive disclosure
pattern but contain no deeprag_sdk dependency:

  SearchDocsTool    search_docs      Hybrid BM25+kNN, scoped by robot_id
  QuickPreviewTool  quick_preview    Concurrent first-2000-char previews
  LoadDocTool       load_doc         Chunk structure with 60-char TLDRs
  ReadChunkTool     read_chunk       Full chunk text by chunk_id
  BrowseFullDocTool browse_full_doc  Full doc split into navigable parts
  ReadDocPartTool   read_doc_part    Read one specific part (~4000 chars)

All 6 tools share a single DocStore instance created by create_deep_rag_tools().
The factory is the intended entry point — it wires the Reader, DocStore, and
robot_id mapping together in one call.

Usage
-----
    from elasticsearch import Elasticsearch
    from scienceclaw.agent.tools.custom.deep_rag_reader import Reader
    from scienceclaw.agent.tools.custom.deep_rag_retrieval import create_deep_rag_tools

    es = Elasticsearch("http://localhost:9200")
    reader = Reader(es, index_name="my_docs")

    tools = create_deep_rag_tools(
        reader=reader,
        robot_id_file_id_dict={
            "robot_abc": ["file1", "file2"],   # restrict to specific files
            "robot_def": [],                   # empty = all files for this robot
        },
        knowledge_base_info={
            "robot_abc": {"description": "2023 annual reports"},
            "robot_def": {"description": "HR policy documents"},
        },
        request_id="req-001",
    )

    # Pass tools to your agent loop alongside other tools
    agent_loop.run(query="平安银行2023年营业收入", tools=tools)

Progressive reading order (cheapest → most expensive)
------------------------------------------------------
  1. search_docs      — document-level scan; costs one embedding call
  2. quick_preview    — concurrent first-2000-char reads; costs N ES queries
  3. load_doc         — chunk structure; costs one ES query per document
  4. read_chunk       — full chunk text; costs one ES query per chunk
  5. browse_full_doc  — fetches entire document; use only as last resort
  6. read_doc_part    — reads from in-memory cache; zero ES cost
"""

import json
import traceback
from typing import Dict, List, Optional

from scienceclaw.agent.tools.base import Tool

from .deep_rag_reader import Reader

# Characters per part when splitting a full document in BrowseFullDocTool
PART_SIZE = 4000


# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory state
# ─────────────────────────────────────────────────────────────────────────────

class DocStore:
    """In-memory cache shared across all progressive reading tools.

    Created once by create_deep_rag_tools() and injected into every tool
    that needs cross-call state (LoadDocTool, ReadChunkTool, BrowseFullDocTool,
    ReadDocPartTool).

    Attributes:
        docs:
            file_id → {
                "file_id", "file_name", "file_urls", "total_tokens",
                "chunks":        {chunk_id → {chunk_id, section_title, description, page_num, token_count}},
                "loaded_chunks": {chunk_id → full_content_string},
            }
        chunk_cache:
            chunk_id → full content string.
            Populated by ReadChunkTool so repeated reads skip ES.
        full_doc_cache:
            file_id → {"file_name": str, "content": str}.
            Populated by BrowseFullDocTool so ReadDocPartTool needs zero ES calls.
    """

    def __init__(self):
        self.docs: Dict[str, Dict] = {}
        self.chunk_cache: Dict[str, str] = {}
        self.full_doc_cache: Dict[str, Dict] = {}

    def reset(self):
        """Clear all caches — call between independent sessions if reusing tools."""
        self.docs.clear()
        self.chunk_cache.clear()
        self.full_doc_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — SearchDocsTool
# ─────────────────────────────────────────────────────────────────────────────

class SearchDocsTool(Tool):
    """Search knowledge bases using hybrid retrieval (BM25 + kNN).

    Returns document-level summaries — one result per document, not per chunk.
    Read the 'content' field (document abstract) to judge relevance quickly
    before calling load_doc or read_chunk for targeted reads.

    Dynamic description: lists available knowledge bases from knowledge_base_info
    so the outer agent can choose the right robot_ids for each query.
    """

    name     = "search_docs"
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    @property
    def description(self) -> str:
        desc = (
            "Search user-uploaded documents using hybrid retrieval (BM25 + vector). "
            "Returns document-level summaries — read the 'content' field to judge "
            "relevance, then call load_doc or read_chunk for targeted reads. "
        )
        if self.knowledge_base_info:
            desc += "\n\nAvailable knowledge bases:\n"
            for kb_id, info in self.knowledge_base_info.items():
                desc += f"- {kb_id}: {info.get('description', 'No description')}\n"
            desc += (
                "\nSpecify 'robot_ids' to search specific knowledge bases, "
                "or omit to search all."
            )
        return desc

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type":        "string",
                "description": "Natural language query to search for in the knowledge base",
            },
            "robot_ids": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": (
                    "Optional: specific knowledge-base IDs to search. "
                    "Omit or pass null to search all configured bases."
                ),
                "default": None,
            },
            "size": {
                "type":        "integer",
                "description": "Number of results to return (default 10, max 50)",
                "default":     10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        reader: Reader,
        doc_store: DocStore,
        robot_id_file_id_dict: Dict[str, List[str]],
        knowledge_base_info: Optional[Dict[str, Dict]] = None,
        request_id: str = "",
    ):
        """
        Args:
            reader:                 ES-backed Reader instance.
            doc_store:              Shared DocStore (created by factory).
            robot_id_file_id_dict:  Maps robot_id → list of file_ids.
                                    Empty list means "all files for this robot".
            knowledge_base_info:    Optional per-robot description dict for dynamic desc.
            request_id:             Trace ID forwarded to error messages.
        """
        self.reader                = reader
        self.doc_store             = doc_store
        self.robot_id_file_id_dict = robot_id_file_id_dict
        self.knowledge_base_info   = knowledge_base_info or {}
        self.request_id            = request_id

    async def execute(
        self,
        query: str,
        robot_ids: Optional[List[str]] = None,
        size: int = 10,
    ) -> str:
        """Search and return document summaries scoped to the selected knowledge bases.

        Returns:
            JSON string — {"total": int, "results": [...]} on success,
            or {"error": ..., "available_bases": [...]} on bad robot_ids.
        """
        # 1. Select which robot_ids to search
        if robot_ids:
            selected = {k: v for k, v in self.robot_id_file_id_dict.items() if k in robot_ids}
            if not selected:
                return json.dumps({
                    "error":           f"No valid knowledge bases found for IDs: {robot_ids}",
                    "available_bases": list(self.robot_id_file_id_dict.keys()),
                }, ensure_ascii=False)
        else:
            selected = self.robot_id_file_id_dict

        selected_robots = list(selected.keys())

        # 2. Map robot_ids → ES session_id + file_ids filter
        #
        # Single robot: use it directly as session_id for efficient ES filtering.
        # Multiple robots: rely on merged file_ids; if any robot is unrestricted
        # (empty list), we cannot safely restrict by file_id and leave filter open.
        if len(selected_robots) == 1:
            session_id   = selected_robots[0]
            explicit_ids = selected[session_id]
            file_ids     = explicit_ids if explicit_ids else None
        else:
            session_id       = ""
            merged: List[str] = []
            has_open_robot   = False
            for rid, fids in selected.items():
                if fids:
                    merged.extend(fids)
                else:
                    has_open_robot = True
            file_ids = merged if (merged and not has_open_robot) else None

        # 3. Execute hybrid search
        try:
            results = self.reader.search(
                query=query,
                session_id=session_id or None,
                size=min(size, 50),
                file_ids=file_ids,
            )
            return json.dumps(results, ensure_ascii=False)
        except Exception:
            return json.dumps({
                "error":      "Search failed",
                "detail":     traceback.format_exc(),
                "request_id": self.request_id,
            }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — QuickPreviewTool
# ─────────────────────────────────────────────────────────────────────────────

class QuickPreviewTool(Tool):
    """Fetch the first 2000 characters of multiple documents concurrently.

    Use after search_docs to quickly scan several candidates and decide which
    ones deserve a full load_doc + read_chunk cycle.  Much faster than calling
    a single-document preview in a loop.
    """

    name        = "quick_preview"
    description = (
        "Fetch the first 2000 characters of multiple documents concurrently. "
        "Pass a list of file_ids from search_docs results. "
        "Use this to quickly scan candidates before committing to load_doc + read_chunk."
    )
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    parameters = {
        "type": "object",
        "properties": {
            "file_ids": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": "List of file IDs to preview (from search_docs results)",
            },
            "session_id": {
                "type":        "string",
                "description": "Optional: robot_id to scope the ES query",
                "default":     "",
            },
        },
        "required": ["file_ids"],
    }

    def __init__(self, reader: Reader, request_id: str = ""):
        self.reader     = reader
        self.request_id = request_id

    async def execute(self, file_ids: List[str], session_id: str = "") -> str:
        """Return first 2000 chars of each document.

        Returns:
            JSON string — list of {"file_id", "file_name", "preview", "truncated"}.
        """
        if not file_ids:
            return json.dumps({"error": "file_ids list is empty."}, ensure_ascii=False)
        try:
            previews = self.reader.quick_preview(
                file_ids=file_ids,
                session_id=session_id or None,
                max_chars=2000,
            )
            return json.dumps(previews, ensure_ascii=False)
        except Exception:
            return json.dumps({
                "error":      "Quick preview failed",
                "detail":     traceback.format_exc(),
                "request_id": self.request_id,
            }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — LoadDocTool
# ─────────────────────────────────────────────────────────────────────────────

class LoadDocTool(Tool):
    """Load a document's chunk structure: every chunk's section title,
    60-char TLDR description, page number, and character count.

    Does NOT return full chunk text.  Inspect the descriptions to decide
    which chunks are worth reading, then call read_chunk(chunk_id).

    Results are cached in DocStore — repeated calls for the same file_id
    return immediately without hitting ES.
    """

    name        = "load_doc"
    description = (
        "Load a document's chunk structure: chunk IDs, section titles, "
        "60-char TLDRs, page numbers, and character counts. "
        "Call this after search_docs or quick_preview to identify relevant chunks. "
        "Then call read_chunk(chunk_id) to read full content."
    )
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type":        "string",
                "description": "Document file ID (from search_docs results)",
            },
            "session_id": {
                "type":        "string",
                "description": "Optional: robot_id to scope the ES query",
                "default":     "",
            },
        },
        "required": ["file_id"],
    }

    def __init__(self, reader: Reader, doc_store: DocStore, request_id: str = ""):
        self.reader     = reader
        self.doc_store  = doc_store
        self.request_id = request_id

    async def execute(self, file_id: str, session_id: str = "") -> str:
        """Load chunk structure and cache it for subsequent read_chunk calls.

        Returns:
            JSON string — {
              "file_id", "file_name", "total_chunks", "total_tokens",
              "chunks": [{"chunk_id", "section_title", "description", "page_num", "token_count"}]
            }
        """
        # Return from cache if already loaded
        if file_id in self.doc_store.docs:
            cached = self.doc_store.docs[file_id]
            return json.dumps({
                "file_id":      file_id,
                "file_name":    cached.get("file_name", ""),
                "total_chunks": len(cached.get("chunks", {})),
                "total_tokens": cached.get("total_tokens", 0),
                "chunks":       list(cached.get("chunks", {}).values()),
                "_cached":      True,
            }, ensure_ascii=False)

        try:
            head = self.reader.head(file_id=file_id, session_id=session_id or None)
            if not head or not head.get("chunks"):
                return json.dumps({
                    "error": f"Document '{file_id}' not found or has no chunks.",
                }, ensure_ascii=False)

            # Cache in DocStore for read_chunk lookups
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
            self.doc_store.docs[file_id] = {
                "file_id":       file_id,
                "file_name":     head.get("file_name", ""),
                "file_urls":     head.get("file_urls", ""),
                "total_tokens":  head.get("total_tokens", 0),
                "chunks":        chunks_dict,
                "loaded_chunks": {},
            }

            return json.dumps({
                "file_id":      file_id,
                "file_name":    head.get("file_name", ""),
                "total_chunks": head.get("total_chunks", 0),
                "total_tokens": head.get("total_tokens", 0),
                "chunks":       head.get("chunks", []),
            }, ensure_ascii=False)

        except Exception:
            return json.dumps({
                "error":      f"Failed to load document '{file_id}'",
                "detail":     traceback.format_exc(),
                "request_id": self.request_id,
            }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — ReadChunkTool
# ─────────────────────────────────────────────────────────────────────────────

class ReadChunkTool(Tool):
    """Read the full text of a specific chunk by its chunk_id.

    This is the primary way to read document content in detail.
    Always inspect the chunk's 60-char 'description' from load_doc before
    calling this — only read chunks whose description suggests relevance.

    Results are cached in DocStore — repeated reads skip ES entirely.
    """

    name        = "read_chunk"
    description = (
        "Read the full text of a specific chunk by its chunk_id. "
        "Get chunk_ids from search_docs or load_doc results. "
        "Inspect the 60-char description from load_doc before reading — "
        "only read chunks likely to contain the answer."
    )
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    parameters = {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type":        "string",
                "description": "Chunk ES _id (from search_docs or load_doc results)",
            },
        },
        "required": ["chunk_id"],
    }

    def __init__(self, reader: Reader, doc_store: DocStore, request_id: str = ""):
        self.reader     = reader
        self.doc_store  = doc_store
        self.request_id = request_id

    async def execute(self, chunk_id: str) -> str:
        """Fetch and return the full content of a chunk.

        Returns:
            JSON string — {
              "chunk_id", "file_id", "file_name",
              "section_title", "content", "page_num", "token_count"
            }
        """
        # Return from cache if already read
        if chunk_id in self.doc_store.chunk_cache:
            cached_content = self.doc_store.chunk_cache[chunk_id]
            file_id, file_name, page_num, section_title = "", "", 0, ""
            for fid, doc in self.doc_store.docs.items():
                if chunk_id in doc.get("chunks", {}):
                    file_id       = fid
                    file_name     = doc.get("file_name", "")
                    meta          = doc["chunks"][chunk_id]
                    page_num      = meta.get("page_num", 0)
                    section_title = meta.get("section_title", "")
                    break
            return json.dumps({
                "chunk_id":      chunk_id,
                "file_id":       file_id,
                "file_name":     file_name,
                "section_title": section_title,
                "content":       cached_content,
                "page_num":      page_num,
                "token_count":   len(cached_content),
                "_cached":       True,
            }, ensure_ascii=False)

        try:
            result = self.reader.read_chunk(chunk_id=chunk_id)
            if not result or not result.get("content"):
                return json.dumps({
                    "error": f"Chunk '{chunk_id}' not found or has no content.",
                }, ensure_ascii=False)

            content = result["content"]
            file_id = result.get("file_id", "")

            # Cache content for future calls
            self.doc_store.chunk_cache[chunk_id] = content
            # Update loaded_chunks in doc cache if this doc was previously loaded
            if file_id and file_id in self.doc_store.docs:
                self.doc_store.docs[file_id]["loaded_chunks"][chunk_id] = content

            return json.dumps({
                "chunk_id":      chunk_id,
                "file_id":       file_id,
                "file_name":     result.get("file_name", ""),
                "section_title": result.get("section_title", ""),
                "content":       content,
                "page_num":      result.get("page_num", 0),
                "token_count":   result.get("token_count", len(content)),
            }, ensure_ascii=False)

        except Exception:
            return json.dumps({
                "error":      f"Failed to read chunk '{chunk_id}'",
                "detail":     traceback.format_exc(),
                "request_id": self.request_id,
            }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — BrowseFullDocTool
# ─────────────────────────────────────────────────────────────────────────────

class BrowseFullDocTool(Tool):
    """Fetch the complete document and show a navigable table of contents.

    The full document is split into numbered parts (PART_SIZE chars each).
    Only the first 150 chars of each part are shown — the agent picks which
    part(s) to read via read_doc_part, avoiding context overflow.

    The full text is cached in DocStore after the first fetch — subsequent
    browse_full_doc or read_doc_part calls for the same file_id cost nothing.

    Use only when search_docs + load_doc + read_chunk cannot find the answer.
    """

    name        = "browse_full_doc"
    description = (
        "Fetch the complete document and display a navigable table of contents. "
        "The document is split into numbered parts (~4000 chars each); only the "
        "first 150 chars of each part are shown. "
        "Then call read_doc_part(file_id, part_index) to read a specific part. "
        "WARNING: do not use as the first tool — try search_docs and read_chunk first."
    )
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type":        "string",
                "description": "Document file ID",
            },
            "session_id": {
                "type":        "string",
                "description": "Optional: robot_id to scope the ES query",
                "default":     "",
            },
        },
        "required": ["file_id"],
    }

    def __init__(
        self,
        reader: Reader,
        doc_store: DocStore,
        request_id: str = "",
        part_size: int = PART_SIZE,
    ):
        self.reader     = reader
        self.doc_store  = doc_store
        self.request_id = request_id
        self.part_size  = part_size

    async def execute(self, file_id: str, session_id: str = "") -> str:
        """Fetch full document and return the parts table of contents.

        Returns:
            JSON string — {
              "file_id", "file_name", "total_chars", "n_parts", "part_size",
              "parts": [{"index", "start", "end", "preview"}]
            }
        """
        # Fetch and cache once per file_id
        if file_id not in self.doc_store.full_doc_cache:
            try:
                result = self.reader.raw(file_id=file_id, session_id=session_id or None)
                if not result or not result.get("content"):
                    return json.dumps({
                        "error": f"Document '{file_id}' not found or has no content.",
                    }, ensure_ascii=False)
                self.doc_store.full_doc_cache[file_id] = {
                    "content":   result["content"],
                    "file_name": result.get("file_name", file_id),
                }
            except Exception:
                return json.dumps({
                    "error":      f"Failed to fetch document '{file_id}'",
                    "detail":     traceback.format_exc(),
                    "request_id": self.request_id,
                }, ensure_ascii=False)

        cached    = self.doc_store.full_doc_cache[file_id]
        content   = cached["content"]
        file_name = cached["file_name"]
        total     = len(content)
        n_parts   = (total + self.part_size - 1) // self.part_size

        parts = []
        for i in range(n_parts):
            start   = i * self.part_size
            end     = min(start + self.part_size, total)
            preview = content[start:start + 150].replace("\n", " ").strip()
            parts.append({"index": i, "start": start, "end": end, "preview": preview})

        return json.dumps({
            "file_id":     file_id,
            "file_name":   file_name,
            "total_chars": total,
            "n_parts":     n_parts,
            "part_size":   self.part_size,
            "parts":       parts,
        }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — ReadDocPartTool
# ─────────────────────────────────────────────────────────────────────────────

class ReadDocPartTool(Tool):
    """Read one specific part of a full document by part index.

    browse_full_doc must be called first — it fetches the document and
    populates DocStore.full_doc_cache.  This tool reads from that cache,
    so it costs zero ES queries.
    """

    name        = "read_doc_part"
    description = (
        "Read one specific part of a full document (identified by part_index). "
        "Must call browse_full_doc first to load the document and see part indices. "
        "Each part is ~4000 chars — readable without context overflow."
    )
    tags:     list[str] = ["retrieval", "knowledge base"]
    category: str       = "custom"

    parameters = {
        "type": "object",
        "properties": {
            "file_id": {
                "type":        "string",
                "description": "Document file ID (must have called browse_full_doc first)",
            },
            "part_index": {
                "type":        "integer",
                "description": "Zero-based part index shown in browse_full_doc output",
            },
        },
        "required": ["file_id", "part_index"],
    }

    def __init__(self, doc_store: DocStore, request_id: str = "", part_size: int = PART_SIZE):
        self.doc_store  = doc_store
        self.request_id = request_id
        self.part_size  = part_size

    async def execute(self, file_id: str, part_index: int) -> str:
        """Return the full text of one document part from cache.

        Returns:
            JSON string — {
              "file_id", "file_name", "part_index", "total_parts",
              "start", "end", "total_chars", "content", "has_next"
            }
        """
        if file_id not in self.doc_store.full_doc_cache:
            return json.dumps({
                "error": (
                    f"Document '{file_id}' is not loaded. "
                    "Call browse_full_doc(file_id) first."
                ),
            }, ensure_ascii=False)

        cached    = self.doc_store.full_doc_cache[file_id]
        content   = cached["content"]
        file_name = cached["file_name"]
        total     = len(content)
        n_parts   = (total + self.part_size - 1) // self.part_size

        if part_index < 0 or part_index >= n_parts:
            return json.dumps({
                "error": (
                    f"part_index {part_index} is out of range. "
                    f"Valid range: 0 – {n_parts - 1}."
                ),
            }, ensure_ascii=False)

        start        = part_index * self.part_size
        end          = min(start + self.part_size, total)
        part_content = content[start:end]

        return json.dumps({
            "file_id":     file_id,
            "file_name":   file_name,
            "part_index":  part_index,
            "total_parts": n_parts,
            "start":       start,
            "end":         end,
            "total_chars": total,
            "content":     part_content,
            "has_next":    end < total,
        }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_deep_rag_tools(
    reader: Reader,
    robot_id_file_id_dict: Dict[str, List[str]],
    knowledge_base_info: Optional[Dict[str, Dict]] = None,
    request_id: str = "",
    part_size: int = PART_SIZE,
) -> List[Tool]:
    """Create the full set of progressive reading tools with shared DocStore.

    This is the intended entry point.  All 6 returned tools share one DocStore
    instance so that caches built by LoadDocTool are available to ReadChunkTool,
    and documents fetched by BrowseFullDocTool are available to ReadDocPartTool.

    Args:
        reader:
            A configured Reader instance backed by Elasticsearch.
        robot_id_file_id_dict:
            Maps robot_id (= ES session_id / knowledge base ID) to the list of
            file_ids belonging to that base.  Empty list means "all files".
        knowledge_base_info:
            Optional per-robot metadata for SearchDocsTool's dynamic description.
            Format: {robot_id: {"description": "..."}, ...}
        request_id:
            Caller-supplied trace ID forwarded to all tool error messages.
        part_size:
            Characters per part for BrowseFullDocTool / ReadDocPartTool.
            Default is 4000.

    Returns:
        [
            SearchDocsTool,    # name="search_docs"
            QuickPreviewTool,  # name="quick_preview"
            LoadDocTool,       # name="load_doc"
            ReadChunkTool,     # name="read_chunk"
            BrowseFullDocTool, # name="browse_full_doc"
            ReadDocPartTool,   # name="read_doc_part"
        ]
    """
    doc_store = DocStore()

    return [
        SearchDocsTool(
            reader=reader,
            doc_store=doc_store,
            robot_id_file_id_dict=robot_id_file_id_dict,
            knowledge_base_info=knowledge_base_info,
            request_id=request_id,
        ),
        QuickPreviewTool(
            reader=reader,
            request_id=request_id,
        ),
        LoadDocTool(
            reader=reader,
            doc_store=doc_store,
            request_id=request_id,
        ),
        ReadChunkTool(
            reader=reader,
            doc_store=doc_store,
            request_id=request_id,
        ),
        BrowseFullDocTool(
            reader=reader,
            doc_store=doc_store,
            request_id=request_id,
            part_size=part_size,
        ),
        ReadDocPartTool(
            doc_store=doc_store,
            request_id=request_id,
            part_size=part_size,
        ),
    ]
