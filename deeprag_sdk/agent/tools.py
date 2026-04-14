"""
Tools available to the RAG ReAct agent.

Seven tools, ordered by information cost (cheapest first):
  search_docs     -> document-level abstract search; returns content summaries
  quick_preview   -> concurrent first-N-chars preview of multiple documents
  load_doc        -> chunk structure with 60-char descriptions + token counts
  preview_doc     -> first 2000 chars of one document
  read_chunk      -> full content of one chunk (targeted read)
  browse_full_doc -> split full document into parts and show a preview of each
  read_doc_part   -> read one specific part of the full document

The last two tools replace the former get_full_doc tool and prevent LLM
context overflow by never sending the entire document at once:

  browse_full_doc(file_id)
      Fetches the complete document once, caches it in Python memory,
      then returns only a table of contents — each part's index, char range,
      and first 150 chars.  The LLM picks which part(s) to read.

  read_doc_part(file_id, part_index)
      Returns only the chosen part (~PART_SIZE chars).  The LLM receives
      a manageable slice rather than the entire document.

Maps to DeepXiv's tool chain:
  search_papers     -> search_docs
  quick_preview     -> quick_preview   (batch, concurrent)
  load_paper        -> load_doc
  get_paper_preview -> preview_doc
  read_section      -> read_chunk
  get_full_paper    -> browse_full_doc + read_doc_part
"""

import json
from typing import Dict, Optional, List, Any

# Default part size for browse_full_doc / read_doc_part (characters)
PART_SIZE = 4000


# ------------------------------------------------------------------
# OpenAI function-calling schema definitions
# ------------------------------------------------------------------

def get_tools_definition() -> List[Dict]:
    """Return tool schemas in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": (
                    "Search internal documents using hybrid retrieval (BM25 + vector). "
                    "Searches document-level abstracts, so each result represents one document. "
                    "Returns file_id, file_name, and a content summary for each hit. "
                    "Read the 'content' field to judge relevance before calling load_doc or read_chunk."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query, e.g. '平安银行2023营业收入'",
                        },
                        "size": {
                            "type": "integer",
                            "description": "Number of results to return (default 10, max 50)",
                            "default": 10,
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Restrict search to these file IDs (optional)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "quick_preview",
                "description": (
                    "Fetch the first 2000 characters of multiple documents concurrently. "
                    "Use after search_docs to quickly scan several documents and decide "
                    "which ones deserve a detailed read via load_doc + read_chunk. "
                    "Much faster than calling preview_doc in a loop."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file IDs to preview (from search_docs results)",
                        },
                    },
                    "required": ["file_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_doc",
                "description": (
                    "Load a document's chunk structure: every chunk's section title, "
                    "60-char description (TLDR), page number, and character count. "
                    "Does NOT return full chunk text. "
                    "Call this after search_docs or quick_preview to see which chunks "
                    "are worth reading, then use read_chunk for targeted content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Document file ID (from search_docs results)",
                        },
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preview_doc",
                "description": (
                    "Get the first 2000 characters of a single document. "
                    "Useful when you need to quickly confirm a document's content "
                    "before committing to a full load_doc + read_chunk cycle."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Document file ID",
                        },
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": (
                    "Read the full text of a specific chunk identified by its chunk_id. "
                    "This is the primary way to read document content in detail. "
                    "Always check the chunk's 'description' from load_doc or search_docs "
                    "before calling this — only read chunks likely to contain the answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "Chunk ES _id, obtained from search_docs or load_doc results",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browse_full_doc",
                "description": (
                    "Fetch the complete document and display a table of contents: "
                    "the document is split into numbered parts (~4000 chars each) and "
                    "only the first 150 chars of each part are shown. "
                    "Use this when read_chunk cannot locate the answer and you need to "
                    "scan the full document. Then call read_doc_part to read the relevant part. "
                    "WARNING: never use this as the first tool — always try search_docs and "
                    "read_chunk first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Document file ID",
                        },
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_doc_part",
                "description": (
                    "Read one specific part of a full document (identified by part_index). "
                    "Must call browse_full_doc first to load the document and see the part list. "
                    "Each part is ~4000 chars — manageable for the LLM without context overflow."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Document file ID",
                        },
                        "part_index": {
                            "type": "integer",
                            "description": "Zero-based part index shown in browse_full_doc output",
                        },
                    },
                    "required": ["file_id", "part_index"],
                },
            },
        },
    ]


# ------------------------------------------------------------------
# Tool executor
# ------------------------------------------------------------------

class ToolExecutor:
    """Executes agent tool calls against the Reader, with state management."""

    def __init__(self, reader: Any):
        self.reader = reader

    # ---- Individual tool methods ----------------------------------------

    def search_docs(
        self,
        query: str,
        session_id: str,
        search_cache: List,
        size: int = 10,
        file_ids: Optional[List[str]] = None,
    ) -> str:
        """Run hybrid search and return formatted document summaries."""
        results = self.reader.search(
            query=query,
            session_id=session_id,
            size=size,
            file_ids=file_ids,
        )

        if not results or not results.get("results"):
            return f"No documents found for '{query}'. Try rephrasing the query."

        search_cache.clear()
        search_cache.extend(results["results"])

        total = results.get("total", 0)
        hits  = results["results"]

        lines = [
            f"=== Search results for '{query}' ===",
            f"Found {total} documents, showing top {len(hits)}\n",
        ]
        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. [{hit['file_name']}]")
            if hit.get("section_title"):
                lines.append(f"   Section : {hit['section_title']}")
            lines.append(f"   file_id : {hit['file_id']}")
            lines.append(f"   Page    : {hit['page_num']} | Chars: {hit['token_count']}")
            lines.append(f"   Content : {hit['content'][:200]}")
            lines.append("")

        lines.append(
            "Tip: read 'Content' to judge relevance, "
            "then call quick_preview or load_doc for details."
        )
        return "\n".join(lines)

    def quick_preview(
        self,
        file_ids: List[str],
        session_id: str,
    ) -> str:
        """Fetch concurrent previews for multiple documents and format them."""
        if not file_ids:
            return "Error: file_ids list is empty."

        previews = self.reader.quick_preview(
            file_ids=file_ids,
            session_id=session_id,
            max_chars=2000,
        )
        if not previews:
            return "No preview content found for the provided file IDs."

        lines = [f"=== Quick preview ({len(previews)}/{len(file_ids)} documents found) ===\n"]
        for p in previews:
            lines.append(f"--- [{p['file_name']}] (file_id={p['file_id']}) ---")
            lines.append(p["preview"])
            if p.get("truncated"):
                lines.append("... [truncated — use load_doc + read_chunk for full content]")
            lines.append("")
        return "\n".join(lines)

    def load_doc(
        self,
        file_id: str,
        session_id: str,
        state_docs: Dict,
    ) -> str:
        """Load document chunk structure and cache it in agent state."""
        if file_id in state_docs:
            return self._format_doc_head(state_docs[file_id], from_cache=True)

        head = self.reader.head(file_id=file_id, session_id=session_id)
        if not head or not head.get("chunks"):
            return f"Could not load document '{file_id}'. Check that the file_id is correct."

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
        state_docs[file_id] = {
            "file_id":       file_id,
            "file_name":     head.get("file_name", ""),
            "file_urls":     head.get("file_urls", ""),
            "total_tokens":  head.get("total_tokens", 0),
            "chunks":        chunks_dict,
            "loaded_chunks": {},
        }
        return self._format_doc_head(state_docs[file_id], from_cache=False)

    def preview_doc(self, file_id: str, session_id: str) -> str:
        """Get a single document preview."""
        result = self.reader.preview(
            file_id=file_id, session_id=session_id, max_chars=2000
        )
        if not result or not result.get("preview"):
            return f"No preview available for document '{file_id}'."

        lines = [f"=== Preview: {result.get('file_name', file_id)} ===\n", result["preview"]]
        if result.get("truncated"):
            lines.append("\n... [truncated — use load_doc + read_chunk to continue]")
        return "\n".join(lines)

    def read_chunk(
        self,
        chunk_id: str,
        state_docs: Dict,
        chunk_cache: Dict,
    ) -> str:
        """Fetch and return the full text of a chunk."""
        if chunk_id in chunk_cache:
            return self._format_chunk(chunk_id, chunk_cache[chunk_id], from_cache=True)

        result = self.reader.read_chunk(chunk_id=chunk_id)
        if not result or not result.get("content"):
            return f"Could not read chunk '{chunk_id}'. Check that the chunk_id is correct."

        content = result["content"]
        chunk_cache[chunk_id] = content

        file_id = result.get("file_id", "")
        if file_id and file_id in state_docs:
            state_docs[file_id]["loaded_chunks"][chunk_id] = content

        lines = [
            "=== Chunk content ===",
            f"Document : {result.get('file_name', '')} | Page: {result.get('page_num', 0)}",
        ]
        if result.get("section_title"):
            lines.append(f"Section  : {result['section_title']}")
        lines.append(f"Chars    : {result.get('token_count', len(content))}\n")
        lines.append(content)
        lines.append("\n=== End of chunk ===")
        return "\n".join(lines)

    def browse_full_doc(
        self,
        file_id: str,
        session_id: str,
        full_doc_cache: Dict,
        part_size: int = PART_SIZE,
    ) -> str:
        """Fetch full document once, cache it, return a part-by-part table of contents.

        The LLM sees only the first 150 chars of each part and picks which
        part(s) to read via read_doc_part — no risk of context overflow.
        """
        # Fetch and cache the full document (only once per file_id)
        if file_id not in full_doc_cache:
            result = self.reader.raw(file_id=file_id, session_id=session_id)
            if not result or not result.get("content"):
                return f"Could not retrieve document '{file_id}'."
            full_doc_cache[file_id] = {
                "content":   result["content"],
                "file_name": result.get("file_name", file_id),
            }

        cached    = full_doc_cache[file_id]
        content   = cached["content"]
        file_name = cached["file_name"]
        total     = len(content)
        n_parts   = (total + part_size - 1) // part_size

        lines = [
            f"=== Full document overview: {file_name} ===",
            f"Total: {total:,} chars | {n_parts} parts (~{part_size} chars each)\n",
            "Part index | Char range          | Preview (first 150 chars)",
            "-" * 70,
        ]

        for i in range(n_parts):
            start   = i * part_size
            end     = min(start + part_size, total)
            preview = content[start:start + 150].replace("\n", " ").strip()
            lines.append(f"Part {i:>3}   | {start:>7,} – {end:>7,} | {preview}")

        lines.append("")
        lines.append(
            "Call read_doc_part(file_id, part_index) to read the content of a specific part."
        )
        return "\n".join(lines)

    def read_doc_part(
        self,
        file_id: str,
        part_index: int,
        full_doc_cache: Dict,
        part_size: int = PART_SIZE,
    ) -> str:
        """Return the full text of one document part.

        browse_full_doc must be called first (it populates full_doc_cache).
        """
        if file_id not in full_doc_cache:
            return (
                f"Document '{file_id}' is not loaded yet. "
                "Call browse_full_doc(file_id) first to load and inspect the document."
            )

        cached    = full_doc_cache[file_id]
        content   = cached["content"]
        file_name = cached["file_name"]
        total     = len(content)
        n_parts   = (total + part_size - 1) // part_size

        if part_index < 0 or part_index >= n_parts:
            return (
                f"part_index {part_index} is out of range. "
                f"Valid range: 0 – {n_parts - 1} (use browse_full_doc to see the part list)."
            )

        start        = part_index * part_size
        end          = min(start + part_size, total)
        part_content = content[start:end]

        lines = [
            f"=== Part {part_index} / {n_parts - 1}: {file_name} ===",
            f"Chars {start:,} – {end:,} of {total:,}\n",
            part_content,
            f"\n=== End of part {part_index} ===",
        ]
        if end < total:
            lines.append(
                f"Continues in part {part_index + 1} — "
                f"call read_doc_part('{file_id}', {part_index + 1}) if needed."
            )
        return "\n".join(lines)

    # ---- Central dispatcher -------------------------------------------------

    def execute_tool_call(self, tool_name: str, tool_args: Dict, state: Dict) -> str:
        """Dispatch a single tool call and return the formatted result string."""
        session_id = state.get("session_id", "")
        try:
            if tool_name == "search_docs":
                # allowed_file_ids is a hard restriction set by the caller (e.g. RAGRetrievalTool).
                # It overrides any file_ids the LLM might have specified, ensuring the agent
                # cannot escape the knowledge-base scope it was given.
                allowed = state.get("allowed_file_ids")
                effective_file_ids = allowed if allowed is not None else tool_args.get("file_ids")
                return self.search_docs(
                    query=tool_args.get("query", ""),
                    session_id=session_id,
                    search_cache=state["search_results_cache"],
                    size=tool_args.get("size", 10),
                    file_ids=effective_file_ids,
                )

            elif tool_name == "quick_preview":
                return self.quick_preview(
                    file_ids=tool_args.get("file_ids", []),
                    session_id=session_id,
                )

            elif tool_name == "load_doc":
                return self.load_doc(
                    file_id=tool_args.get("file_id", ""),
                    session_id=session_id,
                    state_docs=state["docs"],
                )

            elif tool_name == "preview_doc":
                return self.preview_doc(
                    file_id=tool_args.get("file_id", ""),
                    session_id=session_id,
                )

            elif tool_name == "read_chunk":
                chunk_id = tool_args.get("chunk_id", "")
                result_text = self.read_chunk(
                    chunk_id=chunk_id,
                    state_docs=state["docs"],
                    chunk_cache=state["chunk_cache"],
                )
                # Record source metadata so callers (e.g. RAGRetrievalTool) can
                # return precise segments rather than the synthesised answer string.
                content = state["chunk_cache"].get(chunk_id, "")
                if content:
                    # Find which file owns this chunk (populated by read_chunk above)
                    file_id, file_name, page_num, section_title = "", "", 0, ""
                    for fid, doc in state["docs"].items():
                        if chunk_id in doc.get("chunks", {}):
                            file_id      = fid
                            file_name    = doc.get("file_name", "")
                            chunk_meta   = doc["chunks"][chunk_id]
                            page_num     = chunk_meta.get("page_num", 0)
                            section_title = chunk_meta.get("section_title", "")
                            break
                    state["read_chunk_results"].append({
                        "chunk_id":      chunk_id,
                        "file_id":       file_id,
                        "file_name":     file_name,
                        "page_num":      page_num,
                        "section_title": section_title,
                        "content":       content,
                    })
                return result_text

            elif tool_name == "browse_full_doc":
                return self.browse_full_doc(
                    file_id=tool_args.get("file_id", ""),
                    session_id=session_id,
                    full_doc_cache=state["full_doc_cache"],
                )

            elif tool_name == "read_doc_part":
                return self.read_doc_part(
                    file_id=tool_args.get("file_id", ""),
                    part_index=tool_args.get("part_index", 0),
                    full_doc_cache=state["full_doc_cache"],
                )

            else:
                return f"Unknown tool: '{tool_name}'"

        except Exception as e:
            return f"Error executing '{tool_name}': {e}"

    # ---- Formatting helpers -------------------------------------------------

    def _format_doc_head(self, doc: Dict, from_cache: bool = False) -> str:
        """Render the chunk list of a loaded document for the LLM."""
        file_name = doc.get("file_name", doc.get("file_id", ""))
        chunks    = doc.get("chunks", {})
        note      = " (cached)" if from_cache else ""

        lines = [
            f"=== Document structure: {file_name}{note} ===",
            f"file_id     : {doc.get('file_id', '')}",
            f"Total chunks: {len(chunks)} | Total chars: {doc.get('total_tokens', 0)}\n",
            "Chunks (chunk_id | page | chars | section | description):",
        ]
        for chunk in sorted(chunks.values(), key=lambda c: c.get("page_num", 0)):
            section = f"[{chunk['section_title']}] " if chunk.get("section_title") else ""
            lines.append(
                f"  {chunk['chunk_id']}"
                f" | p{chunk['page_num']}"
                f" | {chunk['token_count']}c"
                f" | {section}{chunk['description']}"
            )
        lines.append(
            "\nTip: call read_chunk(chunk_id) for any chunk whose description "
            "suggests it contains the answer."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_chunk(chunk_id: str, content: str, from_cache: bool = False) -> str:
        """Format cached chunk content."""
        note = " (cached)" if from_cache else ""
        return (
            f"=== Chunk content{note} ===\n"
            f"chunk_id: {chunk_id}\n\n"
            f"{content}\n\n"
            f"=== End of chunk ==="
        )


def format_doc_context(docs: Dict) -> str:
    """Render a summary of loaded documents for the system prompt."""
    if not docs:
        return "No documents loaded yet."
    lines = []
    for file_id, doc in docs.items():
        n = len(doc.get("chunks", {}))
        lines.append(
            f"- {doc.get('file_name', file_id)}  (file_id={file_id}, {n} chunks)"
        )
    return "\n".join(lines)
