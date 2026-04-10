"""
Tools available to the RAG ReAct agent.

Five tools, ordered by token cost (cheapest first):
  search_docs  → chunk metadata + TLDRs (no full content)
  load_doc     → document structure with all chunk TLDRs
  preview_doc  → first ~2000 chars of a document
  read_chunk   → full content of one chunk (targeted read)
  get_full_doc → complete document (use sparingly)

Mirrors DeepXiv's tool design:
  search_papers  → search_docs
  load_paper     → load_doc
  get_paper_preview → preview_doc
  read_section   → read_chunk
  get_full_paper → get_full_doc
"""

import json
from typing import Dict, Optional, List, Any


# ------------------------------------------------------------------
# OpenAI function-calling definitions
# ------------------------------------------------------------------

def get_tools_definition() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": (
                    "在内部文档库中进行混合检索（BM25 + 向量语义搜索）。"
                    "返回最相关的文档分块列表，每个结果包含：chunk_id、file_id、"
                    "file_name、section_title、description（分块摘要TLDR）、page_num、token_count。"
                    "注意：不返回分块全文，请先通过 description 判断相关性，"
                    "再调用 read_chunk 精读。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索查询，如'平安银行2023营业收入'、'员工绩效考核流程'",
                        },
                        "size": {
                            "type": "integer",
                            "description": "返回结果数量，默认10，最大50",
                            "default": 10,
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "限定在指定 file_id 列表内搜索（可选）",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "按文档创建时间过滤，起始日期 YYYY-MM-DD（可选）",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "按文档创建时间过滤，截止日期 YYYY-MM-DD（可选）",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_doc",
                "description": (
                    "加载某文档的完整分块结构，返回每个 chunk 的：chunk_id、"
                    "section_title、description（TLDR）、page_num、token_count。"
                    "不返回分块全文。在 search_docs 找到相关文档后，"
                    "用此工具了解文档内部结构，决定读哪些 chunk。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "文档的唯一 ID（从 search_docs 结果中获取）",
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
                    "获取文档的前 2000 字内容，用于快速了解文档整体内容和风格。"
                    "适合在不确定文档相关性时使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "文档的唯一 ID",
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
                    "读取某个分块的完整内容。这是精读文档的主要方式。"
                    "在 search_docs 或 load_doc 返回的结果中找到目标 chunk_id，"
                    "调用此工具获取该分块的完整文本。"
                    "请在读 description 确认相关后再调用，避免浪费 token。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "分块的唯一 ID（ES _id），从 search_docs 或 load_doc 结果中获取",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_full_doc",
                "description": (
                    "获取完整文档内容（所有分块按顺序拼接）。"
                    "警告：文档可能非常长（数万字）。"
                    "仅在以下情况使用：需要理解文档整体脉络、"
                    "或答案可能分散在多个位置且无法通过 read_chunk 定位时。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "文档的唯一 ID",
                        },
                    },
                    "required": ["file_id"],
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

    # ---- Individual tool methods ----

    def search_docs(
        self,
        query: str,
        user_id: str,
        state_docs: Dict,
        search_cache: List,
        size: int = 10,
        file_ids: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        results = self.reader.search(
            query=query,
            user_id=user_id,
            size=size,
            file_ids=file_ids,
            date_from=date_from,
            date_to=date_to,
        )

        if not results or not results.get("results"):
            return f"未找到与"{query}"相关的文档分块。请尝试换个关键词。"

        # Update cache
        search_cache.clear()
        search_cache.extend(results["results"])

        total = results.get("total", 0)
        hits = results["results"]

        lines = [
            f"=== 检索结果："{query}" ===",
            f"共找到 {total} 个相关分块，显示前 {len(hits)} 个\n",
        ]

        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. 【{hit['file_name']}】")
            if hit.get("section_title"):
                lines.append(f"   章节：{hit['section_title']}")
            lines.append(f"   chunk_id：{hit['chunk_id']}")
            lines.append(f"   file_id：{hit['file_id']}")
            lines.append(f"   页码：第 {hit['page_num']} 页 | 字符数：{hit['token_count']}")
            lines.append(f"   摘要：{hit['description']}")
            lines.append("")

        lines.append("提示：先通过'摘要'判断相关性，再调用 read_chunk 获取全文。")
        return "\n".join(lines)

    def load_doc(
        self,
        file_id: str,
        user_id: str,
        state_docs: Dict,
    ) -> str:
        # Check cache
        if file_id in state_docs:
            doc = state_docs[file_id]
            return self._format_doc_head(doc, from_cache=True)

        head = self.reader.head(file_id=file_id, user_id=user_id)
        if not head or not head.get("chunks"):
            return f"未能加载文档 {file_id}，请检查 file_id 是否正确。"

        # Build DocInfo and store in state
        chunks_dict = {}
        for c in head["chunks"]:
            chunks_dict[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "chunk_index": c["chunk_index"],
                "section_title": c["section_title"],
                "description": c["description"],
                "page_num": c["page_num"],
                "token_count": c["token_count"],
            }

        state_docs[file_id] = {
            "file_id": file_id,
            "file_name": head.get("file_name", ""),
            "title": head.get("title", ""),
            "file_urls": head.get("file_urls", ""),
            "total_tokens": head.get("total_tokens", 0),
            "chunks": chunks_dict,
            "loaded_chunks": {},
        }

        return self._format_doc_head(state_docs[file_id], from_cache=False)

    def preview_doc(
        self,
        file_id: str,
        user_id: str,
    ) -> str:
        result = self.reader.preview(file_id=file_id, user_id=user_id, max_chars=2000)
        if not result or not result.get("preview"):
            return f"无法获取文档 {file_id} 的预览内容。"

        lines = [
            f"=== 文档预览：{result.get('file_name', file_id)} ===\n",
            result["preview"],
        ]
        if result.get("truncated"):
            lines.append("\n... [内容已截断，如需继续请使用 load_doc + read_chunk]")
        return "\n".join(lines)

    def read_chunk(
        self,
        chunk_id: str,
        state_docs: Dict,
        chunk_cache: Dict,
    ) -> str:
        # Check cache
        if chunk_id in chunk_cache:
            content = chunk_cache[chunk_id]
            return self._format_chunk_content(chunk_id, content, from_cache=True)

        result = self.reader.read_chunk(chunk_id=chunk_id)
        if not result or not result.get("content"):
            return f"无法读取分块 {chunk_id}，请检查 chunk_id 是否正确。"

        content = result["content"]
        # Cache
        chunk_cache[chunk_id] = content

        # Also update state_docs loaded_chunks if we know the file
        file_id = result.get("file_id", "")
        if file_id and file_id in state_docs:
            state_docs[file_id]["loaded_chunks"][chunk_id] = content

        lines = [
            f"=== 分块内容 ===",
            f"文档：{result.get('file_name', '')} | 页码：第 {result.get('page_num', 0)} 页",
        ]
        if result.get("section_title"):
            lines.append(f"章节：{result['section_title']}")
        lines.append(f"字符数：{result.get('token_count', len(content))}\n")
        lines.append(content)
        lines.append("\n=== 分块结束 ===")
        return "\n".join(lines)

    def get_full_doc(
        self,
        file_id: str,
        user_id: str,
    ) -> str:
        result = self.reader.raw(file_id=file_id, user_id=user_id)
        if not result or not result.get("content"):
            return f"无法获取文档 {file_id} 的完整内容。"

        content = result["content"]
        lines = [
            f"=== 完整文档：{result.get('file_name', file_id)} ===",
            f"共 {result.get('total_chunks', 0)} 个分块\n",
            content,
            "\n=== 文档结束 ===",
        ]
        return "\n".join(lines)

    # ---- Dispatch ----

    def execute_tool_call(
        self,
        tool_name: str,
        tool_args: Dict,
        state: Dict,
    ) -> str:
        user_id = state.get("user_id", "")
        try:
            if tool_name == "search_docs":
                return self.search_docs(
                    query=tool_args.get("query", ""),
                    user_id=user_id,
                    state_docs=state["docs"],
                    search_cache=state["search_results_cache"],
                    size=tool_args.get("size", 10),
                    file_ids=tool_args.get("file_ids"),
                    date_from=tool_args.get("date_from"),
                    date_to=tool_args.get("date_to"),
                )
            elif tool_name == "load_doc":
                return self.load_doc(
                    file_id=tool_args.get("file_id", ""),
                    user_id=user_id,
                    state_docs=state["docs"],
                )
            elif tool_name == "preview_doc":
                return self.preview_doc(
                    file_id=tool_args.get("file_id", ""),
                    user_id=user_id,
                )
            elif tool_name == "read_chunk":
                return self.read_chunk(
                    chunk_id=tool_args.get("chunk_id", ""),
                    state_docs=state["docs"],
                    chunk_cache=state["chunk_cache"],
                )
            elif tool_name == "get_full_doc":
                return self.get_full_doc(
                    file_id=tool_args.get("file_id", ""),
                    user_id=user_id,
                )
            else:
                return f"未知工具：{tool_name}"
        except Exception as e:
            return f"工具 {tool_name} 执行出错：{e}"

    # ---- Formatting helpers ----

    def _format_doc_head(self, doc: Dict, from_cache: bool = False) -> str:
        file_name = doc.get("file_name", doc.get("file_id", ""))
        chunks = doc.get("chunks", {})
        total_tokens = doc.get("total_tokens", 0)
        cache_note = "（已缓存）" if from_cache else ""

        lines = [
            f"=== 文档结构：{file_name} {cache_note}===",
            f"file_id：{doc.get('file_id', '')}",
            f"共 {len(chunks)} 个分块，约 {total_tokens} 字符\n",
            "分块列表（chunk_id | 页码 | 字符数 | 章节 | 摘要）：",
        ]

        for chunk in sorted(chunks.values(), key=lambda c: c.get("chunk_index", 0)):
            section = f"[{chunk['section_title']}] " if chunk.get("section_title") else ""
            lines.append(
                f"  chunk_id={chunk['chunk_id']}"
                f" | 第{chunk['page_num']}页"
                f" | {chunk['token_count']}字"
                f" | {section}{chunk['description']}"
            )

        lines.append(
            "\n提示：通过 read_chunk(chunk_id) 读取你认为相关的分块全文。"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_chunk_content(
        chunk_id: str, content: str, from_cache: bool = False
    ) -> str:
        note = "（缓存）" if from_cache else ""
        return f"=== 分块内容 {note}===\nchunk_id: {chunk_id}\n\n{content}\n\n=== 分块结束 ==="


def format_doc_context(docs: Dict) -> str:
    """Render loaded documents summary for the system prompt."""
    if not docs:
        return "暂无已加载文档。"
    parts = []
    for file_id, doc in docs.items():
        n = len(doc.get("chunks", {}))
        parts.append(f"- {doc.get('file_name', file_id)} (file_id={file_id}, {n}个分块)")
    return "\n".join(parts)
