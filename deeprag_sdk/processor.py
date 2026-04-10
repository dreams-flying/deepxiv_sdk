"""
Document processing pipeline: parse → chunk → embed → index.

Input format (from your document parser):
    {
        "type": "pdf",
        "title": "平安银行2023年报.pdf",
        "url": "https://...",
        "file_id": "unique_file_id",
        "content": [
            {"index": 0, "type": "text",  "text": "...", "title": "...", "para_num": 1, "page_num": 1},
            {"index": 1, "type": "title", "text": "一、营业收入", "title": "...", "para_num": 2, "page_num": 1},
            ...
        ],
        "pubtime": "2025-09-19 17:30:54"
    }

Chunking strategy:
  1. Encounter type=title → force a new chunk boundary
  2. Accumulated chars exceed max_chunk_chars → force split
  3. Each chunk carries a 'description' (first 150 chars as TLDR)
  4. Two ES docs per chunk: one 'chunk', one 'full_text' for the whole document

This mirrors DeepXiv's section-level granularity:
  full_text doc  = the whole paper (for document-level preview)
  chunk docs     = individual sections (for precise reading)
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any


class Processor:
    """
    Ingest documents into Elasticsearch with BGE-M3 embeddings.

    Supports concurrent batch ingestion via process_batch().
    """

    def __init__(
        self,
        es_client: Any,
        index_name: str,
        embed_model: Any,
        max_chunk_chars: int = 1000,
        overlap_chars: int = 100,
        batch_size: int = 32,
    ):
        """
        Args:
            es_client:       Elasticsearch client.
            index_name:      Target ES index.
            embed_model:     BGE-M3 model (FlagEmbedding.BGEM3FlagModel).
            max_chunk_chars: Maximum characters per chunk (approx. 512 tokens
                             for Chinese text where 1 char ≈ 1.5 tokens).
            overlap_chars:   Characters to repeat at the start of next chunk
                             so context is not cut off at boundaries.
            batch_size:      Number of texts to embed in a single model call.
        """
        self.es = es_client
        self.index = index_name
        self.model = embed_model
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        doc: Dict,
        user_id: str,
        session_id: Optional[str] = None,
        permission: str = "user",
    ) -> int:
        """
        Process a single parsed document and store to Elasticsearch.

        Args:
            doc:        Parser output dict (see module docstring for schema).
            user_id:    Owner user ID — used for multi-tenant filtering.
            session_id: Optional session context; defaults to user_id.
            permission: Access permission label (default "user").

        Returns:
            Number of chunk documents indexed (excluding the full_text doc).
        """
        content_items: List[Dict] = doc.get("content", [])
        if not content_items:
            return 0

        session_id = session_id or user_id
        chunks = self._chunk(content_items)

        # Embed all chunks in one batch call for efficiency
        texts_to_embed = [c["text"] for c in chunks]
        full_text = self._build_full_text(content_items)
        all_texts = [full_text] + texts_to_embed  # index 0 = full doc

        vectors = self._embed_batch(all_texts)
        full_vector = vectors[0]
        chunk_vectors = vectors[1:]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = {
            "user_id": user_id,
            "session_id": session_id,
            "file_id": doc.get("file_id", str(uuid.uuid4())),
            "file_name": doc.get("title", ""),
            "title": doc.get("title", ""),
            "file_type": doc.get("type", "unknown"),
            "file_urls": doc.get("url", ""),
            "file_size": 0,
            "file_source": "",
            "vectorization_model": "bge-m3",
            "permission": permission,
            "is_use": 1,
            "create_by": "user",
            "update_by": "",
            "data_key": "",
            "create_time": now,
        }

        actions = []

        # 1. Full-text document (for preview and document-level retrieval)
        full_doc = {
            "_index": self.index,
            "_id": str(uuid.uuid4()),
            "_source": {
                **base,
                "content": full_text[:20000],  # store up to 20k chars
                "description": full_text[:200],
                "page_num": 0,
                "para_num": 0,
                "location_in_para": "",
                "chunk_index": -1,
                "section_title": "",
                "content_type": "text",
                "vectorization_method": "full_text",
                "data_purpose": "vec_full_text",
                "token_count": len(full_text),
                "content_vector_1024": full_vector,
            },
        }
        actions.append(full_doc)

        # 2. Chunk documents
        for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
            chunk_doc = {
                "_index": self.index,
                "_id": str(uuid.uuid4()),
                "_source": {
                    **base,
                    "content": chunk["text"],
                    "description": chunk["description"],
                    "page_num": chunk["page_num"],
                    "para_num": chunk["para_num"],
                    "location_in_para": chunk.get("location_in_para", ""),
                    "chunk_index": i,
                    "section_title": chunk["section_title"],
                    "content_type": "text",
                    "vectorization_method": "chunk",
                    "data_purpose": "vec_chunk",
                    "token_count": chunk["token_count"],
                    "content_vector_1024": vector,
                },
            }
            actions.append(chunk_doc)

        self._bulk_index(actions)
        return len(chunks)

    def process_batch(
        self,
        docs: List[Dict],
        user_id: str,
        session_id: Optional[str] = None,
        permission: str = "user",
    ) -> Dict[str, int]:
        """
        Process multiple documents. Returns {file_id: chunk_count}.
        """
        results = {}
        for doc in docs:
            file_id = doc.get("file_id", "unknown")
            try:
                count = self.process(doc, user_id, session_id, permission)
                results[file_id] = count
            except Exception as e:
                results[file_id] = -1
                print(f"[Processor] Failed to process {file_id}: {e}")
        return results

    def delete_document(self, file_id: str, user_id: Optional[str] = None) -> int:
        """
        Delete all ES documents for a given file_id.
        Returns number of deleted documents.
        """
        filters: List[Dict] = [{"term": {"file_id": file_id}}]
        if user_id:
            filters.append({"term": {"user_id": user_id}})

        resp = self.es.delete_by_query(
            index=self.index,
            body={"query": {"bool": {"filter": filters}}},
        )
        return resp.get("deleted", 0)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk(self, content_items: List[Dict]) -> List[Dict]:
        """
        Split content items into overlapping chunks.

        Strategy:
          - Start a new chunk when a title item is encountered
          - Also split when accumulated text exceeds max_chunk_chars
          - Carry overlap_chars from the previous chunk into the next
        """
        chunks: List[Dict] = []
        current_texts: List[str] = []
        current_chars = 0
        current_section = ""
        current_page = 0
        current_para = 0

        def flush(texts: List[str], section: str, page: int, para: int) -> None:
            if not texts:
                return
            text = " ".join(texts)
            chunks.append(
                {
                    "text": text,
                    "description": self._make_description(text),
                    "section_title": section,
                    "page_num": page,
                    "para_num": para,
                    "location_in_para": "",
                    "token_count": len(text),
                }
            )

        for item in content_items:
            raw = item.get("text", "").strip()
            if not raw:
                continue

            item_type = item.get("type", "text")
            page = item.get("page_num", current_page)
            para = item.get("para_num", current_para)

            if item_type == "title":
                # Title boundary: flush current buffer, start new chunk
                flush(current_texts, current_section, current_page, current_para)
                # Carry overlap from end of previous chunk
                overlap = self._tail_overlap(current_texts)
                current_section = raw
                current_texts = overlap + [raw]
                current_chars = sum(len(t) for t in current_texts)
            else:
                item_chars = len(raw)

                if current_chars + item_chars > self.max_chunk_chars and current_texts:
                    # Size limit reached: flush and start new chunk with overlap
                    flush(current_texts, current_section, current_page, current_para)
                    overlap = self._tail_overlap(current_texts)
                    current_texts = overlap + [raw]
                    current_chars = sum(len(t) for t in current_texts)
                else:
                    current_texts.append(raw)
                    current_chars += item_chars

            current_page = page
            current_para = para

        # Final flush
        flush(current_texts, current_section, current_page, current_para)
        return chunks

    def _tail_overlap(self, texts: List[str]) -> List[str]:
        """
        Return the tail of `texts` whose total char count ≤ overlap_chars.
        """
        overlap: List[str] = []
        acc = 0
        for text in reversed(texts):
            if acc + len(text) <= self.overlap_chars:
                overlap.insert(0, text)
                acc += len(text)
            else:
                break
        return overlap

    @staticmethod
    def _make_description(text: str, max_chars: int = 150) -> str:
        """Extract first sentence(s) up to max_chars as the chunk TLDR."""
        # Split on Chinese/English sentence terminators
        for sep in ("。", "！", "？", ".", "!", "?", "\n"):
            idx = text.find(sep)
            if 0 < idx < max_chars:
                return text[: idx + 1].strip()
        return text[:max_chars].strip()

    @staticmethod
    def _build_full_text(content_items: List[Dict]) -> str:
        return " ".join(
            item.get("text", "").strip()
            for item in content_items
            if item.get("text", "").strip()
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using BGE-M3, respecting batch_size.
        Returns a list of 1024-dim float lists.
        """
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = result["dense_vecs"]
            for vec in vecs:
                all_vectors.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
        return all_vectors

    # ------------------------------------------------------------------
    # ES indexing
    # ------------------------------------------------------------------

    def _bulk_index(self, actions: List[Dict]) -> None:
        """Bulk index documents into Elasticsearch."""
        from elasticsearch.helpers import bulk
        bulk(self.es, actions)
