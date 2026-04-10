"""
Core retrieval interface backed by Elasticsearch 8.x.

Progressive disclosure pattern (mirrors DeepXiv):
  search()     → lightweight chunk metadata + TLDR descriptions
  head()       → document structure: all chunk TLDRs + token counts
  preview()    → first N characters of the document
  read_chunk() → full content of a specific chunk (by ES _id)
  raw()        → complete document (all chunks concatenated)
"""

from typing import List, Dict, Optional, Any


class Reader:
    """
    Retrieval interface over Elasticsearch.

    Hybrid search combines BM25 (lexical) with kNN (semantic) using
    Elasticsearch 8.x native RRF (Reciprocal Rank Fusion), requiring ES 8.8+.
    Falls back to score-based fusion for older versions.
    """

    def __init__(
        self,
        es_client: Any,
        index_name: str,
        embed_model: Any,
        default_size: int = 10,
    ):
        """
        Args:
            es_client:   Elasticsearch client instance.
            index_name:  Target ES index name.
            embed_model: BGE-M3 (or compatible) embedding model.
                         Must implement .encode(text, return_dense=True).
            default_size: Default number of results returned by search().
        """
        self.es = es_client
        self.index = index_name
        self.model = embed_model
        self.default_size = default_size

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        size: int = 10,
        file_ids: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict:
        """
        Hybrid search: BM25 + kNN with RRF ranking (ES 8.8+).

        Returns lightweight results — full chunk content is excluded.
        Each result contains a 'description' field (the chunk TLDR),
        which the agent can use for low-cost relevance judgement.

        Args:
            query:     Natural language query.
            user_id:   Filter results to this user's documents.
            size:      Number of results to return.
            file_ids:  Optional list of file IDs to restrict the search.
            date_from: Filter by create_time >= date_from (YYYY-MM-DD).
            date_to:   Filter by create_time <= date_to (YYYY-MM-DD).

        Returns:
            {
              "total": int,
              "results": [
                {
                  "chunk_id": str,      # ES _id, used for read_chunk()
                  "file_id": str,
                  "file_name": str,
                  "title": str,
                  "section_title": str,
                  "description": str,   # TLDR — read this before read_chunk()
                  "page_num": int,
                  "chunk_index": int,
                  "token_count": int,
                  "score": float,
                },
                ...
              ]
            }
        """
        query_vector = self._embed(query)
        filters = self._build_filters(
            user_id=user_id,
            file_ids=file_ids,
            date_from=date_from,
            date_to=date_to,
            vectorization_method="chunk",
        )

        body = {
            "size": size,
            # BM25 leg
            "query": {
                "bool": {
                    "must": [{"match": {"content": {"query": query, "boost": 1.0}}}],
                    "filter": filters,
                }
            },
            # kNN leg
            "knn": {
                "field": "content_vector_1024",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": size * 10,
                "filter": filters,
            },
            # RRF fusion (ES 8.8+)
            "rank": {"rrf": {"window_size": size * 5, "rank_constant": 60}},
            # Exclude heavy fields from results
            "_source": {
                "excludes": ["content_vector_1024", "content"]
            },
        }

        try:
            resp = self.es.search(index=self.index, body=body)
        except Exception as e:
            # Fallback: run BM25 and kNN separately, fuse in Python
            resp = self._fallback_hybrid_search(query, query_vector, filters, size)

        return self._format_search_response(resp)

    def head(self, file_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Load document structure: metadata + all chunk TLDRs + token counts.

        Cheap to call — does NOT return full chunk content.
        Use this after search() to decide which chunks are worth reading.

        Returns:
            {
              "file_id": str,
              "file_name": str,
              "title": str,
              "file_urls": str,
              "total_chunks": int,
              "total_tokens": int,
              "chunks": [
                {
                  "chunk_id": str,
                  "chunk_index": int,
                  "section_title": str,
                  "description": str,   # TLDR of this chunk
                  "page_num": int,
                  "token_count": int,
                },
                ...
              ]
            }
        """
        filters = self._build_filters(
            user_id=user_id,
            file_ids=[file_id],
            vectorization_method="chunk",
        )

        body = {
            "size": 500,  # support up to 500 chunks per document
            "query": {"bool": {"filter": filters}},
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": {
                "excludes": ["content_vector_1024", "content"]
            },
        }

        resp = self.es.search(index=self.index, body=body)
        chunks = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            chunks.append(
                {
                    "chunk_id": hit["_id"],
                    "chunk_index": src.get("chunk_index", 0),
                    "section_title": src.get("section_title", ""),
                    "description": src.get("description", ""),
                    "page_num": src.get("page_num", 0),
                    "token_count": src.get("token_count", 0),
                }
            )

        meta = self._get_file_meta(file_id, user_id)
        return {
            "file_id": file_id,
            "file_name": meta.get("file_name", ""),
            "title": meta.get("title", ""),
            "file_urls": meta.get("file_urls", ""),
            "total_chunks": len(chunks),
            "total_tokens": sum(c["token_count"] for c in chunks),
            "chunks": chunks,
        }

    def preview(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        max_chars: int = 2000,
    ) -> Dict:
        """
        Get first N characters of the document.

        Tries the stored full_text first (faster), then assembles from
        the first few chunks as a fallback.

        Returns:
            {
              "file_id": str,
              "file_name": str,
              "preview": str,
              "truncated": bool,
            }
        """
        # Try full_text document first
        full_filters = self._build_filters(
            user_id=user_id,
            file_ids=[file_id],
            vectorization_method="full_text",
        )
        body = {
            "size": 1,
            "query": {"bool": {"filter": full_filters}},
            "_source": ["content", "title", "file_name", "file_urls"],
        }
        resp = self.es.search(index=self.index, body=body)

        if resp["hits"]["hits"]:
            src = resp["hits"]["hits"][0]["_source"]
            content = src.get("content", "")
            return {
                "file_id": file_id,
                "file_name": src.get("file_name", ""),
                "title": src.get("title", ""),
                "preview": content[:max_chars],
                "truncated": len(content) > max_chars,
            }

        # Fallback: first few chunks
        chunk_filters = self._build_filters(
            user_id=user_id,
            file_ids=[file_id],
            vectorization_method="chunk",
        )
        body = {
            "size": 5,
            "query": {"bool": {"filter": chunk_filters}},
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["content", "title", "file_name"],
        }
        resp = self.es.search(index=self.index, body=body)

        texts = [hit["_source"].get("content", "") for hit in resp["hits"]["hits"]]
        combined = " ".join(texts)
        file_name = (
            resp["hits"]["hits"][0]["_source"].get("file_name", "")
            if resp["hits"]["hits"]
            else ""
        )
        return {
            "file_id": file_id,
            "file_name": file_name,
            "title": "",
            "preview": combined[:max_chars],
            "truncated": len(combined) > max_chars,
        }

    def read_chunk(self, chunk_id: str) -> Dict:
        """
        Fetch the full content of a specific chunk by its ES _id.

        This is the primary way to read document content after using
        head() or search() to identify relevant chunks.

        Returns:
            {
              "chunk_id": str,
              "file_id": str,
              "file_name": str,
              "section_title": str,
              "content": str,     # Full chunk text
              "page_num": int,
              "chunk_index": int,
              "token_count": int,
            }
        """
        resp = self.es.get(index=self.index, id=chunk_id)
        src = resp["_source"]
        return {
            "chunk_id": chunk_id,
            "file_id": src.get("file_id", ""),
            "file_name": src.get("file_name", ""),
            "section_title": src.get("section_title", ""),
            "content": src.get("content", ""),
            "page_num": src.get("page_num", 0),
            "chunk_index": src.get("chunk_index", 0),
            "token_count": src.get("token_count", 0),
        }

    def raw(self, file_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Get the complete document by concatenating all chunks in order.

        Use sparingly — can be very large. Prefer read_chunk() for targeted reads.

        Returns:
            {
              "file_id": str,
              "file_name": str,
              "title": str,
              "content": str,       # Full document text
              "total_chunks": int,
            }
        """
        filters = self._build_filters(
            user_id=user_id,
            file_ids=[file_id],
            vectorization_method="chunk",
        )
        body = {
            "size": 500,
            "query": {"bool": {"filter": filters}},
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["content", "section_title", "page_num", "chunk_index"],
        }
        resp = self.es.search(index=self.index, body=body)
        meta = self._get_file_meta(file_id, user_id)

        parts = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            section = src.get("section_title", "")
            content = src.get("content", "")
            if section:
                parts.append(f"## {section}\n{content}")
            else:
                parts.append(content)

        return {
            "file_id": file_id,
            "file_name": meta.get("file_name", ""),
            "title": meta.get("title", ""),
            "content": "\n\n".join(parts),
            "total_chunks": len(resp["hits"]["hits"]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Generate BGE-M3 dense embedding for a single text."""
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = result["dense_vecs"][0]
        # Convert numpy array to Python list for ES serialization
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _build_filters(
        self,
        user_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        vectorization_method: Optional[str] = None,
    ) -> List[Dict]:
        filters: List[Dict] = [{"term": {"is_use": 1}}]
        if user_id:
            filters.append({"term": {"user_id": user_id}})
        if file_ids:
            filters.append({"terms": {"file_id": file_ids}})
        if vectorization_method:
            filters.append({"term": {"vectorization_method": vectorization_method}})
        if date_from or date_to:
            range_clause: Dict = {}
            if date_from:
                range_clause["gte"] = date_from
            if date_to:
                range_clause["lte"] = date_to
            filters.append({"range": {"create_time": range_clause}})
        return filters

    def _get_file_meta(
        self, file_id: str, user_id: Optional[str] = None
    ) -> Dict:
        filters = self._build_filters(
            user_id=user_id,
            file_ids=[file_id],
            vectorization_method="full_text",
        )
        body = {
            "size": 1,
            "query": {"bool": {"filter": filters}},
            "_source": ["file_name", "title", "file_urls", "create_time"],
        }
        resp = self.es.search(index=self.index, body=body)
        if resp["hits"]["hits"]:
            return resp["hits"]["hits"][0]["_source"]
        return {}

    def _format_search_response(self, resp: Dict) -> Dict:
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                {
                    "chunk_id": hit["_id"],
                    "file_id": src.get("file_id", ""),
                    "file_name": src.get("file_name", ""),
                    "title": src.get("title", ""),
                    "section_title": src.get("section_title", ""),
                    "description": src.get("description", ""),
                    "page_num": src.get("page_num", 0),
                    "chunk_index": src.get("chunk_index", 0),
                    "token_count": src.get("token_count", 0),
                    "score": hit.get("_score") or 0.0,
                }
            )
        return {
            "total": resp["hits"]["total"]["value"],
            "results": results,
        }

    def _fallback_hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        filters: List[Dict],
        size: int,
    ) -> Dict:
        """
        Fallback hybrid search for ES versions that don't support RRF.
        Runs BM25 and kNN separately, fuses with Python-side RRF.
        """
        # BM25
        bm25_body = {
            "size": size * 2,
            "query": {
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": filters,
                }
            },
            "_source": {"excludes": ["content_vector_1024", "content"]},
        }
        bm25_resp = self.es.search(index=self.index, body=bm25_body)

        # kNN (ES 8.x _search with knn parameter, no rank)
        knn_body = {
            "size": size * 2,
            "knn": {
                "field": "content_vector_1024",
                "query_vector": query_vector,
                "k": size * 2,
                "num_candidates": size * 20,
                "filter": filters,
            },
            "_source": {"excludes": ["content_vector_1024", "content"]},
        }
        knn_resp = self.es.search(index=self.index, body=knn_body)

        # Python-side RRF
        rrf_scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}

        for rank, hit in enumerate(bm25_resp["hits"]["hits"], 1):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            sources[doc_id] = hit

        for rank, hit in enumerate(knn_resp["hits"]["hits"], 1):
            doc_id = hit["_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            if doc_id not in sources:
                sources[doc_id] = hit

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:size]

        hits = []
        for doc_id in sorted_ids:
            hit = dict(sources[doc_id])
            hit["_score"] = rrf_scores[doc_id]
            hits.append(hit)

        return {
            "hits": {
                "total": {"value": len(rrf_scores)},
                "hits": hits,
            }
        }
