"""
Core retrieval interface backed by Elasticsearch 8.x.

Progressive disclosure pattern (mirrors DeepXiv):
  search()       -> document-level hybrid search (BM25 + kNN), returns TLDR/abstract
  head()         -> document structure: all chunk descriptions + token counts
  preview()      -> first N characters of one document
  quick_preview()-> first N characters of multiple documents (concurrent)
  read_chunk()   -> full content of a specific chunk (by ES _id)
  raw()          -> complete document from full_text record
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any


def get_embedding(data_list: list) -> list:
    """Convert a list of texts into dense embeddings via the remote BGE service.

    Args:
        data_list: List of strings to embed.

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    import requests

    GET_EMBEDDING_BGE_URL = "http://172.16.10.65:41110/analysis"
    payload = {
        "id": "82cf544a-e5f3-4c9b-bb14-40847a6410da",
        "content": {
            "function": "get_embedding",
            "data_list": data_list,
        },
    }
    response = requests.post(GET_EMBEDDING_BGE_URL, json=payload)
    response.raise_for_status()
    return response.json()["data"]["embedding_list"]


class Reader:
    """Retrieval interface over Elasticsearch.

    Uses Python-side RRF to fuse BM25 and kNN results, compatible with all
    ES 8.x versions (no dependency on the ES 8.8+ native RRF feature).

    Search targets the 'full_text_abstract' index layer (document-level).
    Chunk reading targets the 'title+content' index layer (chunk-level).
    """

    def __init__(
        self,
        es_client: Any,
        index_name: str,
        embed_model: Optional[Any] = None,  # reserved for local model; unused when get_embedding() is active
        default_size: int = 10,
    ):
        """
        Args:
            es_client:    Elasticsearch client instance.
            index_name:   Target ES index name.
            embed_model:  Reserved; pass None when using the remote embedding service.
            default_size: Default number of search results.
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
        session_id: Optional[str] = None,
        size: int = 10,
        file_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Hybrid search over document-level abstracts (BM25 + kNN, Python-side RRF).

        Searches 'full_text_abstract' documents so each result represents one
        document, not an individual chunk.  The 'content' field returned is
        the document abstract/summary — cheap to scan for relevance.

        Args:
            query:      Natural language query.
            session_id: Filter to this session's documents.
            size:       Number of results to return.
            file_ids:   Restrict search to these file IDs (optional).

        Returns:
            {
              "total": int,
              "results": [
                {
                  "chunk_id":      str,   # ES _id of the abstract doc
                  "file_id":       str,
                  "file_name":     str,
                  "title":         str,
                  "section_title": str,
                  "content":       str,   # abstract / summary — read before head()
                  "page_num":      int,
                  "polarity":      int,
                  "token_count":   int,
                  "score":         float,
                }
              ]
            }
        """
        query_vector = self._embed(query)[0]
        filters = self._build_filters(
            session_id=session_id,
            file_ids=file_ids,
            vectorization_method="full_text_abstract",
        )

        # BM25 leg
        bm25_body = {
            "size": size * 2,
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": query, "fields": ["title^1", "content^2"]}}
                    ],
                    "filter": filters,
                }
            },
            "_source": {"excludes": ["content_vector_1024"]},
        }
        bm25_resp = self.es.search(index=self.index, body=bm25_body)

        # kNN leg
        knn_body = {
            "size": size * 2,
            "knn": {
                "field": "content_vector_1024",
                "query_vector": query_vector,
                "k": size * 2,
                "num_candidates": size * 20,
                "filter": filters,
            },
            "_source": {"excludes": ["content_vector_1024"]},
        }
        knn_resp = self.es.search(index=self.index, body=knn_body)

        # Python-side RRF fusion
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

        return self._format_search_response({
            "hits": {
                "total": {"value": len(rrf_scores)},
                "hits": hits,
            }
        })

    def head(self, file_id: str, session_id: Optional[str] = None) -> Dict:
        """Load document chunk structure with descriptions and token counts.

        Fetches all 'title+content' chunks for the document.
        Does NOT return full chunk text — only the first 60 chars as a TLDR.
        Call read_chunk() to get the full content of a specific chunk.

        Returns:
            {
              "file_id":      str,
              "file_name":    str,
              "file_urls":    str,
              "total_chunks": int,
              "total_tokens": int,
              "chunks": [
                {
                  "chunk_id":     str,   # pass to read_chunk()
                  "section_title":str,
                  "description":  str,   # first 60 chars — judge relevance cheaply
                  "page_num":     int,
                  "token_count":  int,
                }
              ]
            }
        """
        filters = self._build_filters(
            session_id=session_id,
            file_ids=[file_id],
            vectorization_method="title+content",
        )
        body = {
            "size": 500,
            "query": {"bool": {"filter": filters}},
            "_source": {"excludes": ["content_vector_1024"]},
        }
        resp = self.es.search(index=self.index, body=body)

        file_name = ""
        file_urls = ""
        if resp["hits"]["hits"]:
            first = resp["hits"]["hits"][0]["_source"]
            file_name = first.get("file_name", "")
            file_urls = first.get("file_urls", "")

        chunks = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            content = src.get("content", "")
            chunks.append({
                "chunk_id":      hit["_id"],
                "section_title": src.get("section_title", ""),
                "description":   content[:60],   # cheap TLDR
                "page_num":      src.get("page_num", 0),
                "token_count":   len(content),
            })

        return {
            "file_id":      file_id,
            "file_name":    file_name,
            "file_urls":    file_urls,
            "total_chunks": len(chunks),
            "total_tokens": sum(c["token_count"] for c in chunks),
            "chunks":       chunks,
        }

    def preview(
        self,
        file_id: str,
        session_id: Optional[str] = None,
        max_chars: int = 2000,
    ) -> Dict:
        """Get the first N characters of a document from its full_text record.

        Returns:
            {
              "file_id":   str,
              "file_name": str,
              "preview":   str,
              "truncated": bool,
            }
            Empty dict if the document is not found.
        """
        filters = self._build_filters(
            session_id=session_id,
            file_ids=[file_id],
            vectorization_method="full_text",
        )
        body = {
            "size": 1,
            "query": {"bool": {"filter": filters}},
            "_source": ["content", "file_name", "file_urls"],
        }
        resp = self.es.search(index=self.index, body=body)

        if not resp["hits"]["hits"]:
            return {}

        src = resp["hits"]["hits"][0]["_source"]
        content = src.get("content", "")
        return {
            "file_id":   file_id,
            "file_name": src.get("file_name", ""),
            "preview":   content[:max_chars],
            "truncated": len(content) > max_chars,
        }

    def quick_preview(
        self,
        file_ids: List[str],
        session_id: Optional[str] = None,
        max_chars: int = 2000,
        max_workers: int = 5,
    ) -> List[Dict]:
        """Fetch previews for multiple documents concurrently.

        Uses a thread pool so N documents take roughly the same time as 1.
        Results are returned in the same order as the input file_ids list.

        Args:
            file_ids:    List of file IDs to preview.
            session_id:  Filter to this session's documents.
            max_chars:   Maximum characters to return per document.
            max_workers: Thread-pool size (default 5).

        Returns:
            List of preview dicts (same schema as preview()), one per file_id.
            Documents that are not found are omitted from the output.
        """
        def _fetch(file_id: str) -> Optional[Dict]:
            result = self.preview(file_id, session_id=session_id, max_chars=max_chars)
            return result if result else None

        # Fetch concurrently, preserve input order
        ordered: Dict[str, Optional[Dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_fetch, fid): fid for fid in file_ids}
            for future in as_completed(future_map):
                fid = future_map[future]
                ordered[fid] = future.result()

        return [ordered[fid] for fid in file_ids if ordered.get(fid)]

    def read_chunk(self, chunk_id: str) -> Dict:
        """Fetch the full content of a specific chunk by its ES _id.

        Primary method for targeted document reading after head() or search()
        has identified the relevant chunk.

        Returns:
            {
              "chunk_id":      str,
              "file_id":       str,
              "file_name":     str,
              "section_title": str,
              "content":       str,   # full chunk text
              "page_num":      int,
              "polarity":      int,
              "token_count":   int,
            }
        """
        resp = self.es.search(
            index=self.index,
            body={
                "query": {"term": {"_id": chunk_id}},
                "size": 1,
            },
        )
        src = resp["hits"]["hits"][0]["_source"]
        content = src.get("content", "")
        return {
            "chunk_id":      chunk_id,
            "file_id":       src.get("file_id", ""),
            "file_name":     src.get("file_name", ""),
            "section_title": src.get("section_title", ""),
            "content":       content,
            "page_num":      src.get("page_num", 0),
            "polarity":      src.get("polarity", 0),
            "token_count":   src.get("token_count", len(content)),
        }

    def raw(self, file_id: str, session_id: Optional[str] = None) -> Dict:
        """Get the complete document content from the full_text record.

        Use sparingly — the full text can be very large.
        Prefer read_chunk() for targeted reads.

        Returns:
            {
              "file_id":   str,
              "file_name": str,
              "content":   str,
            }
        """
        filters = self._build_filters(
            session_id=session_id,
            file_ids=[file_id],
            vectorization_method="full_text",
        )
        body = {
            "size": 1,
            "query": {"bool": {"filter": filters}},
            "_source": ["file_name", "file_id", "file_urls", "content"],
        }
        resp = self.es.search(index=self.index, body=body)
        src = resp["hits"]["hits"][0]["_source"]
        return {
            "file_id":   file_id,
            "file_name": src.get("file_name", ""),
            "content":   src.get("content", ""),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[List[float]]:
        """Embed a single text via the remote BGE service.

        Returns a list containing one vector (wraps get_embedding output).
        """
        return get_embedding([text])

    def _build_filters(
        self,
        session_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        vectorization_method: Optional[str] = None,
    ) -> List[Dict]:
        """Build Elasticsearch bool filter clauses."""
        filters: List[Dict] = []
        if session_id:
            filters.append({"term": {"session_id": session_id}})
        if file_ids:
            filters.append({"terms": {"file_id": file_ids}})
        if vectorization_method:
            filters.append({"term": {"vectorization_method": vectorization_method}})
        return filters

    def _format_search_response(self, resp: Dict) -> Dict:
        """Convert a raw ES response into the standard search result schema."""
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "chunk_id":      hit["_id"],
                "file_id":       src.get("file_id", ""),
                "file_name":     src.get("file_name", ""),
                "title":         src.get("title", ""),
                "section_title": src.get("section_title", ""),
                "content":       src.get("content", ""),
                "page_num":      src.get("page_num", 0),
                "polarity":      src.get("polarity", 0),
                "token_count":   src.get("token_count", 0),
                "score":         hit.get("_score") or 0.0,
            })
        return {
            "total":   resp["hits"]["total"]["value"],
            "results": results,
        }
