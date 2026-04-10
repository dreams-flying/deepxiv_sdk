"""
DeepRAG SDK - Agent-first RAG system for internal document retrieval.

Mirrors DeepXiv's progressive disclosure philosophy:
  search → head (chunk TLDRs) → read_chunk → raw (full doc)

Usage:
    from deeprag_sdk import Reader, Processor
    from deeprag_sdk.agent import Agent
    from elasticsearch import Elasticsearch
    from FlagEmbedding import BGEM3FlagModel

    es = Elasticsearch("http://localhost:9200")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    reader = Reader(es, index_name="my_docs", embed_model=model)
    processor = Processor(es, index_name="my_docs", embed_model=model)

    # Ingest a document
    processor.process(doc, user_id="user_001")

    # Query with agent
    agent = Agent(api_key="sk-...", reader=reader)
    answer = agent.query("2023年平安银行营业收入是多少？", user_id="user_001")
"""

from .reader import Reader
from .processor import Processor

__version__ = "0.1.0"
__all__ = ["Reader", "Processor"]
