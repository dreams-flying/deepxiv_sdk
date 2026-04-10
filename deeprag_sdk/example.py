"""
DeepRAG SDK — quickstart example.

Demonstrates the full flow:
  1. Ingest a parsed document into Elasticsearch
  2. Query the agent: "2023年平安银行营业收入是多少？"
"""

from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel

from deeprag_sdk import Reader, Processor
from deeprag_sdk.agent import Agent


# ------------------------------------------------------------------
# 1. Setup: ES client + embedding model
# ------------------------------------------------------------------

ES_HOST = "http://localhost:9200"
INDEX_NAME = "my_internal_docs"

es = Elasticsearch(ES_HOST)
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

reader = Reader(es, index_name=INDEX_NAME, embed_model=model)
processor = Processor(es, index_name=INDEX_NAME, embed_model=model)


# ------------------------------------------------------------------
# 2. Ingest a document (your parser output format)
# ------------------------------------------------------------------

doc = {
    "type": "pdf",
    "title": "平安银行2023年年度报告.pdf",
    "url": "https://example.com/pingan_2023.pdf",
    "file_id": "pingan_annual_2023",
    "content": [
        {
            "index": 0,
            "type": "title",
            "text": "一、主要财务指标",
            "title": "一、主要财务指标",
            "para_num": 1,
            "page_num": 3,
        },
        {
            "index": 1,
            "type": "text",
            "text": (
                "2023年，平安银行实现营业收入1,646.99亿元，同比下降8.4%；"
                "净利润464.55亿元，同比增长2.1%。"
                "全年加权平均净资产收益率（ROE）为11.38%。"
            ),
            "title": "一、主要财务指标",
            "para_num": 2,
            "page_num": 3,
        },
        {
            "index": 2,
            "type": "title",
            "text": "二、资产负债情况",
            "title": "二、资产负债情况",
            "para_num": 3,
            "page_num": 5,
        },
        {
            "index": 3,
            "type": "text",
            "text": (
                "截至2023年末，平安银行总资产达5.59万亿元，较上年末增长2.7%；"
                "客户存款余额3.20万亿元，较上年末增长6.9%。"
            ),
            "title": "二、资产负债情况",
            "para_num": 4,
            "page_num": 5,
        },
    ],
    "pubtime": "2024-03-15 00:00:00",
}

user_id = "user_demo_001"

print("正在入库文档...")
chunk_count = processor.process(doc, user_id=user_id)
print(f"入库完成：{chunk_count} 个分块已写入 ES\n")


# ------------------------------------------------------------------
# 3. Query with the agent
# ------------------------------------------------------------------

agent = Agent(
    api_key="sk-your-api-key-here",   # 替换为你的 LLM API Key
    reader=reader,
    model="gpt-4o",
    # base_url="https://api.deepseek.com/v1",  # 使用 DeepSeek 等兼容接口
    print_process=True,               # 打印推理过程
)

answer = agent.query(
    question="2023年平安银行营业收入是多少？",
    user_id=user_id,
)

print("\n" + "=" * 60)
print("最终答案：")
print("=" * 60)
print(answer)


# ------------------------------------------------------------------
# 4. Follow-up query (documents cached, no re-fetch)
# ------------------------------------------------------------------

answer2 = agent.query(
    question="平安银行2023年净利润同比增长了多少？",
    user_id=user_id,
)

print("\n" + "=" * 60)
print("追问答案：")
print("=" * 60)
print(answer2)
