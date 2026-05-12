import os
from openai import OpenAI

key = os.getenv("EMBEDDING_API_KEY", "not-set")
print(f"key prefix: {key[:12]}...")

client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
resp = client.embeddings.create(
    model="text-embedding-v3",
    input="测试"
)
print(f"成功，向量维度: {len(resp.data[0].embedding)}")
