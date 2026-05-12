import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2"))
import embeddings

tests = [
    "你好",
    "RAG检索增强生成是什么",
    "向量数据库和普通数据库的区别是什么，请详细解释一下它们在存储结构、查询方式、适用场景上的不同",
]

# 预热（第一次加载模型会慢）
embeddings.get_embedding("warmup")

for text in tests:
    t0 = time.perf_counter()
    embeddings.get_embedding(text)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"{elapsed:6.1f}ms | {text[:30]}")
