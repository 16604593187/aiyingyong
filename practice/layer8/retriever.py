"""
Layer 8 - 检索组件

三个独立的检索/精排模块，每个模块只做自己的事：

  VectorRetriever   : 复用 layer3/knowledge_base.py，向量检索返回 top-n 结果
  BM25Retriever     : 基于 rank-bm25 + jieba 分词的关键词检索
  Reranker          : 基于 bge-reranker-base 的 Cross-Encoder 精排

混合检索的 RRF 合并逻辑在 hybrid_rag.py 中，这里只负责各自的检索/打分。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER3_DIR  = CURRENT_DIR.parent / "layer3"

# ── 工具：动态加载模块 ──────────────────────────────────────────
def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════
# 1. 向量检索器
# ═══════════════════════════════════════════════════════════════
_kb_module: Any | None = None


def _get_kb() -> Any:
    global _kb_module
    if _kb_module is None:
        _kb_module = _load_module(LAYER3_DIR / "knowledge_base.py", "layer3_kb")
    return _kb_module


class VectorRetriever:
    """封装 layer3 knowledge_base.search，返回统一格式的结果列表。"""

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """
        返回格式：
            [{"text": "...", "metadata": {...}, "score": 0.85}, ...]
        score 是余弦相似度，越高越好。
        """
        if not query.strip():
            return []
        kb = _get_kb()
        return kb.search(query.strip(), top_k=top_n)

    def get_all_documents(self) -> list[dict]:
        """
        取出知识库中所有文档，供 BM25Retriever 构建索引使用。
        返回格式：[{"text": "...", "metadata": {...}}, ...]
        """
        kb = _get_kb()
        collection = kb._get_collection()
        total = collection.count()
        if total == 0:
            return []

        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        output: list[dict] = []
        for text, meta in zip(documents, metadatas):
            if text and text.strip():
                output.append({"text": text.strip(), "metadata": meta or {}})
        return output


# ═══════════════════════════════════════════════════════════════
# 2. BM25 检索器
# ═══════════════════════════════════════════════════════════════
class BM25Retriever:
    """
    基于 rank-bm25 + jieba 的关键词检索器。

    使用前需要调用 build_index(documents) 构建索引，
    documents 格式和 VectorRetriever.get_all_documents() 的返回一致。

    BM25 索引是内存结构，每次程序启动都需要重建。
    """

    def __init__(self) -> None:
        self._bm25: Any | None = None
        self._documents: list[dict] = []

    def build_index(self, documents: list[dict]) -> None:
        """
        对文档列表分词后构建 BM25 索引。
        documents: [{"text": "...", "metadata": {...}}, ...]
        """
        if not documents:
            self._bm25 = None
            self._documents = []
            return

        try:
            import jieba
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "jieba and rank-bm25 are required. "
                "Run: pip install jieba rank-bm25"
            ) from exc

        # jieba 默认会打印初始化日志，静默模式
        jieba.setLogLevel("ERROR")

        self._documents = documents
        tokenized_corpus = [
            list(jieba.cut(doc["text"]))
            for doc in documents
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """
        返回格式：
            [{"text": "...", "metadata": {...}, "score": float}, ...]
        score 是 BM25 原始分数（非归一化），只用于内部排序，不跨检索器比较。
        """
        if self._bm25 is None or not self._documents:
            return []
        if not query.strip():
            return []

        try:
            import jieba
        except ImportError as exc:
            raise RuntimeError("jieba is required") from exc

        jieba.setLogLevel("ERROR")
        query_tokens = list(jieba.cut(query.strip()))
        scores = self._bm25.get_scores(query_tokens)

        # 按分数降序取 top_n
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_n]

        results: list[dict] = []
        for idx, score in indexed_scores:
            if score <= 0:
                # BM25 分数为 0 表示完全不相关，跳过
                continue
            doc = self._documents[idx]
            results.append({
                "text":     doc["text"],
                "metadata": doc["metadata"],
                "score":    float(score),
            })
        return results

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._documents) > 0


# ═══════════════════════════════════════════════════════════════
# 3. Reranker（Cross-Encoder 精排）
# ═══════════════════════════════════════════════════════════════
_RERANKER_MODEL = "BAAI/bge-reranker-base"

_reranker_instance: Any | None = None


def _get_reranker() -> Any:
    global _reranker_instance
    if _reranker_instance is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Reranker. "
                "Run: pip install sentence-transformers"
            ) from exc
        _reranker_instance = CrossEncoder(_RERANKER_MODEL)
    return _reranker_instance


class Reranker:
    """
    使用 bge-reranker-base（Cross-Encoder）对候选片段精排。

    输入：query + 候选文档列表（同 VectorRetriever/BM25Retriever 的返回格式）
    输出：按相关性降序排列后的文档列表，新增 rerank_score 字段
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        对 candidates 精排，返回 top_k 个结果。

        返回格式：
            [{"text": "...", "metadata": {...}, "score": 原始分, "rerank_score": float}, ...]
        rerank_score 是 Cross-Encoder 的 logit，越高越相关。
        """
        if not candidates or not query.strip():
            return candidates[:top_k]

        reranker = _get_reranker()

        pairs = [(query.strip(), doc["text"]) for doc in candidates]
        scores = reranker.predict(pairs)

        scored = [
            {**doc, "rerank_score": float(score)}
            for doc, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
