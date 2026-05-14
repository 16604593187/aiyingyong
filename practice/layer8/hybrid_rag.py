"""
Layer 8 - 混合检索 RAG 流水线

完整链路：
    query
      ├─ VectorRetriever.search(top_n=10)  ──┐
      └─ BM25Retriever.search(top_n=10)    ──┤
                                             ▼
                                      rrf_merge(top_m=15)
                                             ▼
                                      Reranker.rerank(top_k=5)
                                             ▼
                                      build_context → LLM → answer

对外暴露：
    warmup()                  预热所有模型和索引
    hybrid_search(query)      只做检索，返回精排后的片段列表
    ask(query, history)       完整 RAG 流程，返回 {"answer": ..., "sources": [...]}
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"

# ── 懒加载：避免 import 时触发模型初始化 ───────────────────────
_config_module: Any | None = None
_openai_client: Any | None = None
_vector_retriever: Any | None = None
_bm25_retriever:   Any | None = None
_reranker:         Any | None = None


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_config() -> Any:
    global _config_module
    if _config_module is None:
        _config_module = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")
    return _config_module


def _get_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from openai import OpenAI
    cfg = _get_config()
    _openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)
    return _openai_client


def _get_vector_retriever():
    global _vector_retriever
    if _vector_retriever is None:
        from retriever import VectorRetriever
        _vector_retriever = VectorRetriever()
    return _vector_retriever


def _get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        from retriever import BM25Retriever
        _bm25_retriever = BM25Retriever()
        # 构建索引：从向量库拉取所有文档
        vr = _get_vector_retriever()
        docs = vr.get_all_documents()
        _bm25_retriever.build_index(docs)
    return _bm25_retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from retriever import Reranker
        _reranker = Reranker()
    return _reranker


# ══════════════════════════════════════════════════════════════
# RRF 合并
# ══════════════════════════════════════════════════════════════
def rrf_merge(
    list_a: list[dict],
    list_b: list[dict],
    top_m: int = 15,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion：把两个排序列表合并成一个。

    公式：score(d) = Σ  1 / (k + rank_i(d))
    其中 rank 从 1 开始，d 不在某个列表里时不贡献分数。

    用文本内容作为文档的唯一标识（去重）。
    返回按 rrf_score 降序排列的 top_m 个文档。
    """
    scores: dict[str, float] = {}
    # 保留每个文档的原始 dict（第一次遇到时存）
    doc_store: dict[str, dict] = {}

    for rank, doc in enumerate(list_a, start=1):
        key = doc["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_store:
            doc_store[key] = doc

    for rank, doc in enumerate(list_b, start=1):
        key = doc["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_store:
            doc_store[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_m]

    merged: list[dict] = []
    for key in sorted_keys:
        doc = dict(doc_store[key])
        doc["rrf_score"] = scores[key]
        merged.append(doc)
    return merged


# ══════════════════════════════════════════════════════════════
# 预热
# ══════════════════════════════════════════════════════════════
def warmup() -> None:
    """
    预热：按顺序初始化所有重型资源，避免第一次查询时的冷启动卡顿。
    启动时调用一次即可。
    """
    print("  [warmup] 加载向量检索器...", flush=True)
    vr = _get_vector_retriever()
    # 触发 Embedding 模型和 Chroma 初始化
    kb = vr._kb_module if hasattr(vr, "_kb_module") else None
    # 通过一次 search 触发懒加载
    _get_vector_retriever().search("warmup", top_n=1)

    print("  [warmup] 构建 BM25 索引...", flush=True)
    _get_bm25_retriever()  # build_index 在内部完成

    print("  [warmup] 加载 Reranker 模型...", flush=True)
    _get_reranker()

    print("  [warmup] 完成\n", flush=True)


# ══════════════════════════════════════════════════════════════
# 混合检索主入口
# ══════════════════════════════════════════════════════════════
def hybrid_search(
    query: str,
    vector_top_n: int = 10,
    bm25_top_n: int = 10,
    rrf_top_m: int = 15,
    rerank_top_k: int = 5,
) -> list[dict]:
    """
    完整的混合检索 + Reranker 精排流程。

    返回精排后的 top_k 个文档列表，每个文档包含：
        text, metadata, score（原始向量/BM25分），
        rrf_score, rerank_score
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    q = query.strip()

    # 1. 向量检索
    vector_results = _get_vector_retriever().search(q, top_n=vector_top_n)

    # 2. BM25 检索
    bm25_results = _get_bm25_retriever().search(q, top_n=bm25_top_n)

    # 3. RRF 合并
    candidates = rrf_merge(vector_results, bm25_results, top_m=rrf_top_m)

    if not candidates:
        return []

    # 4. Reranker 精排
    final = _get_reranker().rerank(q, candidates, top_k=rerank_top_k)
    return final


# ══════════════════════════════════════════════════════════════
# Prompt 组装
# ══════════════════════════════════════════════════════════════
def _build_context(hits: list[dict], max_chars: int = 3000) -> str:
    used = 0
    sections: list[str] = []

    for hit in hits:
        text = (hit.get("text") or "").strip()
        if not text:
            continue

        metadata = hit.get("metadata") or {}
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "na")
        rerank_score = hit.get("rerank_score", hit.get("rrf_score", 0.0))

        block = (
            f"[source={source}, chunk={chunk_index}, rerank={rerank_score:.4f}]\n"
            f"{text}"
        )

        if used + len(block) > max_chars and sections:
            break

        sections.append(block)
        used += len(block)

    return "\n\n".join(sections)


# ══════════════════════════════════════════════════════════════
# 完整 RAG 问答
# ══════════════════════════════════════════════════════════════
def ask(
    query: str,
    history: list[dict] | None = None,
    rerank_top_k: int = 5,
) -> dict:
    """
    混合检索 RAG 完整流程。

    参数：
        query      : 用户当前问题
        history    : 对话历史（list of {"role": ..., "content": ...}），可为 None
        rerank_top_k: 注入 prompt 的最终片段数

    返回：
        {
            "answer":  "模型生成的回答",
            "sources": [{"text": ..., "source": ..., "rerank_score": ...}, ...]
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    q = query.strip()
    hits = hybrid_search(q, rerank_top_k=rerank_top_k)

    sources: list[dict] = []
    for hit in hits:
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        metadata = hit.get("metadata") or {}
        sources.append({
            "text":         text,
            "source":       str(metadata.get("source", "unknown")),
            "rerank_score": float(hit.get("rerank_score", 0.0)),
        })

    context = _build_context(hits) if hits else ""

    cfg    = _get_config()
    client = _get_client()

    system_content = cfg.SYSTEM_PROMPT
    if context:
        system_content += (
            "\n\n以下是从知识库检索到的相关片段，请优先基于这些内容回答，"
            "知识库中没有的内容可结合自身知识补充，但需说明。\n\n"
            f"{context}"
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": q})

    response = client.chat.completions.create(
        model=cfg.OPENAI_API_MODEL,
        messages=messages,
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_TOKENS,
        stream=False,
    )

    answer = (response.choices[0].message.content or "").strip()
    return {"answer": answer, "sources": sources}
