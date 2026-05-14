"""
Layer 9 - 带内容压缩的 RAG 流水线

在 Layer 8 的混合检索 + Reranker 基础上，增加 Contextual Compression 步骤：
    混合检索 + Reranker（复用 layer8）
        → 逐片段 LLM 抽取压缩
        → 过滤空结果
        → 拼接注入 prompt
        → LLM 生成答案

对外暴露：
    warmup()   : 预热（调用 layer8 warmup）
    ask(query, history)  : 完整压缩 RAG 流程
        返回 {"answer": ..., "sources": [...], "compression_stats": {...}}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"
LAYER8_DIR  = CURRENT_DIR.parent / "layer8"

# layer8 和 layer9 都需要在 sys.path 里
for d in (str(LAYER8_DIR), str(CURRENT_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config_module: Any | None = None
_openai_client: Any | None = None


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


# ── 预热 ───────────────────────────────────────────────────────
def warmup() -> None:
    """预热 layer8 的所有模型和索引（向量检索 + BM25 + Reranker）。"""
    from hybrid_rag import warmup as layer8_warmup
    layer8_warmup()


# ── 压缩后的 context 拼接 ──────────────────────────────────────
def _build_compressed_context(
    compressed_hits: list[dict],
    max_chars: int = 2000,
) -> str:
    """
    把压缩后的片段拼接成注入 prompt 的 context 字符串。
    compressed_hits 里每个 dict 包含 compressed_text、metadata、rerank_score 等。
    """
    used = 0
    sections: list[str] = []

    for hit in compressed_hits:
        text = (hit.get("compressed_text") or "").strip()
        if not text:
            continue

        metadata     = hit.get("metadata") or {}
        source       = metadata.get("source", "unknown")
        chunk_index  = metadata.get("chunk_index", "na")
        rerank_score = hit.get("rerank_score", 0.0)

        block = (
            f"[source={source}, chunk={chunk_index}, rerank={rerank_score:.4f}]\n"
            f"{text}"
        )

        if used + len(block) > max_chars and sections:
            break

        sections.append(block)
        used += len(block)

    return "\n\n".join(sections)


# ── 完整流水线 ─────────────────────────────────────────────────
def ask(
    query: str,
    history: list[dict] | None = None,
    rerank_top_k: int = 5,
) -> dict:
    """
    带内容压缩的完整 RAG 流程。

    返回：
    {
        "answer": "模型生成的回答",
        "sources": [{"text": 原始片段, "compressed_text": 压缩后, "source": ...,
                     "rerank_score": ...}, ...],
        "compression_stats": {
            "total_original_chars": int,   # 压缩前总字符数
            "total_compressed_chars": int, # 压缩后总字符数
            "filtered_count": int,         # 被完全过滤掉的片段数
            "compression_ratio": float,    # 压缩率（越低说明压缩越多）
        }
    }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    q = query.strip()

    # 1. 混合检索 + Reranker（复用 layer8）
    from hybrid_rag import hybrid_search
    hits = hybrid_search(q, rerank_top_k=rerank_top_k)

    if not hits:
        return {
            "answer": "知识库中没有检索到足够相关的信息。",
            "sources": [],
            "compression_stats": {
                "total_original_chars": 0,
                "total_compressed_chars": 0,
                "filtered_count": 0,
                "compression_ratio": 1.0,
            },
        }

    # 2. 逐片段压缩
    from compressor import compress

    sources: list[dict] = []
    filtered_count = 0
    total_original_chars = 0
    total_compressed_chars = 0

    for hit in hits:
        original_text = (hit.get("text") or "").strip()
        if not original_text:
            continue

        metadata     = hit.get("metadata") or {}
        rerank_score = float(hit.get("rerank_score", 0.0))

        compressed_text = compress(q, original_text)
        total_original_chars   += len(original_text)
        total_compressed_chars += len(compressed_text)

        if not compressed_text:
            filtered_count += 1

        sources.append({
            "text":            original_text,
            "compressed_text": compressed_text,
            "source":          str(metadata.get("source", "unknown")),
            "rerank_score":    rerank_score,
        })

    # 3. 只取有压缩内容的片段构建 context
    compressed_hits = [
        {**s, "metadata": hits[i].get("metadata", {})}
        for i, s in enumerate(sources)
        if s["compressed_text"]
    ]

    compression_ratio = (
        total_compressed_chars / total_original_chars
        if total_original_chars > 0
        else 1.0
    )

    compression_stats = {
        "total_original_chars":   total_original_chars,
        "total_compressed_chars": total_compressed_chars,
        "filtered_count":         filtered_count,
        "compression_ratio":      round(compression_ratio, 3),
    }

    context = _build_compressed_context(compressed_hits)

    # 4. 组装 prompt，调用主 LLM 生成答案
    cfg    = _get_config()
    client = _get_client()

    system_content = cfg.SYSTEM_PROMPT
    if context:
        system_content += (
            "\n\n以下是从知识库检索并提炼后的相关内容，请优先基于这些内容回答，"
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

    return {
        "answer":            answer,
        "sources":           sources,
        "compression_stats": compression_stats,
    }
