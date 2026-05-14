"""
Layer 10 - 工具实现层

三个工具，每个函数只负责执行，不涉及任何 LLM 调用（知识库压缩除外）：
- search_knowledge_base(query) : 混合检索 + Reranker + 内容压缩，返回精简片段
- search_web(query)            : Tavily 联网搜索，返回摘要文本
- write_note(filename, content): 写笔记到 layer10/notes/
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── 路径定义 ──────────────────────────────────────────────────
CURRENT_DIR  = Path(__file__).resolve().parent
PRACTICE_DIR = CURRENT_DIR.parent
LAYER1_DIR   = PRACTICE_DIR / "layer1"
LAYER8_DIR   = PRACTICE_DIR / "layer8"
LAYER9_DIR   = PRACTICE_DIR / "layer9"

NOTES_DIR = CURRENT_DIR / "notes"

# 把 layer8、layer9 加入 sys.path（hybrid_rag、compressor 需要）
for d in (str(LAYER8_DIR), str(LAYER9_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 懒加载模块 ────────────────────────────────────────────────
_config_module: Any | None = None
_hybrid_rag: Any | None = None
_compressor: Any | None = None


def _get_config() -> Any:
    global _config_module
    if _config_module is None:
        _config_module = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")
    return _config_module


def _get_hybrid_rag() -> Any:
    global _hybrid_rag
    if _hybrid_rag is None:
        _hybrid_rag = _load_module(LAYER8_DIR / "hybrid_rag.py", "layer8_hybrid_rag")
    return _hybrid_rag


def _get_compressor() -> Any:
    global _compressor
    if _compressor is None:
        _compressor = _load_module(LAYER9_DIR / "compressor.py", "layer9_compressor")
    return _compressor


# ── 预热（Agent 启动时调用一次）──────────────────────────────
def warmup() -> None:
    """预热所有检索相关模型（Embedding + BM25 索引 + Reranker）。"""
    hybrid_rag = _get_hybrid_rag()
    hybrid_rag.warmup()


# ── 工具 1：知识库检索（混合检索 + Reranker + 内容压缩）─────
def search_knowledge_base(query: str) -> str:
    """
    搜索本地知识库。

    内部链路：向量+BM25 混合检索 → RRF 合并 → Reranker 精排 top-5
              → 逐片段 LLM 抽取压缩 → 过滤无关片段 → 拼接返回

    参数：
        query: 搜索关键词或问题，应简洁精准

    返回：
        格式化的检索结果文本，每个片段标注来源和相关度分数。
        如果没有检索到相关内容，返回提示文字。
    """
    if not isinstance(query, str) or not query.strip():
        return "[tool_error] query must be a non-empty string"

    q = query.strip()

    try:
        hybrid_rag = _get_hybrid_rag()
        compressor = _get_compressor()

        # 混合检索 + Reranker
        hits = hybrid_rag.hybrid_search(q, rerank_top_k=5)
        if not hits:
            return "知识库中未检索到相关内容。"

        # 逐片段压缩
        results: list[str] = []
        for hit in hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            compressed = compressor.compress(q, text)
            if not compressed:
                continue  # 该片段与 query 无关，过滤掉

            metadata = hit.get("metadata") or {}
            source = metadata.get("source", "unknown")
            rerank_score = float(hit.get("rerank_score", 0.0))

            results.append(
                f"[来源: {source}, 相关度: {rerank_score:.3f}]\n{compressed}"
            )

        if not results:
            return "知识库中检索到的内容与问题相关度不高，无有效片段。"

        return "\n\n---\n\n".join(results)

    except Exception as exc:
        return f"[tool_error] search_knowledge_base failed: {exc}"


# ── 工具 2：联网搜索 ──────────────────────────────────────────
def search_web(query: str) -> str:
    """
    使用 Tavily API 搜索网页，返回摘要文本。
    需要在 .env 中配置 TAVILY_API_KEY。

    返回格式：
        [1] 标题\n内容摘要\n来源: URL\n\n[2] ...
    """
    if not isinstance(query, str) or not query.strip():
        return "[tool_error] query must be a non-empty string"

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "[tool_error] TAVILY_API_KEY is not set"

    try:
        from tavily import TavilyClient
    except ImportError:
        return "[tool_error] tavily-python package is required"

    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(query.strip(), max_results=3)

        items = result.get("results", []) if isinstance(result, dict) else []
        if not items:
            return "未检索到相关网页结果。"

        blocks: list[str] = []
        for index, item in enumerate(items, start=1):
            title   = str(item.get("title", "无标题")).strip() or "无标题"
            content = str(item.get("content", "")).strip() or "无摘要"
            url     = str(item.get("url", "")).strip() or "无链接"
            blocks.append(f"[{index}] {title}\n{content}\n来源: {url}")

        return "\n\n".join(blocks)

    except Exception as exc:
        return f"[tool_error] search_web failed: {exc}"


# ── 工具 3：写笔记 ────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("filename must be a string")
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-.]", "_", name.strip())
    return cleaned or "untitled"


def write_note(filename: str, content: str) -> str:
    """
    将 content 写入 layer10/notes/<filename>.txt。
    filename 不需要带 .txt 后缀，函数会自动添加。
    目录不存在时自动创建。

    返回：成功写入的文件绝对路径字符串。
    """
    if not isinstance(content, str) or not content.strip():
        return "[tool_error] content must be a non-empty string"

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    if not safe_name.lower().endswith(".txt"):
        safe_name = f"{safe_name}.txt"

    file_path = NOTES_DIR / safe_name
    file_path.write_text(content, encoding="utf-8")
    return f"笔记已保存到: {file_path.resolve()}"
