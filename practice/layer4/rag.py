from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR = CURRENT_DIR.parent / "layer1"
LAYER3_DIR = CURRENT_DIR.parent / "layer3"

_knowledge_base_module: Any | None = None
_config_module: Any | None = None
_openai_client: Any | None = None


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_knowledge_base() -> Any:
    global _knowledge_base_module
    if _knowledge_base_module is None:
        _knowledge_base_module = _load_module(
            LAYER3_DIR / "knowledge_base.py",
            "layer3_knowledge_base",
        )
    return _knowledge_base_module


def _get_config() -> Any:
    global _config_module
    if _config_module is None:
        _config_module = _load_module(
            LAYER1_DIR / "config_mine.py",
            "layer1_config_mine",
        )
    return _config_module


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("openai package is required for rag query") from exc

    config = _get_config()
    _openai_client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )
    return _openai_client


def warmup() -> None:
    """预热：提前加载 Embedding 模型和向量库，避免第一次查询时的冷启动延迟"""
    kb = _get_knowledge_base()
    _get_openai_client()

    # 先触发 Chroma collection初始化
    kb.count()

    # 再触发 Embedding 模型实际 encode，避免首次查询卡顿
    embedding_utils = kb._get_embedding_utils()
    embedding_utils.get_embedding("warmup")


def retrieve(question: str, top_k: int = 5) -> str:
    """只做检索，返回可直接拼接到提示词中的上下文片段。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        return ""

    knowledge_base = _get_knowledge_base()
    hits = knowledge_base.search(question.strip(), top_k=top_k)
    if not hits:
        return ""

    return _build_context(hits)


def _build_context(hits: list[dict], max_context_chars: int = 3000) -> str:
    used = 0
    sections: list[str] = []

    for hit in hits:
        text = (hit.get("text") or "").strip()
        if not text:
            continue

        metadata = hit.get("metadata") or {}
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "na")
        score = float(hit.get("score", 0.0))

        block = (
            f"[source={source}, chunk={chunk_index}, score={score:.4f}]\n"
            f"{text}"
        )

        if used + len(block) > max_context_chars and sections:
            break

        sections.append(block)
        used += len(block)

    return "\n\n".join(sections)


def query(user_input: str, top_k: int = 5) -> dict:
    """
    完整的 RAG 流程：检索 + 生成
    返回格式：
    {
        "answer": "模型生成的答案",
        "sources": [
            {"text": "片段内容", "source": "文件名", "score": 0.85},
            ...
        ]
    }
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")
    if top_k <= 0:
        return {"answer": "", "sources": []}

    question = user_input.strip()
    knowledge_base = _get_knowledge_base()
    hits = knowledge_base.search(question, top_k=top_k)

    sources: list[dict] = []
    for item in hits:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        metadata = item.get("metadata") or {}
        sources.append(
            {
                "text": text,
                "source": str(metadata.get("source", "unknown")),
                "score": float(item.get("score", 0.0)),
            }
        )

    if not sources:
        return {
            "answer": "知识库中没有检索到足够相关的信息。",
            "sources": [],
        }

    context = _build_context(hits)
    config = _get_config()
    client = _get_openai_client()

    system_prompt = (
        "你是一个严格基于知识库片段回答问题的助手。"
        "如果证据不足，明确说不知道。不要编造来源。"
    )

    user_prompt = (
        f"用户问题：{question}\n\n"
        f"知识库片段：\n{context}\n\n"
        "请基于以上片段给出简洁、准确的答案。"
        "若片段不足以支持结论，请直接说明。"
    )

    response = client.chat.completions.create(
        model=config.OPENAI_API_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=config.MAX_TOKENS,
        stream=False,
    )

    answer = response.choices[0].message.content or ""
    return {
        "answer": answer.strip(),
        "sources": sources,
    }
