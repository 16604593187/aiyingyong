"""
Layer 9 - 抽取式内容压缩器

职责：对单个检索片段，根据当前 query 用 LLM 抽取相关句子，
      不改写，只删减。整个片段都不相关时返回空字符串。

只做压缩，不涉及检索和对话管理。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"

_config_module: Any | None = None
_openai_client: Any | None = None


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


# ── 压缩 prompt ────────────────────────────────────────────────
_COMPRESS_SYSTEM = (
    "你是一个信息抽取助手。"
    "任务：从给定的文本片段中，找出与【用户问题】直接相关的句子，"
    "原文照录，不改写，不添加任何内容。"
    "如果文本中没有任何与问题相关的句子，只输出：<无相关内容>"
    "不要解释，不要客套，直接输出抽取结果。"
)

_COMPRESS_USER_TMPL = (
    "【用户问题】\n{query}\n\n"
    "【文本片段】\n{passage}\n\n"
    "请抽取与问题相关的句子："
)

# 压缩调用使用较小的 max_tokens，只需要返回几句话
_COMPRESS_MAX_TOKENS = 300


def compress(query: str, passage: str) -> str:
    """
    对单个片段做抽取式压缩。

    参数：
        query  : 用户当前问题
        passage: 待压缩的检索片段原文

    返回：
        压缩后的文本（原文句子的子集），或空字符串（片段与 query 无关时）。
        返回空字符串表示该片段可以从注入列表中丢弃。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(passage, str) or not passage.strip():
        return ""

    client = _get_client()
    cfg    = _get_config()

    user_content = _COMPRESS_USER_TMPL.format(
        query=query.strip(),
        passage=passage.strip(),
    )

    response = client.chat.completions.create(
        model=cfg.OPENAI_API_MODEL,
        messages=[
            {"role": "system", "content": _COMPRESS_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,   # 抽取任务要确定性，不要随机性
        max_tokens=_COMPRESS_MAX_TOKENS,
        stream=False,
    )

    result = (response.choices[0].message.content or "").strip()

    # 模型返回"无相关内容"标记时视为空
    if "<无相关内容>" in result or result == "<无相关内容>":
        return ""

    return result


def compress_batch(
    query: str,
    passages: list[str],
) -> list[str]:
    """
    对多个片段顺序压缩，返回同等长度的结果列表。
    空字符串表示该片段被过滤掉。
    """
    return [compress(query, p) for p in passages]
