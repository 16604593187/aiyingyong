"""
Layer 7 - 工具实现层

三个工具，每个函数只负责执行，不涉及任何 LLM 调用：
- search_web(query)         : Tavily 联网搜索，返回摘要文本
- write_note(filename, content) : 写笔记到 test/notes/
- save_quiz(filename, content)  : 保存题目到 test/exam/
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ── 路径定义 ──────────────────────────────────────────────────
# __file__ = practice/layer7/tools.py
# BASE_DIR = practice/
BASE_DIR = Path(__file__).resolve().parent.parent
NOTES_DIR = BASE_DIR / "layer7" / "test" / "notes"
EXAM_DIR = BASE_DIR / "layer7" / "test" / "exam"


def _safe_filename(name: str) -> str:
    """
    清洗文件名，只保留字母、数字、中文、下划线、连字符和点。
    防止模型生成的文件名包含斜杠、空格等危险字符。
    """
    if not isinstance(name, str):
        raise ValueError("filename must be a string")

    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-.]", "_", name.strip())
    return cleaned or "untitled"


def _normalize_txt_filename(filename: str) -> str:
    safe_name = _safe_filename(filename)
    if safe_name.lower().endswith(".txt"):
        return safe_name
    return f"{safe_name}.txt"


def _write_text_file(target_dir: Path, filename: str, content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("content must be a string")

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / _normalize_txt_filename(filename)
    file_path.write_text(content, encoding="utf-8")
    return str(file_path.resolve())


# ── 工具 1：联网搜索 ──────────────────────────────────────────
def search_web(query: str) -> str:
    """
    使用 Tavily API 搜索网页，返回摘要文本。
    需要在 .env 中配置 TAVILY_API_KEY。

    返回格式示例：
        [1] 标题\n内容摘要\n来源: URL\n\n[2] ...
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("tavily-python package is required. Please install it first.") from exc

    client = TavilyClient(api_key=api_key)
    result = client.search(query.strip(), max_results=3)

    items = result.get("results", []) if isinstance(result, dict) else []
    if not items:
        return "未检索到相关网页结果。"

    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        title = str(item.get("title", "无标题")).strip() or "无标题"
        content = str(item.get("content", "")).strip() or "无摘要"
        url = str(item.get("url", "")).strip() or "无链接"
        blocks.append(f"[{index}] {title}\n{content}\n来源: {url}")

    return "\n\n".join(blocks)


# ── 工具 2：写笔记 ────────────────────────────────────────────
def write_note(filename: str, content: str) -> str:
    """
    将 content 写入 layer7/test/notes/<filename>.txt。
    filename 不需要带 .txt 后缀，函数会自动添加。
    目录不存在时自动创建。

    返回：成功写入的文件绝对路径字符串。
    """
    return _write_text_file(NOTES_DIR, filename, content)


# ── 工具 3：保存题目 ──────────────────────────────────────────
def save_quiz(filename: str, content: str) -> str:
    """
    将 content 写入 layer7/test/exam/<filename>.txt。
    filename 不需要带 .txt 后缀，函数会自动添加。
    目录不存在时自动创建。

    返回：成功写入的文件绝对路径字符串。
    """
    return _write_text_file(EXAM_DIR, filename, content)
