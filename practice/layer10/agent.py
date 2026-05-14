"""
Layer 10 - Agent 主循环（Function Calling）

职责：
- 定义工具 JSON Schema（模型通过描述决策是否调用）
- 实现工具调度（解析 tool_calls → 分发到 tools.py）
- 实现 Agent 主循环（messages → LLM → tool_calls? → 执行 → 追加 → LLM → 最终答案）

不负责：对话历史管理、语义缓存、终端 IO
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

# ── 路径 ──────────────────────────────────────────────────────
CURRENT_DIR  = Path(__file__).resolve().parent
PRACTICE_DIR = CURRENT_DIR.parent
LAYER1_DIR   = PRACTICE_DIR / "layer1"


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config_module: Any | None = None
_openai_client: Any | None = None
_tools_module: Any | None = None


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


def _get_tools_module() -> Any:
    global _tools_module
    if _tools_module is None:
        _tools_module = _load_module(CURRENT_DIR / "tools.py", "layer10_tools")
    return _tools_module


# ── 工具 JSON Schema ──────────────────────────────────────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "搜索本地知识库，获取关于 AI 应用开发、向量数据库、RAG、Embedding、"
                "Reranker、Agent、Function Calling 等主题的知识。"
                "当问题涉及这些主题的概念解释、原理说明、实验结果时使用。"
                "闲聊、纯计算、需要最新网络信息时不要使用。"
                "输入应为简洁精准的搜索关键词，不要原样传入用户的完整问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，简洁精准，如「BM25 参数 k1」「Reranker 原理」",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "搜索互联网获取实时信息。当问题涉及最新事件、时效性数据、"
                "或本地知识库中没有的内容时使用。返回若干条网页摘要和来源链接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，建议简洁精准",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": (
                "将内容写入笔记文件，保存到本地。"
                "适合保存总结、学习笔记、摘要等文字内容。"
                "只在用户明确要求保存或记录时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，不需要带 .txt 后缀",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入文件的完整文字内容",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# ── 工具调度 ──────────────────────────────────────────────────
def _dispatch_tool(name: str, arguments: dict) -> str:
    """根据工具名调用 tools.py 中对应的函数。"""
    if not isinstance(arguments, dict):
        return f"[tool_error] invalid arguments type for {name}: expected dict"

    tools = _get_tools_module()

    try:
        if name == "search_knowledge_base":
            return tools.search_knowledge_base(**arguments)
        elif name == "search_web":
            return tools.search_web(**arguments)
        elif name == "write_note":
            return tools.write_note(**arguments)
        else:
            return f"[tool_error] unknown tool: {name}"
    except Exception as exc:
        return f"[tool_error] {name} failed: {exc}"


# ── Agent 主循环 ──────────────────────────────────────────────
def run_agent(messages: list[dict], max_turns: int = 10) -> tuple[str, list[dict]]:
    """
    Function Calling Agent 主循环。

    参数：
        messages  : 完整的对话消息列表（含 system prompt + 历史 + 当前 user）
        max_turns : 最大工具调用轮次（防止死循环）

    返回：
        (answer, updated_messages)
        - answer: 模型最终生成的文字回复
        - updated_messages: 追加了本轮所有中间消息后的完整列表

    循环逻辑：
        1. 带 messages + TOOLS_SCHEMA 调用模型
        2. 有 tool_calls → 追加 assistant(tool_calls) + tool 结果 → 下一轮
        3. 无 tool_calls → 最终答案，退出循环
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    msgs   = list(messages)
    client = _get_client()
    cfg    = _get_config()

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=cfg.OPENAI_API_MODEL,
            messages=msgs,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_TOKENS,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            stream=False,
        )

        message    = response.choices[0].message
        tool_calls = message.tool_calls or []

        if tool_calls:
            # 追加含 tool_calls 的 assistant 消息
            assistant_msg: dict[str, Any] = {
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [],
            }
            for call in tool_calls:
                assistant_msg["tool_calls"].append({
                    "id":       call.id,
                    "type":     "function",
                    "function": {
                        "name":      call.function.name,
                        "arguments": call.function.arguments,
                    },
                })
            msgs.append(assistant_msg)

            # 逐个执行工具并追加结果
            for call in tool_calls:
                raw_args = call.function.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                    if not isinstance(parsed_args, dict):
                        raise ValueError("arguments must decode to an object")
                except Exception as exc:
                    tool_result = f"[tool_error] invalid arguments for {call.function.name}: {exc}"
                else:
                    tool_result = _dispatch_tool(call.function.name, parsed_args)

                msgs.append({
                    "role":         "tool",
                    "tool_call_id": call.id,
                    "content":      tool_result,
                })

            continue  # 下一轮

        # 无 tool_calls → 最终答案
        answer = (message.content or "").strip()
        msgs.append({"role": "assistant", "content": answer})
        return answer, msgs

    # 超过最大轮次
    timeout_msg = f"[agent] exceeded max tool-call turns ({max_turns}), aborted."
    msgs.append({"role": "assistant", "content": timeout_msg})
    return timeout_msg, msgs
