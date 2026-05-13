"""
Layer 7 - Function Calling 核心循环

职责：
- 定义三个工具的 JSON Schema（告诉模型有哪些工具可以用）
- 实现工具调度：解析模型返回的 tool_calls，分发到 tools.py 的实际函数
- 实现 Agent 主循环：messages -> LLM -> (tool_calls?) -> 执行 -> 追加结果 -> LLM -> 最终答案

不负责：对话历史管理（由 chat_agent.py 负责）、终端 IO
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

# ── 路径：复用 layer1/config_mine.py ─────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"


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
        _tools_module = _load_module(CURRENT_DIR / "tools.py", "layer7_tools")
    return _tools_module


# ── 工具 JSON Schema 定义 ─────────────────────────────────────
# 这里的描述直接影响模型判断"该不该调用这个工具、传什么参数"
# 描述要清晰准确，模型看不到函数代码，只看这里

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "搜索互联网获取实时信息。当问题涉及最新事件、时效性数据、"
                "或知识库中没有的内容时使用。返回若干条网页摘要和来源链接。"
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
                "将内容写入笔记文件，保存到 layer7/test/notes/ 目录。"
                "适合保存总结、学习笔记、摘要等文字内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，不需要带 .txt 后缀，例如：Python学习笔记",
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
    {
        "type": "function",
        "function": {
            "name": "save_quiz",
            "description": (
                "将题目内容写入文件，保存到 layer7/test/exam/ 目录。"
                "适合保存练习题、测验题、问答题等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，不需要带 .txt 后缀，例如：Python基础练习题",
                    },
                    "content": {
                        "type": "string",
                        "description": "题目的完整内容，包含题目和答案",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# ── 工具调度 ──────────────────────────────────────────────────
def _dispatch_tool(name: str, arguments: dict) -> str:
    """
    根据工具名调用 tools.py 中对应的函数，返回字符串结果。
    模型传来的 arguments 已经是 dict，直接 ** 展开传参。
    """
    if not isinstance(arguments, dict):
        return f"[tool_error] invalid arguments type for {name}: expected dict"

    tools_module = _get_tools_module()

    try:
        if name == "search_web":
            result = tools_module.search_web(**arguments)
        elif name == "write_note":
            result = tools_module.write_note(**arguments)
        elif name == "save_quiz":
            result = tools_module.save_quiz(**arguments)
        else:
            return f"[tool_error] unknown tool: {name}"
    except Exception as exc:  # pragma: no cover - depends on tool runtime
        return f"[tool_error] {name} failed: {exc}"

    return str(result)


# ── Agent 主循环 ──────────────────────────────────────────────
def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Function Calling 主循环。

    参数：
        messages: 完整的对话历史（含 system prompt），由调用方传入

    返回：
        (answer, updated_messages)
        - answer: 模型最终生成的文字回复
        - updated_messages: 追加了本轮 tool_calls / tool 结果 / assistant 回复后的完整列表
          调用方负责把这个列表存回记忆

    循环逻辑：
        1. 带着 messages 和 TOOLS_SCHEMA 调用模型
        2. 如果模型返回了 tool_calls：
               a. 把含 tool_calls 的 assistant 消息追加到 messages
               b. 遍历每个 tool_call，调用 _dispatch_tool，把结果以 role=tool 追加
               c. 回到第 1 步（继续循环）
        3. 如果模型没有返回 tool_calls，说明已经生成了最终答案，退出循环
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    msgs = list(messages)
    client = _get_client()
    cfg = _get_config()

    max_turns = 10
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=cfg.OPENAI_API_MODEL,
            messages=msgs,
            temperature=cfg.TEMPERATURE,
            max_tokens=cfg.MAX_TOKENS,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            stream=False,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [],
            }

            for call in tool_calls:
                assistant_msg["tool_calls"].append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )

            msgs.append(assistant_msg)

            for call in tool_calls:
                raw_arguments = call.function.arguments or "{}"
                try:
                    parsed_arguments = json.loads(raw_arguments)
                    if not isinstance(parsed_arguments, dict):
                        raise ValueError("arguments must decode to an object")
                except Exception as exc:
                    tool_result = (
                        f"[tool_error] invalid arguments for {call.function.name}: {exc}"
                    )
                else:
                    tool_result = _dispatch_tool(call.function.name, parsed_arguments)

                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )

            continue

        answer = (message.content or "").strip()
        msgs.append({"role": "assistant", "content": answer})
        return answer, msgs

    timeout_message = f"[agent] exceeded max tool-call turns ({max_turns}), aborted."
    msgs.append({"role": "assistant", "content": timeout_message})
    return timeout_message, msgs
