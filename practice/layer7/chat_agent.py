"""
Layer 7 - Agent 对话终端

职责：
- 会话管理：session 选择、history 加载与持久化（复用 layer5/memory.py）
- 对话循环：读取用户输入 → 构建 messages → 调用 run_agent → 打印结果 → 保存历史
- 终端命令：/history / /clear / /exit

与前几层的主要区别：
- 不再手动判断 intent / cache，由 Agent 自己决定调用哪些工具
- history 中会包含 tool_calls / tool 消息，/history 命令需要能正确展示
- messages 的构建包含 system prompt + 历史（含工具消息）+ 当前 user 问题
  注意：tool_calls / tool 消息不能截断，必须成对保留，否则 API 报错
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"
LAYER5_DIR  = CURRENT_DIR.parent / "layer5"


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 复用模块加载 ──────────────────────────────────────────────
config   = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")
mem_mod  = _load_module(LAYER5_DIR / "memory.py",      "layer5_memory")

MemoryManager = mem_mod.MemoryManager

# layer7/agent.py 在同目录，直接 import
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from agent import run_agent  # noqa: E402


# ── 记忆目录：layer7/history/ ─────────────────────────────────
MEMORY_DIR = CURRENT_DIR / "history"


# ── history 截断：保留工具消息的完整对 ────────────────────────
def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """
    截断历史，保留最近 max_turns 轮对话。

    和前几层不同的地方：引入 Function Calling 后，一轮对话不再是简单的
    user + assistant 两条消息，可能是：
        user → assistant(tool_calls) → tool → tool → assistant(最终回复)
    如果从中间截断，tool_calls 消息和 tool 消息不成对，API 会报错。

    安全的截断策略：只在 role=user 的消息处截断，确保每段历史都从
    user 开始，tool_calls / tool 消息组始终完整保留。

    实现思路：
    1. 找出所有 role=user 的消息下标，构成截断点列表
    2. 如果截断点数量 > max_turns，取倒数第 max_turns 个截断点的位置
    3. 从该位置开始切片
    """
    if max_turns <= 0:
        return []

    user_indices = [idx for idx, msg in enumerate(history) if msg.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return history

    start_index = user_indices[len(user_indices) - max_turns]
    return history[start_index:]


# ── 构建发送给 API 的 messages ────────────────────────────────
def build_messages(history: list[dict], question: str) -> list[dict]:
    """
    组装本轮发送给 run_agent 的完整 messages 列表。
    结构：system prompt + 截断后的历史 + 当前 user 问题

    注意：历史里已包含了 tool_calls / tool 消息，直接 extend 进去即可。
    当前 user 问题单独 append，不提前写入 history（等 agent 跑完再存）。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    system_msg = {"role": "system", "content": config.SYSTEM_PROMPT}
    trimmed_history = trim_history(history, config.MAX_HISTORY_TURNS)

    messages = [system_msg]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question.strip()})
    return messages


# ── 打印历史（兼容 tool 消息）────────────────────────────────
def print_history(history: list[dict]) -> None:
    if not history:
        print("\n[对话历史为空]\n")
        return

    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 40)
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        if role == "user":
            content = str(msg.get("content", ""))
            preview = content[:100] + ("..." if len(content) > 100 else "")
            print(f"  [{i}] 用户：{preview}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                print(f"  [{i}] 助手（调用工具）：{', '.join(names)}")
            else:
                content = str(msg.get("content", ""))
                preview = content[:100] + ("..." if len(content) > 100 else "")
                print(f"  [{i}] 助手：{preview}")
        elif role == "tool":
            tool_content = str(msg.get("content", ""))
            preview = tool_content[:80] + ("..." if len(tool_content) > 80 else "")
            print(f"  [{i}] 工具结果：{preview}")
    print("-" * 40)
    print()


# ── 欢迎信息 ─────────────────────────────────────────────────
def print_welcome() -> None:
    print("=" * 60)
    print("  Second Brain —— 第七层：Agent + Function Calling")
    print("=" * 60)
    print(f"  模型：{config.OPENAI_API_MODEL}")
    print(f"  历史窗口：最近 {config.MAX_HISTORY_TURNS} 轮（含工具消息）")
    print()
    print("  工具：")
    print("    search_web  —— 联网搜索（Tavily）")
    print("    write_note  —— 写笔记到 layer7/test/notes/")
    print("    save_quiz   —— 保存题目到 layer7/test/exam/")
    print()
    print("  命令：")
    print("    /history  —— 查看当前会话历史")
    print("    /clear    —— 清空当前会话历史")
    print("    /exit     —— 退出程序")
    print("=" * 60)
    print()


# ── session 选择 ──────────────────────────────────────────────
def choose_session(memory: MemoryManager) -> tuple[str, list[dict]]:
    sessions = memory.list_sessions()

    if sessions:
        print("[已有 sessions]")
        for sid in sessions:
            print(f"  - {sid}")
    else:
        print("[当前没有历史 session，将创建新会话]")

    print()
    while True:
        session_id = input("请输入 session_id（输入已有可继续，输入新名字可新建）：\n").strip()
        if not session_id:
            print("session_id 不能为空，请重新输入。\n")
            continue

        history = memory.load(session_id)
        if session_id in set(sessions):
            print(f"\n[已载入 session: {session_id}，历史消息 {len(history)} 条]\n")
        else:
            print(f"\n[已创建新 session: {session_id}]\n")
        return session_id, history


# ── 主循环 ────────────────────────────────────────────────────
def main() -> None:
    print_welcome()

    memory    = MemoryManager(base_dir=MEMORY_DIR)
    session_id, history = choose_session(memory)

    while True:
        user_input = input("用户：\n").strip()
        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────────────────────
        if user_input == "/exit":
            # 退出前检查末尾是否有孤立的 user 消息（和 layer5 一致）
            if history and history[-1].get("role") == "user":
                history.pop()
                print("\n[检测到最后一轮未完成，已移除末尾 user 消息后再保存]\n")
            memory.save(session_id, history)
            print(f"\n[对话结束，session={session_id} 历史已保存]\n")
            break

        if user_input == "/clear":
            history = []
            memory.save(session_id, history)
            print("\n[当前 session 对话历史已清空并保存]\n")
            continue

        if user_input == "/history":
            print_history(history)
            continue

        # ── 正常对话流程 ──────────────────────────────────────
        try:
            # 1. 构建本轮 messages（system + history + user）
            messages = build_messages(history, user_input)

            # 2. 进入 Agent 循环，返回最终答案和更新后的 messages
            print("\n助手：")
            answer, updated_messages = run_agent(messages)
            print(answer)
            print()

            # 3. 从 updated_messages 里提取本轮新增的消息写入 history
            #    updated_messages = [system] + old_history + [新增消息...]
            #    新增消息 = updated_messages[len(messages):]
            new_messages = updated_messages[len(messages):]
            history.extend(new_messages)
            memory.save(session_id, history)

        except Exception as exc:
            print(f"\n[错误] {exc}\n")


if __name__ == "__main__":
    main()
