"""
Layer 10 - 统一 Agent 对话终端

完整流程：
  用户输入
    → followup 检测（是则跳过缓存）
    → 语义缓存检查（命中则直接返回）
    → Agent 主循环（Function Calling，模型自主决定工具调用）
    → 最终回答
    → 写入缓存 + 写入历史
    → 懒压缩检查（超阈值则压缩旧历史）

命令：
    /history    —— 查看当前会话历史
    /cache      —— 查看缓存命中/未命中状态
    /compress   —— 手动触发历史压缩
    /clear      —— 清空当前会话历史
    /exit       —— 退出程序
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR  = Path(__file__).resolve().parent
PRACTICE_DIR = CURRENT_DIR.parent
LAYER1_DIR   = PRACTICE_DIR / "layer1"
LAYER5_DIR   = PRACTICE_DIR / "layer5"

# 确保当前目录在 path 中（agent, cache, tools）
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config        = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")
mem_mod       = _load_module(LAYER5_DIR / "memory.py",      "layer5_memory")
MemoryManager    = mem_mod.MemoryManager
HistoryCompressor = mem_mod.HistoryCompressor

from agent import run_agent  # noqa: E402
from cache import SemanticCache, is_followup, infer_intent_from_tools  # noqa: E402
from tools import warmup  # noqa: E402

HISTORY_DIR = CURRENT_DIR / "history"


# ── OpenAI 客户端（给 HistoryCompressor 用）────────────────────
_openai_client: Any | None = None


def _get_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from openai import OpenAI
    _openai_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_API_BASE)
    return _openai_client


# ── 历史截断（只在 role=user 处截断，保证 tool_calls/tool 组完整）
def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """
    Agent 安全截断：只在 role=user 处切片。

    如果历史最前面有摘要消息（role=system，约定前缀），截断时保留它。
    """
    if max_turns <= 0:
        return []

    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return history

    start = user_indices[len(user_indices) - max_turns]

    # 保留摘要消息
    prefix: list[dict] = []
    if (history and history[0].get("role") == "system"
            and history[0].get("content", "").startswith("[以下是早期对话的摘要")):
        prefix = [history[0]]
        if start == 0:
            start = 1

    return prefix + history[start:]


# ── 打印历史 ──────────────────────────────────────────────────
def print_history(history: list[dict]) -> None:
    if not history:
        print("\n[对话历史为空]\n")
        return

    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 50)
    for i, msg in enumerate(history, 1):
        role    = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip()

        # 处理不同消息类型的展示
        if role == "assistant" and msg.get("tool_calls"):
            tool_names = [tc.get("function", {}).get("name", "?")
                          for tc in msg.get("tool_calls", [])]
            label = f"助手（调用工具：{', '.join(tool_names)}）"
            preview = content[:60] + ("..." if len(content) > 60 else "") if content else ""
        elif role == "tool":
            label = "工具结果"
            preview = content[:80] + ("..." if len(content) > 80 else "")
        elif role == "system":
            label = "系统/摘要"
            preview = content[:80] + ("..." if len(content) > 80 else "")
        elif role == "user":
            label = "用户"
            preview = content[:100] + ("..." if len(content) > 100 else "")
        elif role == "assistant":
            label = "助手"
            preview = content[:100] + ("..." if len(content) > 100 else "")
        else:
            label = role
            preview = content[:80] + ("..." if len(content) > 80 else "")

        print(f"  [{i}] {label}：{preview}")
    print("-" * 50)
    print()


# ── 欢迎信息 ──────────────────────────────────────────────────
def print_welcome() -> None:
    print("=" * 60)
    print("  Second Brain —— 第十层：Agent + RAG 统一架构")
    print("=" * 60)
    print(f"  模型：{config.OPENAI_API_MODEL}")
    print(f"  历史窗口：最近 {config.MAX_HISTORY_TURNS} 轮")
    print()
    print("  架构：语义缓存 → Agent（Function Calling）")
    print("  工具：search_knowledge_base / search_web / write_note")
    print("  特性：懒压缩 + 安全截断 + followup 跳过缓存")
    print()
    print("  命令：")
    print("    /history   —— 查看当前会话历史")
    print("    /compress  —— 手动触发历史压缩")
    print("    /clear     —— 清空当前会话历史")
    print("    /exit      —— 退出程序")
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

    print("[初始化中，首次加载模型需要一点时间...]")
    warmup()
    print("[预热完成]\n")

    # 组件初始化
    memory     = MemoryManager(base_dir=HISTORY_DIR)
    cache      = SemanticCache(max_items=500)
    compressor = HistoryCompressor(
        client=_get_client(),
        model=config.OPENAI_API_MODEL,
        compress_threshold=20,
        compress_turns=10,
    )

    session_id, history = choose_session(memory)

    while True:
        user_input = input("用户：\n").strip()
        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────────────────────
        if user_input == "/exit":
            if history and history[-1].get("role") == "user":
                history.pop()
                print("\n[移除末尾未完成的 user 消息]")
            memory.save(session_id, history)
            print(f"\n[对话结束，session={session_id} 历史已保存]\n")
            break

        if user_input == "/clear":
            history = []
            cache.clear()
            memory.save(session_id, history)
            print("\n[会话历史和缓存已清空]\n")
            continue

        if user_input == "/history":
            print_history(history)
            continue

        if user_input == "/compress":
            before = len(history)
            print("\n[手动触发历史压缩...]")
            history = compressor.compress(history)
            after = len(history)
            memory.save(session_id, history)
            print(f"[压缩完成：{before} 条 → {after} 条，已保存]\n")
            continue

        # ── 正常对话流程 ──────────────────────────────────────
        try:
            # 1. 语义缓存检查（followup 跳过）
            cache_hit = False
            if is_followup(user_input):
                print("[调试] followup 检测命中，跳过缓存")
            else:
                cached_answer = cache.get(user_input)
                if cached_answer is not None:
                    cache_hit = True
                    answer = cached_answer
                    print(f"\n[调试] cache=hit")
                    print(f"助手（缓存命中）：\n{answer}\n")
                    # 缓存命中也写入历史
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": answer})
                    memory.save(session_id, history)
                    continue
                else:
                    print("[调试] cache=miss")

            # 2. 未命中 → Agent 主循环
            # 组装 messages：system prompt + 截断后历史 + 当前 user
            trimmed = trim_history(history, config.MAX_HISTORY_TURNS)
            messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
            messages.extend(trimmed)
            messages.append({"role": "user", "content": user_input})

            answer, updated_msgs = run_agent(messages)

            # 调试：从 Agent 消息中提取工具调用信息
            tools_called = []
            for msg in updated_msgs:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tools_called.append(tc.get("function", {}).get("name", "?"))

            if tools_called:
                print(f"\n[调试] 工具调用：{' → '.join(tools_called)}")
            else:
                print(f"\n[调试] 未调用工具（直接回答）")

            print(f"助手：\n{answer}\n")

            # 3. 写入缓存（followup 不写，避免上下文依赖的答案被复用）
            if not is_followup(user_input):
                intent = infer_intent_from_tools(updated_msgs)
                print(f"[调试] 推断类型={intent}，写入缓存（阈值={cache._threshold_for(intent):.2f}）")
                cache.set(user_input, answer, intent=intent)
            else:
                print("[调试] followup 请求，跳过缓存写入")

            # 4. 写入历史
            # 从 updated_msgs 中提取本轮新增的消息（跳过 system prompt 和历史部分）
            # 新增部分 = updated_msgs[len(messages)-1:]，即从当前 user 开始的所有消息
            new_msgs_start = len(messages) - 1  # 当前 user 消息的位置
            new_msgs = updated_msgs[new_msgs_start:]
            history.extend(new_msgs)
            memory.save(session_id, history)

            # 5. 懒压缩检查
            if compressor.should_compress(history):
                before = len(history)
                print("[历史超过阈值，自动触发压缩...]")
                history = compressor.compress(history)
                after = len(history)
                memory.save(session_id, history)
                print(f"[压缩完成：{before} 条 → {after} 条]\n")

        except Exception as exc:
            print(f"\n[错误] {exc}\n")


if __name__ == "__main__":
    main()
