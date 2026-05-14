from __future__ import annotations

import sys
import time
from pathlib import Path

import importlib.util

from openai import OpenAI

from memory import MemoryManager, HistoryCompressor

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR = CURRENT_DIR.parent / "layer1"
LAYER4_DIR = CURRENT_DIR.parent / "layer4"
if str(LAYER1_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER1_DIR))
if str(LAYER4_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER4_DIR))

CONFIG_FILE = LAYER1_DIR / "config_mine.py"
_config_spec = importlib.util.spec_from_file_location("layer1_config_mine", CONFIG_FILE)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"cannot load config module from {CONFIG_FILE}")

config_mine = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config_mine)

RAG_FILE = LAYER4_DIR / "rag.py"
_rag_spec = importlib.util.spec_from_file_location("layer4_rag", RAG_FILE)
if _rag_spec is None or _rag_spec.loader is None:
    raise ImportError(f"cannot load rag module from {RAG_FILE}")

rag = importlib.util.module_from_spec(_rag_spec)
_rag_spec.loader.exec_module(rag)

client = OpenAI(
    api_key=config_mine.OPENAI_API_KEY,
    base_url=config_mine.OPENAI_API_BASE,
)

# ── 历史压缩器 ─────────────────────────────────────────────────
# compress_threshold=20 条消息（=10 轮）时触发自动压缩
# compress_turns=10     每次压缩最旧的 10 轮（20 条）
compressor = HistoryCompressor(
    client=client,
    model=config_mine.OPENAI_API_MODEL,
    compress_threshold=20,
    compress_turns=10,
)


def trim_messages(messages: list[dict], max_turns: int) -> list[dict]:
    """
    发送给 API 前的短期截断：只保留最近 max_turns 轮。
    注意：这里的 messages 可能已经包含摘要消息（role=system），
    截断时保留最前面的摘要消息 + 最近若干轮原文。
    """
    if max_turns <= 0:
        return []

    # 找出所有 role=user 的下标（安全截断点）
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]

    if len(user_indices) <= max_turns:
        return messages

    # 从倒数第 max_turns 个 user 消息处截断
    start = user_indices[len(user_indices) - max_turns]

    # 如果最前面有摘要消息，保留它
    prefix: list[dict] = []
    if messages and messages[0].get("role") == "system" and messages[0].get("content", "").startswith("[以下是早期对话的摘要"):
        prefix = [messages[0]]
        # 确保 start 不把摘要消息也截掉
        if start == 0:
            start = 1

    return prefix + messages[start:]


def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    question = user_input.strip()

    request_history = trim_messages(history, config_mine.MAX_HISTORY_TURNS)
    request_messages = [{"role": "system", "content": config_mine.SYSTEM_PROMPT}] + request_history

    # RAG 检索：复用 layer4/rag.py 提供的公开接口
    retrieval_context = rag.retrieve(question, top_k=5)
    if retrieval_context:
        request_messages.append(
            {
                "role": "system",
                "content": "以下是知识库检索到的相关片段，请优先基于这些内容回答：\n\n" + retrieval_context,
            }
        )

    # 当前问题必须放在最后
    request_messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=config_mine.OPENAI_API_MODEL,
        messages=request_messages,
        temperature=config_mine.TEMPERATURE,
        max_tokens=config_mine.MAX_TOKENS,
        stream=True,
    )

    print("\n助手:\n", end="", flush=True)
    full_reply = ""
    for chunk in response:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            full_reply += delta_content
            print(delta_content, end="", flush=True)
    print()

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_reply})
    return full_reply, history


def print_welcome() -> None:
    print("=" * 60)
    print("  Second Brain —— 第五层：带持久记忆 + 历史压缩的对话终端")
    print("=" * 60)
    print(f"  模型：{config_mine.OPENAI_API_MODEL}")
    print(f"  上下文最大轮数（发送给 API）：{config_mine.MAX_HISTORY_TURNS}")
    print(f"  懒压缩阈值：{compressor.compress_threshold} 条消息")
    print(f"  每次压缩轮数：{compressor.compress_turns} 轮")
    print()
    print("  命令：")
    print("    /history      —— 查看当前会话历史（完整存盘）")
    print("    /api_history  —— 查看发送给 API 的历史窗口")
    print("    /compress     —— 手动触发一次历史压缩")
    print("    /clear        —— 清空当前会话历史")
    print("    /exit         —— 退出程序")
    print("=" * 60)
    print()


def print_history(history: list[dict]) -> None:
    if not history:
        print("\n[对话历史为空]\n")
        return

    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 40)
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        content_preview = content[:100]
        if len(content) > 100:
            content_preview += "..."
        label = {"user": "用户", "assistant": "助手", "system": "摘要"}.get(role, role)
        print(f"  [{i}] {label}：{content_preview}")
    print("-" * 40)
    print()


def print_api_history(history: list[dict]) -> None:
    window = trim_messages(history, config_mine.MAX_HISTORY_TURNS)

    print("\n[发送给 API 的历史窗口]")
    print(f"  system：{config_mine.SYSTEM_PROMPT[:80]}{'...' if len(config_mine.SYSTEM_PROMPT) > 80 else ''}")

    if not window:
        print("  [无历史消息，将仅发送 system prompt + 当前问题]\n")
        return

    print(f"  [历史消息 {len(window)} 条]")
    for i, msg in enumerate(window, 1):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        preview = content[:100] + ("..." if len(content) > 100 else "")
        print(f"  [{i}] {role}: {preview}")
    print()


def choose_session(memory: MemoryManager) -> tuple[str, list[dict]]:
    sessions = memory.list_sessions()
    session_set = set(sessions)

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
        if session_id in session_set:
            print(f"\n[已载入 session: {session_id}，历史消息 {len(history)} 条]\n")
        else:
            print(f"\n[已创建新 session: {session_id}]\n")
        return session_id, history


def main() -> None:
    print_welcome()

    print("[系统] 正在进行 RAG 组件预热（Embedding + Chroma + OpenAI 客户端）...")
    started_at = time.time()
    rag.warmup()
    elapsed = time.time() - started_at
    print(f"[系统] 预热完成，耗时 {elapsed:.2f}s\n")

    memory = MemoryManager()
    session_id, history = choose_session(memory)

    while True:
        user_input = input("用户：\n")
        if not user_input:
            continue

        if user_input == "/exit":
            dropped_incomplete_user = False
            if history and history[-1].get("role") == "user":
                history.pop()
                dropped_incomplete_user = True

            memory.save(session_id, history)
            if dropped_incomplete_user:
                print("\n[检测到最后一轮未完成，已移除末尾 user 消息后再保存]\n")
            print(f"\n[对话结束，session={session_id} 历史已保存]\n")
            break

        elif user_input == "/clear":
            history = []
            memory.save(session_id, history)
            print("\n[当前 session 对话历史已清空并保存]\n")
            continue

        elif user_input == "/history":
            print_history(history)
            continue

        elif user_input == "/api_history":
            print_api_history(history)
            continue

        elif user_input == "/compress":
            before = len(history)
            print("\n[手动触发历史压缩...]")
            history = compressor.compress(history)
            after = len(history)
            memory.save(session_id, history)
            print(f"[压缩完成：{before} 条 → {after} 条，已保存]\n")
            continue

        try:
            _, history = chat(user_input, history)

            # ── 懒压缩：每轮对话后检查是否超过阈值 ──────────────
            if compressor.should_compress(history):
                before = len(history)
                print("\n[历史超过阈值，自动触发压缩...]")
                history = compressor.compress(history)
                after = len(history)
                print(f"[压缩完成：{before} 条 → {after} 条]\n")

            memory.save(session_id, history)
        except Exception as exc:
            print(f"Error: {exc}\n")
            if history and history[-1].get("role") == "user":
                history.pop()


if __name__ == "__main__":
    main()
