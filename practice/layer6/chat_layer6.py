from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import cache as semantic_cache_module
import router

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR = CURRENT_DIR.parent / "layer1"
LAYER4_DIR = CURRENT_DIR.parent / "layer4"
LAYER5_DIR = CURRENT_DIR.parent / "layer5"

if str(LAYER1_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER1_DIR))
if str(LAYER4_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER4_DIR))
if str(LAYER5_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER5_DIR))


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_mine = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config_mine")
rag = _load_module(LAYER4_DIR / "rag.py", "layer4_rag")
memory_module = _load_module(LAYER5_DIR / "memory.py", "layer5_memory")
MemoryManager = memory_module.MemoryManager

client = OpenAI(
    api_key=config_mine.OPENAI_API_KEY,
    base_url=config_mine.OPENAI_API_BASE,
)
cache = semantic_cache_module.cache


def trim_messages(messages: list[dict], max_turns: int) -> list[dict]:
    """只保留最近 max_turns 轮（一轮 = user + assistant 各一条）。"""
    if max_turns <= 0:
        return []

    max_messages = max_turns * 2
    if len(messages) <= max_messages:
        return messages

    return messages[-max_messages:]


def build_request_messages(history: list[dict], retrieval_context: str, question: str) -> list[dict]:
    request_history = trim_messages(history, config_mine.MAX_HISTORY_TURNS)
    request_messages: list[dict] = [{"role": "system", "content": config_mine.SYSTEM_PROMPT}]

    # 1) system -> 2) history
    request_messages.extend(request_history)

    # 3) retrieval snippets
    if retrieval_context:
        request_messages.append(
            {
                "role": "system",
                "content": "以下是知识库检索到的相关片段，请优先基于这些内容回答：\n\n"
                + retrieval_context,
            }
        )

    # 4) current question
    request_messages.append({"role": "user", "content": question})
    return request_messages


def call_chat_api(request_messages: list[dict]) -> str:
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
    return full_reply


def print_welcome() -> None:
    print("=" * 60)
    print("  Second Brain —— 第六层：Router + Cache + RAG + Memory")
    print("=" * 60)
    print(f"  模型：{config_mine.OPENAI_API_MODEL}")
    print(f"  API 历史窗口最大轮数：{config_mine.MAX_HISTORY_TURNS}")
    print(f"  Cache 阈值：rag={cache.rag_threshold:.2f}, chitchat={cache.chitchat_threshold:.2f}")
    print()
    print("  命令：")
    print("    /history      —— 查看当前会话历史（完整存盘）")
    print("    /api_history  —— 查看发送给 API 的历史窗口")
    print("    /debug        —— 开关调试信息（intent / cache / rag）")
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
        role_label = "用户" if msg.get("role") == "user" else "助手"
        content = str(msg.get("content", ""))
        preview = content[:100] + ("..." if len(content) > 100 else "")
        print(f"  [{i}] {role_label}：{preview}")
    print("-" * 40)
    print()


def print_api_history(history: list[dict]) -> None:
    window = trim_messages(history, config_mine.MAX_HISTORY_TURNS)

    print("\n[发送给 API 的历史窗口]")
    system_preview = config_mine.SYSTEM_PROMPT[:80]
    if len(config_mine.SYSTEM_PROMPT) > 80:
        system_preview += "..."
    print(f"  system：{system_preview}")

    if not window:
        print("  [无历史消息，将仅发送 system + 检索片段 + 当前问题]\n")
        return

    print(f"  [历史消息 {len(window)} 条，来自最近 {config_mine.MAX_HISTORY_TURNS} 轮]")
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

    print("[系统] 正在进行组件预热（Router / Cache Embedding / RAG / OpenAI）...")
    started_at = time.time()
    rag.warmup()
    router.classify("你好")
    elapsed = time.time() - started_at
    print(f"[系统] 预热完成，耗时 {elapsed:.2f}s\n")

    memory = MemoryManager()
    session_id, history = choose_session(memory)
    debug_mode = True
    print("[系统] 调试信息已开启（可输入 /debug 关闭）\n")

    while True:
        user_input = input("用户：\n")
        if not user_input:
            continue

        if user_input == "/exit":
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

        if user_input == "/api_history":
            print_api_history(history)
            continue

        if user_input == "/debug":
            debug_mode = not debug_mode
            status = "开启" if debug_mode else "关闭"
            print(f"\n[系统] 调试信息已{status}\n")
            continue

        question = user_input.strip()

        try:
            # 1) Router 分类
            intent = router.classify(question)
            if debug_mode:
                print(f"\n[debug] intent={intent}")

            # 2) Cache 查询（按 intent 阈值）
            cached_answer = cache.get(question, intent)
            if cached_answer is not None:
                if debug_mode:
                    print("[debug] cache=hit, rag=skip")
                print("\n[cache hit]", intent)
                print("\n助手:\n", end="")
                print(cached_answer)
                print()

                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": cached_answer})
                memory.save(session_id, history)
                continue

            # 3) 未命中：RAG 检索（仅 rag 意图）+ Memory 加载（独立）
            retrieval_context = ""
            if intent == "rag":
                if debug_mode:
                    print("[debug] cache=miss, rag=retrieve")
                retrieval_context = rag.retrieve(question, top_k=5)
                if debug_mode:
                    retrieved = "yes" if retrieval_context else "no"
                    print(f"[debug] rag_retrieved={retrieved}, context_chars={len(retrieval_context)}")
            else:
                if debug_mode:
                    print(f"[debug] cache=miss, rag=skip (intent={intent})")
            history = memory.load(session_id)

            # 4) 汇聚构建 prompt
            request_messages = build_request_messages(history, retrieval_context, question)

            # 5) 调用 API
            answer = call_chat_api(request_messages)

            # 6) 保存历史 + 写入缓存
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            memory.save(session_id, history)
            cache.set(question, answer, intent)

        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
