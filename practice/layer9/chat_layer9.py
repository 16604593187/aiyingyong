"""
Layer 9 - 内容压缩对话终端

在 Layer 8 基础上的变化：
  - 检索链路：混合检索 + Reranker + Contextual Compression
  - 新增 /context 命令：查看上一轮压缩前后的原始片段 vs 压缩结果，
    以及压缩统计数据（原始字符数、压缩后字符数、压缩率、过滤片段数）
  - 其余逻辑（session 管理、历史、持久化）与 layer8 保持一致
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR  = CURRENT_DIR.parent / "layer1"
LAYER5_DIR  = CURRENT_DIR.parent / "layer5"

for d in (str(CURRENT_DIR),):
    if d not in sys.path:
        sys.path.insert(0, d)


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config        = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")
mem_mod       = _load_module(LAYER5_DIR / "memory.py",      "layer5_memory")
MemoryManager = mem_mod.MemoryManager

from compressed_rag import warmup, ask  # noqa: E402

MEMORY_DIR = CURRENT_DIR / "history"


# ── 历史截断 ───────────────────────────────────────────────────
def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    if max_turns <= 0:
        return []
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return history
    start = user_indices[len(user_indices) - max_turns]
    return history[start:]


# ── /context 展示：压缩前后对比 ───────────────────────────────
def print_context(sources: list[dict], stats: dict) -> None:
    if not sources:
        print("\n[本轮未检索到相关知识库片段]\n")
        return

    print(f"\n[压缩统计]")
    print(f"  原始总字符：{stats.get('total_original_chars', 0)}")
    print(f"  压缩后字符：{stats.get('total_compressed_chars', 0)}")
    print(f"  压缩率：    {stats.get('compression_ratio', 1.0):.1%}")
    print(f"  过滤片段数：{stats.get('filtered_count', 0)}")
    print()

    print(f"[逐片段对比，共 {len(sources)} 条]")
    print("=" * 60)
    for i, s in enumerate(sources, 1):
        src          = s.get("source", "unknown")
        rerank_score = s.get("rerank_score", 0.0)
        original     = (s.get("text") or "").strip()
        compressed   = (s.get("compressed_text") or "").strip()

        print(f"[{i}] {src}  rerank={rerank_score:.4f}")
        print(f"  原始（{len(original)} 字）：")
        # 只展示前 200 字，避免刷屏
        preview = original[:200] + ("..." if len(original) > 200 else "")
        for line in preview.splitlines():
            print(f"    {line}")
        print()
        if compressed:
            print(f"  压缩后（{len(compressed)} 字）：")
            for line in compressed.splitlines():
                print(f"    {line}")
        else:
            print(f"  压缩后：[已过滤，与问题无关]")
        print("-" * 60)
    print()


# ── 打印历史 ───────────────────────────────────────────────────
def print_history(history: list[dict]) -> None:
    if not history:
        print("\n[对话历史为空]\n")
        return
    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 40)
    for i, msg in enumerate(history, 1):
        role    = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        preview = content[:100] + ("..." if len(content) > 100 else "")
        label   = {"user": "用户", "assistant": "助手", "system": "系统"}.get(role, role)
        print(f"  [{i}] {label}：{preview}")
    print("-" * 40)
    print()


# ── 欢迎信息 ───────────────────────────────────────────────────
def print_welcome() -> None:
    print("=" * 60)
    print("  Second Brain —— 第九层：混合检索 + Reranker + 内容压缩")
    print("=" * 60)
    print(f"  模型：{config.OPENAI_API_MODEL}")
    print(f"  历史窗口：最近 {config.MAX_HISTORY_TURNS} 轮")
    print()
    print("  检索链路：向量+BM25 → RRF → Reranker → LLM 抽取压缩")
    print()
    print("  命令：")
    print("    /context  —— 查看上一轮压缩前后片段对比及压缩统计")
    print("    /history  —— 查看当前会话历史")
    print("    /clear    —— 清空当前会话历史")
    print("    /exit     —— 退出程序")
    print("=" * 60)
    print()


# ── session 选择 ───────────────────────────────────────────────
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


# ── 主循环 ─────────────────────────────────────────────────────
def main() -> None:
    print_welcome()

    print("[初始化中，首次加载模型需要一点时间...]")
    warmup()

    memory     = MemoryManager(base_dir=MEMORY_DIR)
    session_id, history = choose_session(memory)

    last_sources: list[dict] = []
    last_stats:   dict       = {}

    while True:
        user_input = input("用户：\n").strip()
        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────────────────────
        if user_input == "/exit":
            if history and history[-1].get("role") == "user":
                history.pop()
                print("\n[检测到最后一轮未完成，已移除末尾 user 消息后再保存]\n")
            memory.save(session_id, history)
            print(f"\n[对话结束，session={session_id} 历史已保存]\n")
            break

        if user_input == "/clear":
            history       = []
            last_sources  = []
            last_stats    = {}
            memory.save(session_id, history)
            print("\n[当前 session 对话历史已清空并保存]\n")
            continue

        if user_input == "/history":
            print_history(history)
            continue

        if user_input == "/context":
            print_context(last_sources, last_stats)
            continue

        # ── 正常对话流程 ──────────────────────────────────────
        try:
            trimmed = trim_history(history, config.MAX_HISTORY_TURNS)
            result  = ask(user_input, history=trimmed)

            answer       = result["answer"]
            last_sources = result["sources"]
            last_stats   = result.get("compression_stats", {})

            print(f"\n助手：\n{answer}\n")

            # 显示压缩摘要
            stats = last_stats
            if stats:
                ratio = stats.get("compression_ratio", 1.0)
                filtered = stats.get("filtered_count", 0)
                orig  = stats.get("total_original_chars", 0)
                comp  = stats.get("total_compressed_chars", 0)
                print(
                    f"[检索压缩：{orig}字 → {comp}字，"
                    f"压缩率 {ratio:.0%}，过滤 {filtered} 片段  /context 查看详情]\n"
                )

            # 写入历史
            history.append({"role": "user",     "content": user_input})
            history.append({"role": "assistant", "content": answer})
            memory.save(session_id, history)

        except Exception as exc:
            print(f"\n[错误] {exc}\n")


if __name__ == "__main__":
    main()
