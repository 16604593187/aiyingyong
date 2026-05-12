from __future__ import annotations

import time

import rag


def print_welcome() -> None:
    """打印欢迎信息和使用说明"""
    print("=" * 60)
    print("  Second Brain —— 第四层：RAG 对话终端")
    print("=" * 60)
    print("  命令：")
    print("    /history  —— 查看当前对话历史")
    print("    /clear    —— 清空对话历史")
    print("    /exit     —— 退出程序")
    print("=" * 60)
    print()


def print_sources(sources: list[dict]) -> None:
    if not sources:
        print("[无检索来源]")
        return

    print("[检索来源]")
    for idx, src in enumerate(sources, 1):
        text_preview = src["text"][:120]
        if len(src["text"]) > 120:
            text_preview += "..."
        print(
            f"  [{idx}] source={src['source']} score={src['score']:.4f}\n"
            f"      {text_preview}"
        )


def print_history(history: list[dict]) -> None:
    if not history:
        print("\n[对话历史为空]\n")
        return

    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 40)
    for i, msg in enumerate(history, 1):
        role_label = "用户" if msg["role"] == "user" else "助手"
        content_preview = msg["content"][:100]
        if len(msg["content"]) > 100:
            content_preview += "..."
        print(f"  [{i}] {role_label}：{content_preview}")
    print("-" * 40)
    print()


def main() -> None:
    print_welcome()

    print("[系统] 正在进行 RAG 组件预热（Embedding + Chroma + OpenAI 客户端）...")
    started_at = time.time()
    rag.warmup()
    elapsed = time.time() - started_at
    print(f"[系统] 预热完成，耗时 {elapsed:.2f}s\n")

    history: list[dict] = []

    while True:
        user_input = input("用户：\n")
        if not user_input:
            continue

        if user_input == "/exit":
            print("\n[对话结束]\n")
            break
        elif user_input == "/clear":
            history = []
            print("\n[对话历史已清空]\n")
            continue
        elif user_input == "/history":
            print_history(history)
            continue

        history.append({"role": "user", "content": user_input})

        try:
            result = rag.query(user_input, top_k=5)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            print("\n助手:\n", end="")
            print(answer)
            print()
            print_sources(sources)
            print()

            history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            print(f"Error: {exc}\n")
            if history and history[-1]["role"] == "user":
                history.pop()


if __name__ == "__main__":
    main()
