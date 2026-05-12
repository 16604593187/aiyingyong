# =============================================================================
# chat.py —— 主入口，多轮对话终端
# 职责：维护对话历史、调用模型 API、处理流式输出、管理 context window
# =============================================================================

from openai import OpenAI
import config

# ── 初始化客户端 ──────────────────────────────────────────────────────────────
# OpenAI SDK 兼容所有遵循 OpenAI 接口规范的模型服务（DeepSeek、Qwen 等）
client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.API_BASE,
)


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def trim_history(history: list[dict]) -> list[dict]:
    """
    裁剪对话历史，确保不超过 MAX_HISTORY_TURNS 轮。

    为什么需要裁剪？
    - 每次 API 调用都会把完整历史发送给模型
    - 历史越长，消耗的 token 越多，费用越高，速度越慢
    - 超过 context window 上限会直接报错

    裁剪策略：保留最新的 N 轮，丢弃最早的对话
    注意：每轮包含 1 条 user + 1 条 assistant，所以最大消息数 = N * 2
    """
    max_messages = config.MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        # 从列表末尾取最新的 max_messages 条
        trimmed = history[-max_messages:]
        print(f"\n[系统提示] 对话历史已超过 {config.MAX_HISTORY_TURNS} 轮，"
              f"已自动丢弃最早的 {len(history) - max_messages} 条消息。\n")
        return trimmed
    return history


def chat(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    发送一条消息并获取回复。

    参数：
        user_input: 用户输入的文本
        history:    当前对话历史（不含 system prompt）

    返回：
        (assistant_reply, updated_history)
        - assistant_reply: 模型的完整回复文本
        - updated_history: 追加了本轮对话后的新历史

    对话历史的数据结构：
        [
            {"role": "user",      "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user",      "content": "今天天气怎么样？"},
            ...
        ]
    """
    # 1. 把用户消息追加到历史
    history.append({"role": "user", "content": user_input})

    # 2. 裁剪历史，防止超出 context window
    history = trim_history(history)

    # 3. 拼装完整消息列表：system prompt 始终在最前面
    #    system prompt 不计入 history，每次调用时动态拼入
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ] + history

    # 4. 调用 API，开启流式输出（stream=True）
    #    流式输出的好处：模型边生成边返回，用户不需要等待完整回复
    stream = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        stream=True,
    )

    # 5. 逐块打印流式输出，同时拼接完整回复
    print("\n助手：", end="", flush=True)
    full_reply = ""
    for chunk in stream:
        # 每个 chunk 包含一小段文本，delta.content 可能为 None（最后一块）
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            print(delta_content, end="", flush=True)
            full_reply += delta_content
    print()  # 换行

    # 6. 把模型回复追加到历史
    history.append({"role": "assistant", "content": full_reply})

    return full_reply, history


def print_welcome():
    """打印欢迎信息和使用说明"""
    print("=" * 60)
    print("  Second Brain —— 第一层：对话终端")
    print("=" * 60)
    print(f"  模型：{config.MODEL_NAME}")
    print(f"  最大历史轮数：{config.MAX_HISTORY_TURNS}")
    print()
    print("  命令：")
    print("    /history  —— 查看当前对话历史")
    print("    /clear    —— 清空对话历史")
    print("    /exit     —— 退出程序")
    print("=" * 60)
    print()


def print_history(history: list[dict]):
    """打印当前对话历史，方便调试"""
    if not history:
        print("\n[对话历史为空]\n")
        return
    print(f"\n[对话历史，共 {len(history)} 条消息]")
    print("-" * 40)
    for i, msg in enumerate(history, 1):
        role_label = "用户" if msg["role"] == "user" else "助手"
        # 只显示前 100 个字符，避免刷屏
        content_preview = msg["content"][:100]
        if len(msg["content"]) > 100:
            content_preview += "..."
        print(f"  [{i}] {role_label}：{content_preview}")
    print("-" * 40)
    print()


# ── 主循环 ────────────────────────────────────────────────────────────────────

def main():
    print_welcome()

    # 对话历史：只存 user 和 assistant 消息，system prompt 单独管理
    history: list[dict] = []

    while True:
        try:
            user_input = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C 或 Ctrl+D 优雅退出
            print("\n\n再见！")
            break

        # 空输入跳过
        if not user_input:
            continue

        # 处理内置命令
        if user_input == "/exit":
            print("再见！")
            break
        elif user_input == "/clear":
            history = []
            print("\n[对话历史已清空]\n")
            continue
        elif user_input == "/history":
            print_history(history)
            continue

        # 正常对话
        try:
            _, history = chat(user_input, history)
        except Exception as e:
            print(f"\n[错误] API 调用失败：{e}\n")
            # 出错时把刚才追加的 user 消息从历史中移除，保持历史一致性
            if history and history[-1]["role"] == "user":
                history.pop()


if __name__ == "__main__":
    main()
