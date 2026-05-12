from openai import OpenAI
import config_mine
client=OpenAI(api_key=config_mine.OPENAI_API_KEY,base_url=config_mine.OPENAI_API_BASE)
"""
对话历史截断
"""
def trim_history(history:list[dict])->list[dict]:
    max_messages=config_mine.MAX_HISTORY_TURNS*2
    if len(history)>max_messages:
        print(f"history is too long,trimming最早的{len(history)-max_messages}个消息\n")
        return history[-max_messages:]
    return history

def chat(user_input:str,history:list[dict])->tuple[str,list[dict]]:
    history.append({"role":"user","content":user_input})
    
    messages=[{"role":"system","content":config_mine.SYSTEM_PROMPT}]+history
    response=client.chat.completions.create(
        model=config_mine.OPENAI_API_MODEL,
        messages=messages,
        temperature=config_mine.TEMPERATURE,
        max_tokens=config_mine.MAX_TOKENS,
        stream=True
    )
    print("\n助手:\n",end="",flush=True)
    full_reply=""
    for chunk in response:
        delta_content=chunk.choices[0].delta.content
        if delta_content:
            full_reply+=delta_content
            print(delta_content,end="",flush=True)
    print()
    history.append({"role":"assistant","content":full_reply})
    history=trim_history(history)
    return full_reply,history

def print_welcome():
    """打印欢迎信息和使用说明"""
    print("=" * 60)
    print("  Second Brain —— 第一层：对话终端")
    print("=" * 60)
    print(f"  模型：{config_mine.OPENAI_API_MODEL}")
    print(f"  最大历史轮数：{config_mine.MAX_HISTORY_TURNS}")
    print()
    print("  命令：")
    print("    /history  —— 查看当前对话历史")
    print("    /clear    —— 清空对话历史")
    print("    /exit     —— 退出程序")
    print("=" * 60)
    print()

def print_history(history: list[dict]):
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

def main():
    print_welcome()
    history = []
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
        
        try:
            _,history=chat(user_input,history)
        except Exception as e:
            print(f"Error: {e}\n")
            if history and history[-1]["role"] == "user":
                history.pop()

if __name__ == "__main__":
    main()