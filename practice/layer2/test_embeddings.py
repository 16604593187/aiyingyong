import embeddings

DEFAULT_CASES = [
    ("苹果手机", "iPhone"),
    ("香蕉", "水果"),
]


def compare_texts(text1: str, text2: str) -> float:
    embedding_1 = embeddings.get_embedding(text1)
    embedding_2 = embeddings.get_embedding(text2)
    return embeddings.cosine_similarity(embedding_1, embedding_2)


def run_default_cases() -> None:
    print("预置测试结果：")
    for text1, text2 in DEFAULT_CASES:
        try:
            score = compare_texts(text1, text2)
            print(f"- {text1} vs {text2}: {score:.6f}")
        except Exception as exc:
            print(f"- {text1} vs {text2}: 失败 ({exc})")


def interactive_mode() -> None:
    print("\n输入两段文本计算余弦相似度，输入 /exit 退出。")
    while True:
        text1 = input("文本1: ").strip()
        if text1 == "/exit":
            print("已退出。")
            return

        text2 = input("文本2: ").strip()
        if text2 == "/exit":
            print("已退出。")
            return

        if not text1 or not text2:
            print("请输入非空文本。")
            continue

        try:
            score = compare_texts(text1, text2)
            print(f"余弦相似度: {score:.6f}\n")
        except Exception as exc:
            print(f"计算失败: {exc}\n")


def main() -> None:
    run_default_cases()
    interactive_mode()


if __name__ == "__main__":
    main()
