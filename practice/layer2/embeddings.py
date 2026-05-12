from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def get_embedding(text: str) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return model.encode(text).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        raise ValueError("input vectors must be non-empty")
    if len(a) != len(b):
        raise ValueError("input vectors must have the same length")

    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for x, y in zip(a, b):
        x_val = float(x)
        y_val = float(y)
        dot += x_val * y_val
        norm_a_sq += x_val * x_val
        norm_b_sq += y_val * y_val

    if norm_a_sq == 0.0 or norm_b_sq == 0.0:
        raise ValueError("zero vector is not allowed")

    return dot / ((norm_a_sq ** 0.5) * (norm_b_sq ** 0.5))
