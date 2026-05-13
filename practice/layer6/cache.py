from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER2_DIR = CURRENT_DIR.parent / "layer2"
EMBEDDINGS_FILE = LAYER2_DIR / "embeddings.py"
if str(LAYER2_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER2_DIR))

_spec = importlib.util.spec_from_file_location("layer2_embeddings", EMBEDDINGS_FILE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load embeddings module from {EMBEDDINGS_FILE}")

_embeddings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_embeddings)

_ALLOWED_INTENTS = {"chitchat", "rag", "followup"}


class SemanticCache:
    """语义缓存器。

    设计上不依赖 Router：调用方在主流程中把 classify 得到的 intent 传进来，
    以降低 cache 与 router 的耦合度。
    """

    def __init__(
        self,
        rag_threshold: float = 0.90,
        chitchat_threshold: float = 0.85,
        max_items: int = 500,
    ) -> None:
        self.rag_threshold = float(rag_threshold)
        self.chitchat_threshold = float(chitchat_threshold)
        self.max_items = int(max_items)
        self._items: list[dict[str, Any]] = []

    def _normalize_intent(self, intent: str) -> str:
        normalized = intent.strip().lower()
        if normalized not in _ALLOWED_INTENTS:
            raise ValueError(f"unsupported intent: {intent}")
        return normalized

    def _threshold_for_intent(self, intent: str) -> float | None:
        if intent == "rag":
            return self.rag_threshold
        if intent == "chitchat":
            return self.chitchat_threshold
        # followup 对上下文依赖强，默认不做语义缓存
        return None

    def get(self, question: str, intent: str) -> str | None:
        normalized_intent = self._normalize_intent(intent)
        threshold = self._threshold_for_intent(normalized_intent)
        if threshold is None:
            return None

        if not isinstance(question, str) or not question.strip():
            return None

        query = question.strip()
        query_embedding = _embeddings.get_embedding(query)

        best_item: dict[str, Any] | None = None
        best_score = -1.0

        for item in self._items:
            if item["intent"] != normalized_intent:
                continue

            score = _embeddings.cosine_similarity(query_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None and best_score >= threshold:
            return str(best_item["answer"])
        return None

    def set(self, question: str, answer: str, intent: str) -> None:
        normalized_intent = self._normalize_intent(intent)
        if not isinstance(question, str) or not question.strip():
            return
        if not isinstance(answer, str) or not answer.strip():
            return

        normalized_question = question.strip()
        normalized_answer = answer.strip()
        question_embedding = _embeddings.get_embedding(normalized_question)

        # 先尝试精确命中，再尝试语义去重，避免同义问法重复占用缓存。
        best_index = -1
        best_score = -1.0
        for idx, item in enumerate(self._items):
            if item["intent"] != normalized_intent:
                continue

            if item["question"] == normalized_question:
                item["answer"] = normalized_answer
                item["embedding"] = question_embedding
                return

            score = _embeddings.cosine_similarity(question_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_index = idx

        threshold = self._threshold_for_intent(normalized_intent)
        if threshold is not None and best_index >= 0 and best_score >= threshold:
            target = self._items[best_index]
            target["question"] = normalized_question
            target["answer"] = normalized_answer
            target["embedding"] = question_embedding
            return

        self._items.append(
            {
                "question": normalized_question,
                "answer": normalized_answer,
                "intent": normalized_intent,
                "embedding": question_embedding,
            }
        )

        if self.max_items > 0 and len(self._items) > self.max_items:
            overflow = len(self._items) - self.max_items
            del self._items[:overflow]

    def clear(self) -> None:
        self._items.clear()


cache = SemanticCache(rag_threshold=0.90, chitchat_threshold=0.85)
