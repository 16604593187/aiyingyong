"""
Layer 10 - 语义缓存（分级阈值版）

与 Layer 6 的区别：
  - 去掉了对 Router 的依赖
  - 类型标记不再由 Router 提供，而是从 Agent 的工具调用行为中反推：
      Agent 调用了 search_knowledge_base → "rag"
      Agent 调用了 search_web            → "web"
      Agent 没调用任何工具                → "chitchat"
  - 缓存写入时记录推断类型，查询时根据条目自身类型用对应阈值匹配
  - 新增 followup 启发式检测（包含指代词时跳过缓存）
  - 作为 Agent 的前置拦截层，命中则完全跳过 Agent 循环
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR  = Path(__file__).resolve().parent
PRACTICE_DIR = CURRENT_DIR.parent
LAYER2_DIR   = PRACTICE_DIR / "layer2"

if str(LAYER2_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER2_DIR))


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_embeddings: Any | None = None


def _get_embeddings() -> Any:
    global _embeddings
    if _embeddings is None:
        _embeddings = _load_module(LAYER2_DIR / "embeddings.py", "layer2_embeddings")
    return _embeddings


# ── followup 检测 ─────────────────────────────────────────────
_FOLLOWUP_PATTERNS = re.compile(
    r"(上文|上面|你刚才|你说的|继续|接着|然后呢|补充|详细一点|展开|"
    r"完整一点|再说说|解释一下刚才|前面提到|之前说的)",
    re.IGNORECASE,
)


def is_followup(question: str) -> bool:
    """启发式判断是否为 followup 请求。"""
    return bool(_FOLLOWUP_PATTERNS.search(question))


# ── 从 Agent 工具调用行为推断类型 ─────────────────────────────
def infer_intent_from_tools(updated_messages: list[dict]) -> str:
    """
    从 Agent 一轮对话产生的 messages 中推断意图类型。

    规则：
      - 如果出现过 search_knowledge_base 调用 → "rag"
      - 如果出现过 search_web 调用             → "web"
      - 都没调用                               → "chitchat"

    优先级：rag > web > chitchat（如果同时调了知识库和网络，按 rag 算）
    """
    has_rag = False
    has_web = False

    for msg in updated_messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            if name == "search_knowledge_base":
                has_rag = True
            elif name == "search_web":
                has_web = True

    if has_rag:
        return "rag"
    if has_web:
        return "web"
    return "chitchat"


# ── 语义缓存器 ────────────────────────────────────────────────
# 分级阈值：从 Agent 行为反推类型，不需要 Router
_DEFAULT_THRESHOLDS = {
    "rag":      0.90,   # 知识问答要求严格匹配
    "web":      0.92,   # 网络搜索结果时效性强，更严格
    "chitchat": 0.85,   # 闲聊宽松一些
}


class SemanticCache:
    """
    基于工具行为推断的分级语义缓存。

    使用方式：
        cache = SemanticCache()

        # 查缓存（不需要知道类型，遍历所有条目按各自阈值匹配）
        if not is_followup(question):
            cached = cache.get(question)
            if cached is not None:
                return cached

        # 未命中，走 Agent 拿到答案后，推断类型再写缓存
        intent = infer_intent_from_tools(updated_messages)
        cache.set(question, answer, intent=intent)
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        max_items: int = 500,
    ) -> None:
        self.thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)
        self.max_items  = int(max_items)
        self._items: list[dict[str, Any]] = []

    def _threshold_for(self, intent: str) -> float:
        """获取指定类型的阈值，未知类型用最严格的 0.92。"""
        return self.thresholds.get(intent, 0.92)

    def get(self, question: str) -> str | None:
        """
        查询缓存。

        遍历所有缓存条目，每个条目按自身类型的阈值判断是否匹配。
        返回得分最高且超过阈值的答案，或 None。
        """
        if not isinstance(question, str) or not question.strip():
            return None

        embeddings = _get_embeddings()
        query = question.strip()
        query_embedding = embeddings.get_embedding(query)

        best_item: dict[str, Any] | None = None
        best_score = -1.0

        for item in self._items:
            score = embeddings.cosine_similarity(query_embedding, item["embedding"])
            item_threshold = self._threshold_for(item.get("intent", "chitchat"))

            if score >= item_threshold and score > best_score:
                best_score = score
                best_item = item

        if best_item is not None:
            return str(best_item["answer"])
        return None

    def set(self, question: str, answer: str, intent: str = "chitchat") -> None:
        """
        写入缓存。

        参数：
            question : 用户问题
            answer   : Agent 生成的答案
            intent   : 从工具行为推断的类型（"rag"/"web"/"chitchat"）
        """
        if not isinstance(question, str) or not question.strip():
            return
        if not isinstance(answer, str) or not answer.strip():
            return

        embeddings = _get_embeddings()
        q = question.strip()
        a = answer.strip()
        q_embedding = embeddings.get_embedding(q)

        # 查找是否已有语义等价条目
        best_index = -1
        best_score = -1.0
        for idx, item in enumerate(self._items):
            if item["question"] == q:
                # 精确命中，直接更新
                item["answer"]    = a
                item["intent"]    = intent
                item["embedding"] = q_embedding
                return

            score = embeddings.cosine_similarity(q_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_index = idx

        # 语义去重
        threshold = self._threshold_for(intent)
        if best_index >= 0 and best_score >= threshold:
            target = self._items[best_index]
            target["question"]  = q
            target["answer"]    = a
            target["intent"]    = intent
            target["embedding"] = q_embedding
            return

        # 新增条目
        self._items.append({
            "question":  q,
            "answer":    a,
            "intent":    intent,
            "embedding": q_embedding,
        })

        # 淘汰最旧
        if self.max_items > 0 and len(self._items) > self.max_items:
            overflow = len(self._items) - self.max_items
            del self._items[:overflow]

    def clear(self) -> None:
        """清空缓存。"""
        self._items.clear()
