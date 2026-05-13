from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from openai import OpenAI

CURRENT_DIR = Path(__file__).resolve().parent
LAYER1_DIR = CURRENT_DIR.parent / "layer1"
if str(LAYER1_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER1_DIR))

CONFIG_FILE = LAYER1_DIR / "config_mine.py"
_config_spec = importlib.util.spec_from_file_location("layer1_config_mine", CONFIG_FILE)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"cannot load config module from {CONFIG_FILE}")

config_mine = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config_mine)

client = OpenAI(
    api_key=config_mine.OPENAI_API_KEY,
    base_url=config_mine.OPENAI_API_BASE,
)

_ALLOWED_LABELS = {"chitchat", "rag", "followup"}


def _normalize_label(raw: str) -> str:
    normalized = raw.strip().lower().replace("`", "")
    normalized = normalized.replace("\n", " ")

    if normalized in _ALLOWED_LABELS:
        return normalized

    keywords = {
        "chitchat": ["chitchat", "chat", "闲聊", "问候", "感谢", "寒暄"],
        "followup": ["followup", "follow up", "追问", "继续", "展开", "举例", "补充"],
        "rag": ["rag", "知识库", "检索", "查询", "问答"],
    }

    for label, words in keywords.items():
        for word in words:
            if word in normalized:
                return label

    return ""


def classify(user_input: str) -> str:
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    system_prompt = (
        "你是一个意图分类器。"
        "你只能返回以下三个标签之一：chitchat、rag、followup。"
        "定义如下："
        "chitchat：闲聊、问候、感谢等不需要知识支撑的输入；"
        "rag：需要查知识库才能回答的问题；"
        "followup：追问上一轮内容，如“能展开说说吗”“举个例子”。"
        "严格要求：只返回标签本身，不能有任何多余文本、标点或解释。"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input.strip()},
        ],
        temperature=0,
        max_tokens=8,
        stream=False,
    )

    raw_label = response.choices[0].message.content or ""
    label = _normalize_label(raw_label)
    if label:
        return label

    # 二次规范化：尝试从整段输出中抽取合法标签
    compact = " ".join(raw_label.strip().lower().split())
    for allowed in _ALLOWED_LABELS:
        if allowed in compact:
            return allowed

    # 最终兜底：未识别时按知识型问题处理
    return "rag"
