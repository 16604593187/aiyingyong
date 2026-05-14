from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        current_dir = Path(__file__).resolve().parent
        self.base_dir = base_dir or (current_dir / "history")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_file(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        return self.base_dir / f"{session_id.strip()}.json"

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """从磁盘加载历史，返回 messages 列表。"""
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return []

        try:
            payload = json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        messages = payload.get("messages", [])
        if isinstance(messages, list):
            return messages
        return []

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """把当前 messages 写入磁盘。"""
        session_file = self._session_file(session_id)
        now = self._now_iso()

        created_at = now
        if session_file.exists():
            try:
                existing = json.loads(session_file.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                created_at = now

        payload = {
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": now,
            "messages": messages,
        }
        session_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_sessions(self) -> list[str]:
        """列出所有已有的 session（按文件修改时间排序）。"""
        files = sorted(
            (path for path in self.base_dir.glob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        return [path.stem for path in files]

    def clear(self, session_id: str) -> bool:
        """删除某个 session 的历史，返回是否实际删除。"""
        session_file = self._session_file(session_id)
        if session_file.exists():
            session_file.unlink()
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 对话历史压缩器
# ──────────────────────────────────────────────────────────────────────────────
_SUMMARIZE_SYSTEM = (
    "你是一个对话摘要助手。"
    "任务：将给定的历史对话轮次压缩成一段简洁的摘要。"
    "要求：\n"
    "  1. 保留所有关键信息点、结论、用户表达的偏好和重要细节\n"
    "  2. 去掉寒暄、重复确认、无实质内容的过渡语\n"
    "  3. 以第三人称叙述（如\"用户询问了…\"\"助手解释了…\"）\n"
    "  4. 摘要要紧凑，不要超过 300 字\n"
    "  5. 直接输出摘要文本，不加任何前缀标签"
)

_SUMMARIZE_USER_TMPL = "请将以下对话历史压缩成摘要：\n\n{dialogue}"


def _format_dialogue(messages: list[dict]) -> str:
    """把 messages 列表格式化成易于 LLM 阅读的对话文本。"""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip()
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
        # system / tool 消息跳过，不参与摘要
    return "\n".join(lines)


class HistoryCompressor:
    """
    对话历史压缩器。

    策略：懒压缩（Lazy Compression）
      - 对话过程中只做滑动窗口截断（保证 API 不超限）
      - 当完整历史超过 compress_threshold 轮时，才触发一次压缩
      - 压缩：把最旧的 compress_turns 轮历史（user+assistant 对）摘要成一条
        role=assistant、content="[历史摘要]…" 的合成消息，替换掉原来的多条消息
      - 压缩后合成消息排在历史最前面，近期原文完整保留

    使用方式：
        compressor = HistoryCompressor(client, model)

        # 每轮对话后判断是否需要压缩
        if compressor.should_compress(history):
            history = compressor.compress(history)

        # 也可以手动触发
        history = compressor.compress(history)
    """

    def __init__(
        self,
        client: Any,
        model: str,
        compress_threshold: int = 20,   # 超过多少条消息才触发自动压缩
        compress_turns: int = 10,       # 每次压缩最旧的多少轮（user+assistant 对）
    ) -> None:
        """
        参数：
            client            : OpenAI 客户端实例
            model             : 用于摘要的模型名称
            compress_threshold: 历史消息条数超过该值时触发懒压缩（默认 20 条 = 10 轮）
            compress_turns    : 每次压缩最旧的多少轮（默认 10 轮 = 20 条消息）
        """
        if compress_turns <= 0:
            raise ValueError("compress_turns must be positive")
        if compress_threshold <= 0:
            raise ValueError("compress_threshold must be positive")

        self.client = client
        self.model = model
        self.compress_threshold = compress_threshold
        self.compress_turns = compress_turns

    def should_compress(self, history: list[dict]) -> bool:
        """判断当前历史是否超过阈值，需要触发压缩。"""
        return len(history) > self.compress_threshold

    def compress(self, history: list[dict]) -> list[dict]:
        """
        对 history 做一次懒压缩。

        流程：
          1. 取出最旧的 compress_turns 轮（compress_turns*2 条）消息
          2. 过滤出 role=user/assistant 的消息格式化成对话文本
          3. 调用 LLM 生成摘要
          4. 用一条合成的 system 消息替换掉那些旧消息
          5. 返回新的 history（合成摘要 + 剩余近期原文）

        如果历史条数不足 compress_turns 轮，则对所有历史做摘要，
        结果是一条合成消息（完全压缩模式）。
        """
        if not history:
            return history

        cut = self.compress_turns * 2  # 要压缩的消息条数

        # 如果历史比 cut 还短，全部压缩
        to_compress = history[:cut]
        remaining   = history[cut:]

        dialogue = _format_dialogue(to_compress)
        if not dialogue.strip():
            # 被压缩段没有有效内容（全是 system/tool），直接丢弃
            return remaining

        summary = self._call_llm(dialogue)

        # 合成摘要消息：用 role=system 便于区分，前缀标注这是摘要
        summary_msg: dict = {
            "role":    "system",
            "content": f"[以下是早期对话的摘要，供参考]\n{summary}",
        }

        return [summary_msg] + remaining

    def _call_llm(self, dialogue: str) -> str:
        """调用 LLM 生成摘要，失败时返回原始对话文本（降级）。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM},
                    {
                        "role":    "user",
                        "content": _SUMMARIZE_USER_TMPL.format(dialogue=dialogue),
                    },
                ],
                temperature=0.3,
                max_tokens=400,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            # 压缩失败时降级：直接用截断的对话文本作为摘要
            max_len = 600
            return dialogue[:max_len] + ("…（摘要生成失败，以下为原文节选）" if len(dialogue) > max_len else "")
