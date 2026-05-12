from __future__ import annotations

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
