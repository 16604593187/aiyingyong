from __future__ import annotations

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
HISTORY_DIR = CURRENT_DIR / "history"
NOTES_DIR = CURRENT_DIR / "test" / "notes"
EXAM_DIR = CURRENT_DIR / "test" / "exam"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_files_by_suffix(path: Path, suffix: str) -> int:
    """删除目录内指定后缀文件，保留目录结构。"""
    _ensure_dir(path)

    removed = 0
    for file_path in path.rglob(f"*{suffix}"):
        if file_path.is_file():
            file_path.unlink()
            removed += 1
    return removed


def clear_layer7_data() -> dict[str, int]:
    """
    清空 layer7 运行产物：
    - history/*.json
    - test/notes/*.txt
    - test/exam/*.txt

    只删除文件，不删除目录。
    """
    history_removed = _clear_files_by_suffix(HISTORY_DIR, ".json")
    notes_removed = _clear_files_by_suffix(NOTES_DIR, ".txt")
    exam_removed = _clear_files_by_suffix(EXAM_DIR, ".txt")

    return {
        "history_removed": history_removed,
        "notes_removed": notes_removed,
        "exam_removed": exam_removed,
        "total_removed": history_removed + notes_removed + exam_removed,
    }


def main() -> None:
    result = clear_layer7_data()
    print("layer7 cleanup done")
    print(f"- history removed: {result['history_removed']}")
    print(f"- notes removed: {result['notes_removed']}")
    print(f"- exam removed: {result['exam_removed']}")
    print(f"- total removed: {result['total_removed']}")


if __name__ == "__main__":
    main()
