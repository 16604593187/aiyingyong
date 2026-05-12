from __future__ import annotations

from pathlib import Path

import knowledge_base

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    normalized = text.strip()
    if not normalized:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []

    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunk = normalized[start:end]
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def collect_source_files(data_dir: Path) -> list[Path]:
    md_files = sorted(data_dir.rglob("*.md"))
    txt_files = sorted(data_dir.rglob("*.txt"))
    return md_files + txt_files


def ingest_data() -> int:
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "data"

    files = collect_source_files(data_dir)
    if not files:
        return 0

    texts: list[str] = []
    metadatas: list[dict] = []

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        pieces = chunk_text(content)
        source_rel = file_path.relative_to(data_dir).as_posix()

        for idx, piece in enumerate(pieces):
            texts.append(piece)
            metadatas.append(
                {
                    "source": source_rel,
                    "chunk_index": idx,
                    "chunk_size": len(piece),
                }
            )

    before = knowledge_base.count()
    knowledge_base.add_documents(texts, metadatas)
    after = knowledge_base.count()
    return max(0, after - before)


def main() -> None:
    inserted = ingest_data()
    print(f"ingest 完成，新增切片数: {inserted}")
    print(f"当前知识库文档数: {knowledge_base.count()}")


if __name__ == "__main__":
    main()
