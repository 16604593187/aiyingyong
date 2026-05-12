from __future__ import annotations

from pathlib import Path

import knowledge_base

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]


def _recursive_split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if separator not in text:
        return _recursive_split_by_separators(text, separators[1:], chunk_size)

    parts = text.split(separator)
    splits: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            continue

        candidate = part + (separator if index < len(parts) - 1 else "")
        if len(candidate) <= chunk_size:
            splits.append(candidate)
        else:
            splits.extend(_recursive_split_by_separators(candidate, separators[1:], chunk_size))

    return splits


def _merge_with_overlap(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for split in splits:
        piece = split.strip()
        if not piece:
            continue

        if not current:
            current = piece
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue

        chunks.append(current)

        if overlap > 0:
            tail = current[-overlap:]
            current = (tail + piece).strip()
        else:
            current = piece

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            if overlap > 0:
                tail = current[chunk_size - overlap : chunk_size]
                current = (tail + current[chunk_size:]).strip()
            else:
                current = current[chunk_size:].strip()

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    normalized = text.strip()
    if not normalized:
        return []

    splits = _recursive_split_by_separators(normalized, SEPARATORS, chunk_size)
    return _merge_with_overlap(splits, chunk_size, overlap)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return recursive_split(text, chunk_size=chunk_size, overlap=overlap)


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
        pieces = recursive_split(content)
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
