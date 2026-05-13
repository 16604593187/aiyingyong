from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
LAYER2_DIR = CURRENT_DIR.parent / "layer2"
EMBEDDINGS_FILE = LAYER2_DIR / "embeddings.py"
CHROMA_DIR = CURRENT_DIR / "chroma_db"
COLLECTION_NAME = "knowledge_base"

_embedding_utils: Any | None = None
_client: Any | None = None
_collection: Any | None = None


def _get_embedding_utils() -> Any:
    global _embedding_utils
    if _embedding_utils is not None:
        return _embedding_utils

    if str(LAYER2_DIR) not in sys.path:
        sys.path.insert(0, str(LAYER2_DIR))

    spec = importlib.util.spec_from_file_location("layer2_embeddings", EMBEDDINGS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load embeddings module from {EMBEDDINGS_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _embedding_utils = module
    return _embedding_utils


def _get_collection() -> Any:
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("chromadb package is required. Please install chromadb first.") from exc

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _normalize_metadata(metadata: dict) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        clean_key = str(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[clean_key] = value
            continue

        try:
            normalized[clean_key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            normalized[clean_key] = str(value)

    return normalized


def _build_doc_id(text: str, metadata: dict[str, Any]) -> str:
    payload = {
        "text": text,
        "metadata": metadata,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def add_documents(texts: list[str], metadatas: list[dict]) -> None:
    """把一批文本 Embedding 后存入向量库"""
    if len(texts) != len(metadatas):
        raise ValueError("texts and metadatas must have the same length")
    if not texts:
        return

    embedding_utils = _get_embedding_utils()
    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    normalized_metadatas: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for text, metadata in zip(texts, metadatas):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each text must be a non-empty string")
        if not isinstance(metadata, dict):
            raise ValueError("each metadata must be a dict")

        clean_text = text.strip()
        clean_metadata = _normalize_metadata(metadata)

        ids.append(_build_doc_id(clean_text, clean_metadata))
        documents.append(clean_text)
        normalized_metadatas.append(clean_metadata)
        vectors.append(embedding_utils.get_embedding(clean_text))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=normalized_metadatas,
        embeddings=vectors,
    )


def search(query: str, top_k: int = 5) -> list[dict]:
    """用一段文本检索最相关的 top_k 条记录

    返回格式：[{"text": "...", "metadata": {...}, "score": 0.85}, ...]
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        return []

    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding_utils = _get_embedding_utils()
    query_embedding = embedding_utils.get_embedding(query.strip())

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    output: list[dict] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        score = 1.0 - float(distance)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0

        output.append(
            {
                "text": text,
                "metadata": metadata or {},
                "score": score,
            }
        )

    return output


def reset_collection() -> None:
    """清空并重建 collection，用于全量重建知识库。"""
    global _collection

    _get_collection()
    if _client is None:
        raise RuntimeError("failed to initialize chroma client")

    _client.delete_collection(name=COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def count() -> int:
    """返回知识库中当前存储的文档数量"""
    return int(_get_collection().count())
