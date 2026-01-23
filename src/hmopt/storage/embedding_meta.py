"""Embedding metadata helpers for index persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from hmopt.llm.openai_like import OpenAIEmbeddingLike

logger = logging.getLogger(__name__)


def embedding_metadata_path(persist_dir: Path) -> Path:
    return persist_dir / "embedding_meta.json"


def persist_embedding_metadata(
    persist_dir: Path,
    *,
    model: str,
    dimension: int,
    index_name: str,
    node_label: str,
    version: Optional[str] = None,
    kind: Optional[str] = None,
    index_id: Optional[str] = None,
) -> None:
    metadata_path = embedding_metadata_path(persist_dir)
    metadata = {
        "model": model,
        "dimension": dimension,
        "index_name": index_name,
        "node_label": node_label,
    }
    if version:
        metadata["version"] = version
    if kind:
        metadata["kind"] = kind
    if index_id:
        metadata["index_id"] = index_id
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def load_embedding_metadata(persist_dir: Path) -> Optional[dict]:
    metadata_path = embedding_metadata_path(persist_dir)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Failed to parse embedding metadata at %s", metadata_path)
        return None


def infer_embedding_dimension(embed: OpenAIEmbeddingLike) -> int:
    return len(embed.get_text_embedding("dimension probe"))


def embedding_dimension_for_query(persist_dir: Path, embed: OpenAIEmbeddingLike) -> int:
    metadata = load_embedding_metadata(persist_dir)
    if metadata and metadata.get("dimension"):
        return int(metadata["dimension"])
    return infer_embedding_dimension(embed)


def ensure_embedding_compat(
    persist_dir: Path,
    *,
    embed_model: str,
    embed_dim: int,
    index_name: str,
    node_label: str,
    index_id: Optional[str] = None,
) -> None:
    metadata = load_embedding_metadata(persist_dir)
    if not metadata:
        logger.warning("Embedding metadata missing for index at %s", persist_dir)
        return
    stored_dim = metadata.get("dimension")
    stored_model = metadata.get("model")
    stored_index = metadata.get("index_name")
    stored_label = metadata.get("node_label")
    stored_index_id = metadata.get("index_id")
    if stored_dim != embed_dim:
        raise RuntimeError(
            "Embedding dimension mismatch for index at "
            f"{persist_dir}. Stored model={stored_model!r} dim={stored_dim}, "
            f"current model={embed_model!r} dim={embed_dim}. "
            "Rebuild the index or drop the Neo4j vector index before querying."
        )
    if stored_index and stored_index != index_name:
        raise RuntimeError(
            "Neo4j index name mismatch for index at "
            f"{persist_dir}. Stored index={stored_index!r}, current index={index_name!r}. "
            "Rebuild the index to match the current configuration."
        )
    if stored_label and stored_label != node_label:
        raise RuntimeError(
            "Neo4j node label mismatch for index at "
            f"{persist_dir}. Stored label={stored_label!r}, current label={node_label!r}. "
            "Rebuild the index to match the current configuration."
        )
    if stored_index_id and index_id and stored_index_id != index_id:
        raise RuntimeError(
            "Index ID mismatch for index at "
            f"{persist_dir}. Stored index_id={stored_index_id!r}, current index_id={index_id!r}. "
            "Rebuild the index to match the current configuration."
        )
