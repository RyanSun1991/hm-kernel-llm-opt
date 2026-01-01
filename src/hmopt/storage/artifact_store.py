"""Content-addressed artifact store."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import logging
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from .db import models

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Stores artifacts on disk and registers metadata in the DB."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _artifact_path(self, sha: str, extension: str | None) -> Path:
        ext = extension if extension and extension.startswith(".") else f".{extension}" if extension else ""
        return self.root / sha[:2] / f"{sha}{ext}"

    def store_bytes(
        self,
        data: bytes,
        kind: str,
        extension: str | None = None,
        *,
        run_id: Optional[str] = None,
        mime: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> models.Artifact:
        sha = self._hash_bytes(data)
        path = self._artifact_path(sha, extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        artifact = models.Artifact(
            artifact_id=sha,
            run_id=run_id,
            kind=kind,
            sha256=sha,
            path=str(path),
            bytes=len(data),
            mime=mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            metadata_json=metadata or {},
        )
        if session is not None:
            session.add(artifact)
        logger.info("Stored artifact: kind=%s sha=%s bytes=%d", kind, sha, len(data))
        return artifact

    def store_text(
        self,
        text: str,
        kind: str,
        *,
        run_id: Optional[str] = None,
        extension: str = ".txt",
        session: Optional[Session] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> models.Artifact:
        return self.store_bytes(
            text.encode("utf-8"),
            kind=kind,
            extension=extension,
            run_id=run_id,
            mime="text/plain",
            metadata=metadata,
            session=session,
        )

    def store_json(
        self,
        obj: Any,
        kind: str,
        *,
        run_id: Optional[str] = None,
        session: Optional[Session] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> models.Artifact:
        data = json.dumps(obj, indent=2, ensure_ascii=False)
        return self.store_text(
            data,
            kind=kind,
            run_id=run_id,
            extension=".json",
            session=session,
            metadata=metadata,
        )

    def store_file(
        self,
        src: Path,
        kind: str,
        *,
        run_id: Optional[str] = None,
        session: Optional[Session] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> models.Artifact:
        data = Path(src).read_bytes()
        extension = Path(src).suffix
        return self.store_bytes(
            data,
            kind=kind,
            extension=extension,
            run_id=run_id,
            mime=mimetypes.guess_type(str(src))[0],
            session=session,
            metadata=metadata,
        )

    def resolve_path(self, artifact: models.Artifact) -> Path:
        """Return the on-disk path for a stored artifact."""
        return Path(artifact.path)
