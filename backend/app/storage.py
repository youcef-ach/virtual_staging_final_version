"""Immutable raw asset storage.

Dev backend: local filesystem. Production backend: S3 (same key layout), wired in P4.
Key layout (PLAN section 2): ``raw/{property_id}/v{version}/{filename}``.
Existing keys are NEVER overwritten — raw scan data is write-once.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class StoredAsset:
    key: str
    sha256: str
    size_bytes: int


class Storage(Protocol):
    def store_raw(
        self, property_id: UUID, version: int, filename: str, stream: BinaryIO
    ) -> StoredAsset: ...


class LocalStorage:
    """Filesystem storage using the exact key layout of the production S3 bucket."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def store_raw(
        self, property_id: UUID, version: int, filename: str, stream: BinaryIO
    ) -> StoredAsset:
        safe_name = Path(filename).name  # strip any client-supplied path components
        if not safe_name:
            raise ValueError("empty filename")
        key = f"raw/{property_id}/v{version}/{safe_name}"
        dest = self._root / key
        if dest.exists():
            raise FileExistsError(f"immutable storage: key already exists: {key}")
        dest.parent.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256()
        size = 0
        with dest.open("wb") as out:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
        return StoredAsset(key=key, sha256=digest.hexdigest(), size_bytes=size)
