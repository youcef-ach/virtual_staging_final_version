"""Storage abstraction for pipeline publish stage.

Provides a unified interface for uploading published artifacts to an S3 bucket
(or copying to a local directory for development).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class StorageProvider(Protocol):
    def upload_file(self, local_path: Path, object_key: str, content_type: str) -> str:
        """Upload a file and return its public URL."""
        ...


class LocalStorageProvider:
    """Development fallback: copies files to a local directory."""

    def __init__(self, publish_root: Path, base_url: str = "/data") -> None:
        self.publish_root = publish_root
        self.base_url = base_url
        self.publish_root.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, object_key: str, content_type: str) -> str:
        dest_path = self.publish_root / object_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_path)
        logger.debug("Copied %s -> %s", local_path, dest_path)
        return f"{self.base_url}/{object_key}"


class S3StorageProvider:
    """Production provider: uploads files to an S3 bucket."""

    def __init__(
        self, bucket_name: str, cdn_base_url: str, aws_profile: str | None = None
    ) -> None:
        import boto3

        self.bucket_name = bucket_name
        self.cdn_base_url = cdn_base_url
        session = boto3.Session(profile_name=aws_profile)
        self.s3_client = session.client("s3")

    def upload_file(self, local_path: Path, object_key: str, content_type: str) -> str:
        logger.info("Uploading %s to s3://%s/%s", local_path.name, self.bucket_name, object_key)
        self.s3_client.upload_file(
            str(local_path),
            self.bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type, "CacheControl": "max-age=31536000"},
        )
        return f"{self.cdn_base_url}/{object_key}"


def get_storage_provider() -> StorageProvider:
    """Return the configured storage provider based on environment variables."""
    bucket = os.environ.get("VS_S3_BUCKET")
    if bucket:
        cdn = os.environ.get("VS_CDN_URL", f"https://{bucket}.s3.amazonaws.com")
        profile = os.environ.get("VS_AWS_PROFILE")
        return S3StorageProvider(bucket, cdn, profile)

    # Fallback to local
    local_dir = Path(os.environ.get("VS_LOCAL_PUBLISH_DIR", "../viewer/public/data")).resolve()
    return LocalStorageProvider(local_dir)
