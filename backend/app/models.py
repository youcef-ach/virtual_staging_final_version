"""Database schema (PostgreSQL 16; JSONB / REAL[] with SQLite fallbacks for tests).

Table set per PLAN.md section 4. ``scan_versions`` raw data is immutable:
only ``status`` and ``is_authoritative`` flags may change after ingestion;
raw files themselves are write-once (see app/storage.py).

All geometry (poses, plane coefficients, placements) is stored in the runtime
Y-up frame — the Coordinate Lock Rule conversion happened at ingestion.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Cross-dialect types: real JSONB / REAL[] on PostgreSQL, JSON on SQLite (unit tests).
JSONB = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
REAL_ARRAY = sa.ARRAY(sa.REAL()).with_variant(sa.JSON(), "sqlite")


def _enum(e: type[enum.Enum]) -> sa.Enum:
    return sa.Enum(e, native_enum=False, length=32, values_callable=lambda x: [m.value for m in x])


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    AGENT = "agent"
    VIEWER = "viewer"


class ScanStatus(str, enum.Enum):
    INGESTED = "ingested"
    PROCESSING = "processing"
    FAILED = "failed"
    PUBLISHED = "published"


class QAStatus(str, enum.Enum):
    PENDING = "pending"
    FLAGGED = "flagged"
    CORRECTED = "corrected"
    APPROVED = "approved"


class PublishStage(str, enum.Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    PUBLISHED = "published"


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(sa.String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(sa.String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(sa.String(255))
    role: Mapped[UserRole] = mapped_column(_enum(UserRole), default=UserRole.VIEWER)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class Property(Base):
    __tablename__ = "properties"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(sa.String(255))
    address: Mapped[str | None] = mapped_column(sa.String(512))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class ScanVersion(Base):
    __tablename__ = "scan_versions"
    __table_args__ = (sa.UniqueConstraint("property_id", "version"),)

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    property_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("properties.id"), index=True)
    version: Mapped[int] = mapped_column(sa.Integer)
    status: Mapped[ScanStatus] = mapped_column(_enum(ScanStatus), default=ScanStatus.INGESTED)
    is_authoritative: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    #: {category: [{filename, key, sha256, size_bytes}]}; categories: e57 / matterpak / cubemaps
    raw_assets: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class ScanPosition(Base):
    __tablename__ = "scan_positions"
    __table_args__ = (sa.UniqueConstraint("scan_version_id", "sweep_uuid"),)

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    scan_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("scan_versions.id"), index=True)
    sweep_uuid: Mapped[str] = mapped_column(sa.String(64))
    sweep_index: Mapped[int] = mapped_column(sa.Integer)
    #: 4x4 pose, row-major REAL[16], runtime Y-up frame (Coordinate Lock Rule).
    pose_rt: Mapped[list[float]] = mapped_column(REAL_ARRAY)
    pano_url: Mapped[str | None] = mapped_column(sa.String(1024))
    ibl_url: Mapped[str | None] = mapped_column(sa.String(1024))
    gbuffer_url: Mapped[str | None] = mapped_column(sa.String(1024))
    #: Viewer hotspot placement: {"position": [x, y, z], ...}
    hotspot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class StructuralPlane(Base):
    __tablename__ = "structural_planes"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    scan_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("scan_versions.id"), index=True)
    #: Stable across stage-5 re-runs (idempotency; Year-2 BIM export relies on it).
    stable_key: Mapped[str] = mapped_column(sa.String(64))
    semantic_class: Mapped[str] = mapped_column(sa.String(32))  # floor / wall_NN / ceiling / ...
    #: Plane equation ax + by + cz + d = 0, REAL[4], runtime Y-up frame.
    coefficients: Mapped[list[float]] = mapped_column(REAL_ARRAY)
    boundary_geojson: Mapped[dict[str, Any]] = mapped_column(JSONB)
    openings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class NavigationGraphRecord(Base):
    __tablename__ = "navigation_graph"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    scan_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("scan_versions.id"), unique=True)
    #: {"nodes": [...], "edges": [{"a": ..., "b": ..., "is_portal": bool}]}
    graph: Mapped[dict[str, Any]] = mapped_column(JSONB)


class CatalogItem(Base):
    __tablename__ = "catalog"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    sku: Mapped[str] = mapped_column(sa.String(64), unique=True)
    manufacturer: Mapped[str] = mapped_column(sa.String(255))
    product_name: Mapped[str] = mapped_column(sa.String(255))
    category: Mapped[str] = mapped_column(sa.String(64), index=True)
    price: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 2))
    currency: Mapped[str | None] = mapped_column(sa.String(3))
    purchase_url: Mapped[str | None] = mapped_column(sa.String(1024))
    materials: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    dimensions_mm: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    weight_kg: Mapped[float | None] = mapped_column(sa.Float)
    glb_url: Mapped[str | None] = mapped_column(sa.String(1024))
    thumbnail_url: Mapped[str | None] = mapped_column(sa.String(1024))
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    polygon_count: Mapped[int | None] = mapped_column(sa.Integer)
    lod_urls: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class StagingProject(Base):
    __tablename__ = "staging_projects"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    property_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("properties.id"), index=True)
    name: Mapped[str] = mapped_column(sa.String(255))
    #: [{"sku": ..., "position": [x, y, z], "rotation": [x, y, z, w]}], runtime Y-up frame.
    placement_payload: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    is_published: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    active_sweep_uuid: Mapped[str | None] = mapped_column(sa.String(64))
    share_token: Mapped[str | None] = mapped_column(sa.String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class QAReview(Base):
    __tablename__ = "qa_reviews"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    scan_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("scan_versions.id"), index=True)
    gate: Mapped[int] = mapped_column(sa.SmallInteger)  # 1 = pose QA, 2 = segmentation QA
    sweep_uuid: Mapped[str | None] = mapped_column(sa.String(64))
    score: Mapped[float | None] = mapped_column(sa.Float)  # e.g. edge-overlap score
    status: Mapped[QAStatus] = mapped_column(_enum(QAStatus), default=QAStatus.PENDING)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


class PublishStateRecord(Base):
    __tablename__ = "publish_states"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    scan_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("scan_versions.id"), unique=True)
    state: Mapped[PublishStage] = mapped_column(_enum(PublishStage), default=PublishStage.DRAFT)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
