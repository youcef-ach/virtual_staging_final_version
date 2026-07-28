"""API request / response schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr


class OrganizationCreate(BaseModel):
    name: str


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    created_at: datetime


class UserRegister(BaseModel):
    organization_name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    email: str
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class PropertyCreate(BaseModel):
    organization_id: UUID
    name: str
    address: str | None = None


class PropertyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    name: str
    address: str | None
    created_at: datetime


class Completeness(BaseModel):
    has_e57: bool
    has_matterpak: bool
    has_cubemaps: bool
    complete: bool


class ScanVersionOut(BaseModel):
    id: UUID
    property_id: UUID
    version: int
    status: str
    is_authoritative: bool
    raw_assets: dict[str, list[dict[str, Any]]]
    completeness: Completeness
    created_at: datetime


class CatalogItemCreate(BaseModel):
    sku: str
    manufacturer: str
    product_name: str
    category: str
    price: Decimal | None = None
    currency: str | None = "USD"
    purchase_url: str | None = None
    materials: dict[str, Any] | None = None
    dimensions_mm: dict[str, Any] | None = None
    weight_kg: float | None = None
    glb_url: str | None = None
    thumbnail_url: str | None = None
    bounding_box: dict[str, Any] | None = None
    polygon_count: int | None = None
    lod_urls: dict[str, Any] | None = None


class CatalogItemUpdate(BaseModel):
    sku: str | None = None
    manufacturer: str | None = None
    product_name: str | None = None
    category: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    purchase_url: str | None = None
    materials: dict[str, Any] | None = None
    dimensions_mm: dict[str, Any] | None = None
    weight_kg: float | None = None
    glb_url: str | None = None
    thumbnail_url: str | None = None
    bounding_box: dict[str, Any] | None = None
    polygon_count: int | None = None
    lod_urls: dict[str, Any] | None = None


class CatalogItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sku: str
    manufacturer: str
    product_name: str
    category: str
    price: Decimal | None
    currency: str | None
    purchase_url: str | None
    materials: dict[str, Any] | None
    dimensions_mm: dict[str, Any] | None
    weight_kg: float | None
    glb_url: str | None
    thumbnail_url: str | None
    bounding_box: dict[str, Any] | None
    polygon_count: int | None
    lod_urls: dict[str, Any] | None
    created_at: datetime


class ShareLinkOut(BaseModel):
    staging_project_id: UUID
    share_token: str
    share_url: str
    is_published: bool


# ---------------------------------------------------------------------------
# Pipeline orchestration schemas
# ---------------------------------------------------------------------------


class PipelineRunCreate(BaseModel):
    scan_version_id: UUID


class StageExecutionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    stage_number: int
    stage_name: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    log_path: str | None


class PipelineRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    scan_version_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class PipelineRunDetail(BaseModel):
    """Full run with all stage executions."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    scan_version_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime
    stages: list[StageExecutionOut]
