"""API request / response schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class OrganizationCreate(BaseModel):
    name: str


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    created_at: datetime


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
