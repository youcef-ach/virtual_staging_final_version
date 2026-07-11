"""Organizations & properties (minimal CRUD; auth/multi-tenancy hardening lands in P2)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db
from app.schemas import OrganizationCreate, OrganizationOut, PropertyCreate, PropertyOut

router = APIRouter(tags=["properties"])


@router.post("/organizations", response_model=OrganizationOut, status_code=201)
def create_organization(
    payload: OrganizationCreate, db: Session = Depends(get_db)
) -> models.Organization:
    existing = db.scalar(
        select(models.Organization).where(models.Organization.name == payload.name)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="organization name already exists")
    org = models.Organization(name=payload.name)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


@router.get("/organizations", response_model=list[OrganizationOut])
def list_organizations(db: Session = Depends(get_db)) -> list[models.Organization]:
    return list(db.scalars(select(models.Organization).order_by(models.Organization.created_at)))


@router.post("/properties", response_model=PropertyOut, status_code=201)
def create_property(payload: PropertyCreate, db: Session = Depends(get_db)) -> models.Property:
    if db.get(models.Organization, payload.organization_id) is None:
        raise HTTPException(status_code=404, detail="organization not found")
    prop = models.Property(
        organization_id=payload.organization_id, name=payload.name, address=payload.address
    )
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return prop


@router.get("/properties", response_model=list[PropertyOut])
def list_properties(db: Session = Depends(get_db)) -> list[models.Property]:
    return list(db.scalars(select(models.Property).order_by(models.Property.created_at)))


@router.get("/properties/{property_id}", response_model=PropertyOut)
def get_property(property_id: uuid.UUID, db: Session = Depends(get_db)) -> models.Property:
    prop = db.get(models.Property, property_id)
    if prop is None:
        raise HTTPException(status_code=404, detail="property not found")
    return prop
