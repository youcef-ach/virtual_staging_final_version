"""Catalog management API endpoints (P1).

Provides catalog CRUD, file uploads, asset normalization pipeline invocation,
category filtering, and local static asset serving.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db
from app.schemas import CatalogItemCreate, CatalogItemOut, CatalogItemUpdate
from vs_pipeline.stages.catalog_normalize import normalize_catalog_asset

router = APIRouter(prefix="/catalog", tags=["catalog"])

CATALOG_STORAGE_ROOT = Path("backend/data/catalog").resolve()


@router.post("", response_model=CatalogItemOut, status_code=201)
def create_catalog_item(
    payload: CatalogItemCreate, db: Session = Depends(get_db)
) -> models.CatalogItem:
    """Create a catalog item record manually."""
    existing = db.scalar(
        select(models.CatalogItem).where(models.CatalogItem.sku == payload.sku)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"SKU '{payload.sku}' already exists")

    item = models.CatalogItem(
        sku=payload.sku,
        manufacturer=payload.manufacturer,
        product_name=payload.product_name,
        category=payload.category,
        price=payload.price,
        currency=payload.currency,
        purchase_url=payload.purchase_url,
        materials=payload.materials,
        dimensions_mm=payload.dimensions_mm,
        weight_kg=payload.weight_kg,
        glb_url=payload.glb_url,
        thumbnail_url=payload.thumbnail_url,
        bounding_box=payload.bounding_box,
        polygon_count=payload.polygon_count,
        lod_urls=payload.lod_urls,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.post("/upload", response_model=CatalogItemOut, status_code=201)
def upload_and_normalize_asset(
    file: UploadFile = File(...),
    sku: str = Form(...),
    manufacturer: str = Form("In-House"),
    product_name: str = Form(...),
    category: str = Form("furniture"),
    price: float | None = Form(None),
    scale: float | None = Form(None),
    db: Session = Depends(get_db),
) -> models.CatalogItem:
    """Upload a raw glTF/GLB file, run normalization pipeline, and save to catalog."""
    existing = db.scalar(
        select(models.CatalogItem).where(models.CatalogItem.sku == sku)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"SKU '{sku}' already exists")

    sku_dir = CATALOG_STORAGE_ROOT / sku
    sku_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        norm_result = normalize_catalog_asset(
            input_path=tmp_path,
            output_dir=sku_dir,
            sku=sku,
            category=category,
            manufacturer=manufacturer,
            product_name=product_name,
            price=price,
            scale_factor=scale,
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Asset normalization failed: {err}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    base_url = f"/api/catalog/files/{sku}"
    item = models.CatalogItem(
        sku=sku,
        manufacturer=manufacturer,
        product_name=product_name,
        category=category,
        price=norm_result.get("price"),
        currency="USD",
        dimensions_mm=norm_result.get("dimensions_mm"),
        bounding_box=norm_result.get("bounding_box"),
        polygon_count=norm_result.get("polygon_count"),
        glb_url=f"{base_url}/lod0.glb",
        thumbnail_url=f"{base_url}/thumbnail.png",
        lod_urls={
            "lod0": f"{base_url}/lod0.glb",
            "lod1": f"{base_url}/lod1.glb",
            "lod2": f"{base_url}/lod2.glb",
        },
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.get("", response_model=list[CatalogItemOut])
def list_catalog_items(
    category: str | None = Query(None, description="Filter by category"),
    search: str | None = Query(None, description="Search product name, manufacturer, or SKU"),
    db: Session = Depends(get_db),
) -> list[models.CatalogItem]:
    """List and filter catalog items."""
    stmt = select(models.CatalogItem)

    if category:
        stmt = stmt.where(models.CatalogItem.category == category)

    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(
            or_(
                models.CatalogItem.product_name.ilike(pattern),
                models.CatalogItem.manufacturer.ilike(pattern),
                models.CatalogItem.sku.ilike(pattern),
            )
        )

    stmt = stmt.order_by(models.CatalogItem.created_at.desc())
    return list(db.scalars(stmt))


@router.get("/{item_id}", response_model=CatalogItemOut)
def get_catalog_item(
    item_id: uuid.UUID, db: Session = Depends(get_db)
) -> models.CatalogItem:
    """Get single catalog item by ID."""
    item = db.get(models.CatalogItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    return item


@router.patch("/{item_id}", response_model=CatalogItemOut)
def update_catalog_item(
    item_id: uuid.UUID, payload: CatalogItemUpdate, db: Session = Depends(get_db)
) -> models.CatalogItem:
    """Update catalog item metadata."""
    item = db.get(models.CatalogItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Catalog item not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=204)
def delete_catalog_item(item_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    """Delete catalog item and its files."""
    item = db.get(models.CatalogItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Catalog item not found")

    sku_dir = CATALOG_STORAGE_ROOT / item.sku
    if sku_dir.exists():
        shutil.rmtree(sku_dir, ignore_errors=True)

    db.delete(item)
    db.commit()


@router.get("/files/{sku}/{filename}")
def serve_catalog_file(sku: str, filename: str) -> FileResponse:
    """Serve normalized catalog files (GLBs, thumbnails, metadata)."""
    safe_file = Path(filename).name
    file_path = CATALOG_STORAGE_ROOT / sku / safe_file

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = "model/gltf-binary" if safe_file.endswith(".glb") else None
    return FileResponse(path=file_path, media_type=media_type)
