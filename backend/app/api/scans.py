"""Scan version ingestion — pipeline Stage 1 (ingest & validate).

Multipart upload -> validate -> stream to immutable storage with SHA-256 ->
register scan_version (status INGESTED, promoted to authoritative).
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app import models
from app.config import get_settings
from app.db import get_db
from app.schemas import Completeness, ScanVersionOut
from app.storage import LocalStorage

router = APIRouter(tags=["scans"])

_ALLOWED = {
    "e57_files": {".e57"},
    "matterpak_files": {".obj", ".mtl", ".jpg", ".jpeg", ".png", ".xyz", ".pdf", ".zip"},
    "cubemap_files": {".jpg", ".jpeg", ".png", ".ktx2", ".zip"},
}


def _validate_suffixes(files: list[UploadFile], field: str) -> None:
    allowed = _ALLOWED[field]
    for f in files:
        name = (f.filename or "").lower()
        if not any(name.endswith(suffix) for suffix in allowed):
            raise HTTPException(
                status_code=422,
                detail=f"{field}: unexpected file type '{f.filename}' "
                f"(allowed: {sorted(allowed)})",
            )


def _completeness(categories: dict[str, list[UploadFile]] | dict[str, list[dict]]) -> Completeness:
    has_e57 = bool(categories.get("e57"))
    has_matterpak = bool(categories.get("matterpak"))
    has_cubemaps = bool(categories.get("cubemaps"))
    return Completeness(
        has_e57=has_e57,
        has_matterpak=has_matterpak,
        has_cubemaps=has_cubemaps,
        complete=has_e57 and has_matterpak and has_cubemaps,
    )


def _to_out(scan: models.ScanVersion) -> ScanVersionOut:
    return ScanVersionOut(
        id=scan.id,
        property_id=scan.property_id,
        version=scan.version,
        status=scan.status.value,
        is_authoritative=scan.is_authoritative,
        raw_assets=scan.raw_assets,
        completeness=_completeness(scan.raw_assets),
        created_at=scan.created_at,
    )


@router.post(
    "/properties/{property_id}/scan-versions",
    response_model=ScanVersionOut,
    status_code=201,
)
def create_scan_version(
    property_id: uuid.UUID,
    e57_files: list[UploadFile] = File(..., description="One or more .e57 files"),
    matterpak_files: list[UploadFile] = File(
        default=[], description="Matterpak: OBJ/MTL/textures, cloud.xyz, floorplans (or .zip)"
    ),
    cubemap_files: list[UploadFile] = File(
        default=[], description="4K cubemap faces (or .zip)"
    ),
    db: Session = Depends(get_db),
) -> ScanVersionOut:
    if db.get(models.Property, property_id) is None:
        raise HTTPException(status_code=404, detail="property not found")

    categories: dict[str, list[UploadFile]] = {
        "e57": e57_files,
        "matterpak": matterpak_files,
        "cubemaps": cubemap_files,
    }
    _validate_suffixes(categories["e57"], "e57_files")
    _validate_suffixes(categories["matterpak"], "matterpak_files")
    _validate_suffixes(categories["cubemaps"], "cubemap_files")

    settings = get_settings()
    if settings.require_complete_ingest and not _completeness(categories).complete:
        raise HTTPException(
            status_code=422,
            detail="incomplete upload: e57 + Matterpak + cubemaps are all required",
        )

    version = (
        db.scalar(
            select(func.max(models.ScanVersion.version)).where(
                models.ScanVersion.property_id == property_id
            )
        )
        or 0
    ) + 1

    storage = LocalStorage(settings.storage_root)
    raw_assets: dict[str, list[dict[str, object]]] = {}
    for category, files in categories.items():
        for f in files:
            try:
                stored = storage.store_raw(
                    property_id, version, f.filename or "unnamed", f.file
                )
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raw_assets.setdefault(category, []).append(
                {
                    "filename": Path(f.filename or "unnamed").name,
                    "key": stored.key,
                    "sha256": stored.sha256,
                    "size_bytes": stored.size_bytes,
                }
            )

    # Promote this version; demote previous authoritative versions (raw data stays immutable).
    db.execute(
        update(models.ScanVersion)
        .where(
            models.ScanVersion.property_id == property_id,
            models.ScanVersion.is_authoritative.is_(True),
        )
        .values(is_authoritative=False)
    )
    scan = models.ScanVersion(
        property_id=property_id,
        version=version,
        status=models.ScanStatus.INGESTED,
        is_authoritative=True,
        raw_assets=raw_assets,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return _to_out(scan)


@router.get(
    "/properties/{property_id}/scan-versions",
    response_model=list[ScanVersionOut],
)
def list_scan_versions(
    property_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[ScanVersionOut]:
    scans = db.scalars(
        select(models.ScanVersion)
        .where(models.ScanVersion.property_id == property_id)
        .order_by(models.ScanVersion.version)
    )
    return [_to_out(s) for s in scans]
