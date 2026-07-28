"""Public share links API endpoints (P2).

Generates unguessable share tokens for published staging projects and serves
unauthenticated public share payloads to external viewers.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.db import get_db
from app.schemas import ShareLinkOut

router = APIRouter(tags=["share"])


@router.post(
    "/api/staging-projects/{project_id}/share",
    response_model=ShareLinkOut,
    status_code=status.HTTP_200_OK,
)
def create_or_get_share_link(
    project_id: uuid.UUID, db: Session = Depends(get_db)
) -> ShareLinkOut:
    """Generate or retrieve a public share token for a staging project."""
    project = db.get(models.StagingProject, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Staging project not found")

    if not project.share_token:
        project.share_token = secrets.token_urlsafe(24)
        project.is_published = True
        db.commit()
        db.refresh(project)

    share_url = f"/share/{project.share_token}"
    return ShareLinkOut(
        staging_project_id=project.id,
        share_token=project.share_token,
        share_url=share_url,
        is_published=project.is_published,
    )


@router.get("/api/share/{share_token}")
def get_public_share_payload(
    share_token: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Retrieve published staging project payload by public share token (unauthenticated)."""
    project = db.scalar(
        select(models.StagingProject).where(models.StagingProject.share_token == share_token)
    )
    if project is None or not project.is_published:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared project not found or not published",
        )

    # Fetch associated property name
    prop = db.get(models.Property, project.property_id)

    return {
        "project_id": str(project.id),
        "property_id": str(project.property_id),
        "property_name": prop.name if prop else "Shared Property",
        "project_name": project.name,
        "active_sweep_uuid": project.active_sweep_uuid,
        "placement_payload": project.placement_payload,
        "is_published": project.is_published,
        "share_token": project.share_token,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }
