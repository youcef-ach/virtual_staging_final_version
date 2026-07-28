"""Authentication API endpoints (P2).

Handles organization registration, user login, token generation, and user profile retrieval.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.db import get_db
from app.schemas import TokenResponse, UserLogin, UserOut, UserRegister

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserRegister, db: Session = Depends(get_db)) -> TokenResponse:
    """Register a new organization and admin user."""
    existing_user = db.scalar(select(models.User).where(models.User.email == payload.email))
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{payload.email}' is already registered",
        )

    # Find or create organization
    org = db.scalar(
        select(models.Organization).where(models.Organization.name == payload.organization_name)
    )
    if org is None:
        org = models.Organization(name=payload.organization_name)
        db.add(org)
        db.flush()

    user = models.User(
        organization_id=org.id,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=models.UserRole.ADMIN,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user_id=user.id, organization_id=org.id, role=user.role)
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    """Authenticate user credentials and return a JWT access token."""
    user = db.scalar(select(models.User).where(models.User.email == payload.email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        user_id=user.id, organization_id=user.organization_id, role=user.role
    )
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def get_user_profile(user: models.User = Depends(get_current_user)) -> models.User:
    """Get authenticated user profile."""
    return user
