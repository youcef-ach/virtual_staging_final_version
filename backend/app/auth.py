"""Authentication & Authorization module (P2).

Provides password hashing, JWT token creation & verification, and FastAPI security dependencies
for multi-tenancy and Role-Based Access Control (RBAC).
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models
from app.config import get_settings
from app.db import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash password using bcrypt with fallback to HMAC-SHA256."""
    try:
        return pwd_context.hash(password)
    except Exception:
        # Fallback for environments with restricted C-extensions
        settings = get_settings()
        key = settings.jwt_secret_key.encode("utf-8")
        return "sha256$" + hmac.new(key, password.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    if hashed_password.startswith("sha256$"):
        settings = get_settings()
        key = settings.jwt_secret_key.encode("utf-8")
        expected = "sha256$" + hmac.new(key, plain_password.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, hashed_password)
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(
    user_id: UUID,
    organization_id: UUID,
    role: models.UserRole,
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token containing sub, org_id, role, and exp."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    payload = {
        "sub": str(user_id),
        "org_id": str(organization_id),
        "role": role.value,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    """FastAPI dependency: extract user from Bearer JWT token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user ID format")

    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


def require_roles(*allowed_roles: models.UserRole):
    """FastAPI dependency factory: enforce role-based access control."""

    def role_checker(user: models.User = Depends(get_current_user)) -> models.User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User role '{user.role.value}' lacks permission for this action",
            )
        return user

    return role_checker
