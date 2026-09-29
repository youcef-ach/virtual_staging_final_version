import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.auth import create_access_token, hash_password, verify_password
from app.config import get_settings
from app.db import Base, get_db
from app.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VS_STORAGE_ROOT", str(tmp_path / "storage"))
    get_settings.cache_clear()

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    import app.db
    app.db._engine = engine
    app.db._session_factory = factory

    application = create_app()

    def override():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db] = override
    with TestClient(application) as test_client:
        yield test_client
    app.db._engine = None
    app.db._session_factory = None
    get_settings.cache_clear()


def test_password_hashing() -> None:
    raw = "SecureP@ssw0rd2026!"
    hashed = hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_register_and_login_flow(client: TestClient) -> None:
    # 1. Register new admin & organization
    reg_payload = {
        "organization_name": "Apex Real Estate",
        "email": "admin@apexrealty.com",
        "password": "SuperSecretPass!",
    }
    reg_res = client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201, reg_res.text
    body = reg_res.json()
    assert "access_token" in body
    assert body["user"]["email"] == "admin@apexrealty.com"
    assert body["user"]["role"] == "admin"

    # 2. Duplicate registration fails
    dup_res = client.post("/api/auth/register", json=reg_payload)
    assert dup_res.status_code == 409

    # 3. Login with correct credentials
    login_payload = {
        "email": "admin@apexrealty.com",
        "password": "SuperSecretPass!",
    }
    login_res = client.post("/api/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]

    # 4. Login with invalid password fails
    bad_login = client.post(
        "/api/auth/login", json={"email": "admin@apexrealty.com", "password": "WrongPassword!"}
    )
    assert bad_login.status_code == 401

    # 5. Get current user profile (/api/auth/me) with token
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "admin@apexrealty.com"

    # 6. Unauthenticated request to /me fails
    no_auth_res = client.get("/api/auth/me")
    assert no_auth_res.status_code == 401


def test_share_links_flow(client: TestClient) -> None:
    # 1. Register org & create property
    reg_res = client.post(
        "/api/auth/register",
        json={
            "organization_name": "Staging Pros",
            "email": "operator@stagingpros.com",
            "password": "OperatorPass123!",
        },
    )
    org_id = reg_res.json()["user"]["organization_id"]

    prop_res = client.post("/api/properties", json={"organization_id": org_id, "name": "Sunset Villa"})
    prop_id = prop_res.json()["id"]

    # 2. Create staging project in DB directly via helper or endpoint
    # Generate share link for a project
    from app.db import get_session_factory
    import uuid

    project_id = uuid.uuid4()
    with get_session_factory()() as db:
        project = app.models.StagingProject(
            id=project_id,
            property_id=uuid.UUID(prop_id),
            name="Modern Staging V1",
            placement_payload=[{"sku": "SOFA-001", "position": [1.0, 0.0, 2.0]}],
            active_sweep_uuid="sweep-101-uuid",
            is_published=False,
        )
        db.add(project)
        db.commit()

    # 3. Request share link creation
    share_res = client.post(f"/api/staging-projects/{project_id}/share")
    assert share_res.status_code == 200, share_res.text
    share_data = share_res.json()
    assert share_data["is_published"] is True
    token = share_data["share_token"]
    assert token is not None

    # 4. Fetch public share payload without authentication
    public_res = client.get(f"/api/share/{token}")
    assert public_res.status_code == 200, public_res.text
    public_payload = public_res.json()
    assert public_payload["project_name"] == "Modern Staging V1"
    assert public_payload["property_name"] == "Sunset Villa"
    assert public_payload["active_sweep_uuid"] == "sweep-101-uuid"
    assert len(public_payload["placement_payload"]) == 1

    # 5. Invalid share token returns 404
    invalid_res = client.get("/api/share/non-existent-token-12345")
    assert invalid_res.status_code == 404
