import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  (register tables on the metadata)
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
    get_settings.cache_clear()


def _create_property(client: TestClient) -> dict:
    org = client.post("/api/organizations", json={"name": "Acme Realty"})
    assert org.status_code == 201
    prop = client.post(
        "/api/properties", json={"organization_id": org.json()["id"], "name": "Test Home"}
    )
    assert prop.status_code == 201
    return prop.json()


def test_upload_registers_ingested_version(client, tmp_path):
    prop = _create_property(client)
    payload = b"fake e57 content for checksum test"

    response = client.post(
        f"/api/properties/{prop['id']}/scan-versions",
        files=[("e57_files", ("cloud_0.e57", payload, "application/octet-stream"))],
    )
    assert response.status_code == 201
    body = response.json()
    assert body["version"] == 1
    assert body["status"] == "ingested"
    assert body["is_authoritative"] is True

    asset = body["raw_assets"]["e57"][0]
    assert asset["sha256"] == hashlib.sha256(payload).hexdigest()
    assert asset["size_bytes"] == len(payload)
    assert asset["key"] == f"raw/{prop['id']}/v1/cloud_0.e57"

    stored = tmp_path / "storage" / asset["key"]
    assert stored.read_bytes() == payload  # immutable raw copy on disk

    assert body["completeness"] == {
        "has_e57": True,
        "has_matterpak": False,
        "has_cubemaps": False,
        "complete": False,
    }


def test_new_version_demotes_previous_authoritative(client):
    prop = _create_property(client)
    for _ in range(2):
        response = client.post(
            f"/api/properties/{prop['id']}/scan-versions",
            files=[("e57_files", ("cloud_0.e57", b"scan bytes", "application/octet-stream"))],
        )
        assert response.status_code == 201

    listing = client.get(f"/api/properties/{prop['id']}/scan-versions").json()
    assert [v["version"] for v in listing] == [1, 2]
    assert [v["is_authoritative"] for v in listing] == [False, True]


def test_rejects_wrong_extension(client):
    prop = _create_property(client)
    response = client.post(
        f"/api/properties/{prop['id']}/scan-versions",
        files=[("e57_files", ("notes.txt", b"nope", "text/plain"))],
    )
    assert response.status_code == 422


def test_unknown_property_returns_404(client):
    response = client.post(
        "/api/properties/00000000-0000-0000-0000-000000000000/scan-versions",
        files=[("e57_files", ("cloud_0.e57", b"x", "application/octet-stream"))],
    )
    assert response.status_code == 404
