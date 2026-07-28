import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
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


def test_catalog_crud(client: TestClient) -> None:
    # 1. Create manual catalog item
    payload = {
        "sku": "SOFA-TEST-100",
        "manufacturer": "Nordic Living",
        "product_name": "Test Modern Sofa",
        "category": "sofas",
        "price": 1299.99,
        "currency": "USD",
        "dimensions_mm": {"x": 2100, "y": 850, "z": 900},
    }
    res = client.post("/api/catalog", json=payload)
    assert res.status_code == 201, res.text
    data = res.json()
    item_id = data["id"]
    assert data["sku"] == "SOFA-TEST-100"
    assert data["product_name"] == "Test Modern Sofa"

    # 2. Duplicate SKU fails
    res_dup = client.post("/api/catalog", json=payload)
    assert res_dup.status_code == 409

    # 3. List catalog items
    res_list = client.get("/api/catalog?category=sofas")
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(i["id"] == item_id for i in items)

    # 4. Filter search
    res_search = client.get("/api/catalog?search=Modern")
    assert res_search.status_code == 200
    assert len(res_search.json()) >= 1

    # 5. Get single item
    res_get = client.get(f"/api/catalog/{item_id}")
    assert res_get.status_code == 200
    assert res_get.json()["sku"] == "SOFA-TEST-100"

    # 6. Patch update
    res_patch = client.patch(f"/api/catalog/{item_id}", json={"price": 1199.99})
    assert res_patch.status_code == 200
    assert float(res_patch.json()["price"]) == 1199.99

    # 7. Delete item
    res_del = client.delete(f"/api/catalog/{item_id}")
    assert res_del.status_code == 204

    # 8. Verify deleted
    res_get_deleted = client.get(f"/api/catalog/{item_id}")
    assert res_get_deleted.status_code == 404
