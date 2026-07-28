"""Unit tests for P3: Pipeline orchestration (DAG, executor, orchestrator, API).

Tests cover:
  - DAG dependency resolution and scheduling
  - SimulatedExecutor execution
  - Orchestrator lifecycle (create run, stage transitions)
  - API endpoints (start, list, detail, retry, cancel)
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.main import create_app
from app.models import (
    Organization,
    PipelineRun,
    PipelineStatus,
    Property,
    ScanStatus,
    ScanVersion,
    StageExecution,
    StageStatus,
    User,
    UserRole,
)
from app.pipeline.dag import (
    STAGES,
    get_downstream,
    get_ready_stages,
    get_stage,
    is_dag_complete,
)
from app.pipeline.executor import ExecutionResult, SimulatedExecutor
from app.pipeline.orchestrator import (
    create_run,
    get_run,
    get_run_stages,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db_engine():
    """In-memory SQLite engine."""
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Scoped session for each test."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def db_factory(db_engine):
    """Callable that returns a new session (for orchestrator background tasks)."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    return factory


@pytest.fixture
def seed_data(db_session: Session):
    """Create an org + property + scan version for pipeline tests."""
    org = Organization(name="Test Org")
    db_session.add(org)
    db_session.flush()

    user = User(
        organization_id=org.id,
        email="test@test.com",
        password_hash="dummy",
        role=UserRole.ADMIN,
    )
    db_session.add(user)

    prop = Property(organization_id=org.id, name="Test Property")
    db_session.add(prop)
    db_session.flush()

    sv = ScanVersion(
        property_id=prop.id,
        version=1,
        status=ScanStatus.INGESTED,
        is_authoritative=True,
    )
    db_session.add(sv)
    db_session.commit()
    return {"org": org, "property": prop, "scan_version": sv}


# ── DAG Tests ─────────────────────────────────────────────────────────


class TestDAG:
    def test_stage_count(self):
        assert len(STAGES) == 10

    def test_get_stage(self):
        s = get_stage(1)
        assert s.name == "Ingest & Validate"
        assert s.depends_on == ()

    def test_get_stage_invalid(self):
        with pytest.raises(ValueError):
            get_stage(99)

    def test_ready_stages_initial(self):
        """Initially only Stage 1 (no deps) should be ready."""
        ready = get_ready_stages(completed=set(), running=set(), failed=set())
        assert len(ready) == 1
        assert ready[0].number == 1

    def test_ready_stages_after_stage1(self):
        """After Stage 1 completes, Stages 2, 3, 8 become ready."""
        ready = get_ready_stages(completed={1}, running=set(), failed=set())
        ready_nums = {s.number for s in ready}
        assert ready_nums == {2, 3, 8}

    def test_ready_stages_with_running(self):
        """Running stages are not returned as ready."""
        ready = get_ready_stages(completed={1}, running={2}, failed=set())
        ready_nums = {s.number for s in ready}
        assert 2 not in ready_nums
        assert {3, 8} <= ready_nums

    def test_ready_stages_blocked_by_failure(self):
        """Failed Stage 3 blocks Stages 4, 5 and their dependents."""
        ready = get_ready_stages(completed={1}, running=set(), failed={3})
        ready_nums = {s.number for s in ready}
        assert 4 not in ready_nums
        assert 5 not in ready_nums
        assert {2, 8} <= ready_nums

    def test_dag_complete(self):
        assert is_dag_complete(set(range(1, 11)))
        assert not is_dag_complete({1, 2, 3})

    def test_downstream(self):
        """Stage 3 downstream should include 4, 5, 6, 7, 9, 10."""
        down = get_downstream(3)
        assert 4 in down
        assert 5 in down
        assert 6 in down
        assert 10 in down

    def test_downstream_of_leaf(self):
        """Stage 10 has no downstream."""
        assert get_downstream(10) == []


# ── Executor Tests ────────────────────────────────────────────────────


class TestSimulatedExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        executor = SimulatedExecutor(speed_factor=0.01)
        stage = get_stage(1)
        result = await executor.execute(stage, "test-run-id", "/tmp/test")
        assert result.success is True
        assert result.duration_s > 0

    @pytest.mark.asyncio
    async def test_execute_with_failure(self):
        executor = SimulatedExecutor(speed_factor=0.01, fail_probability=1.0)
        stage = get_stage(1)
        result = await executor.execute(stage, "test-run-id", "/tmp/test")
        assert result.success is False
        assert result.error_message is not None


# ── Orchestrator Tests ────────────────────────────────────────────────


class TestOrchestrator:
    def test_create_run(self, db_session, seed_data):
        sv = seed_data["scan_version"]
        run = create_run(db_session, sv.id)
        assert run.status == PipelineStatus.PENDING
        assert run.scan_version_id == sv.id

        stages = get_run_stages(db_session, run.id)
        assert len(stages) == 10
        assert all(s.status == StageStatus.PENDING for s in stages)
        assert [s.stage_number for s in stages] == list(range(1, 11))

    def test_get_run(self, db_session, seed_data):
        sv = seed_data["scan_version"]
        run = create_run(db_session, sv.id)
        fetched = get_run(db_session, run.id)
        assert fetched is not None
        assert fetched.id == run.id

    def test_get_run_not_found(self, db_session):
        assert get_run(db_session, uuid.uuid4()) is None


# ── API Tests ─────────────────────────────────────────────────────────


@pytest.fixture
def api_client(db_engine):
    """TestClient with database tables created, fully isolated from PostgreSQL."""
    import os
    from importlib import reload
    import app.db
    import app.config

    # Force SQLite for tests
    old_url = os.environ.get("VS_DATABASE_URL")
    os.environ["VS_DATABASE_URL"] = "sqlite:///:memory:"

    # Reset cached engine/session factory
    app.db._engine = db_engine
    app.db._session_factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    app.config.get_settings.cache_clear()

    Base.metadata.create_all(db_engine)
    app_instance = create_app()
    client = TestClient(app_instance)

    # Seed data
    session = app.db._session_factory()
    org = Organization(name="API Test Org")
    session.add(org)
    session.flush()
    prop = Property(organization_id=org.id, name="API Test Property")
    session.add(prop)
    session.flush()
    sv = ScanVersion(
        property_id=prop.id,
        version=1,
        status=ScanStatus.INGESTED,
        is_authoritative=True,
    )
    session.add(sv)
    session.commit()

    yield client, str(sv.id)

    session.close()
    # Restore env
    if old_url is None:
        os.environ.pop("VS_DATABASE_URL", None)
    else:
        os.environ["VS_DATABASE_URL"] = old_url
    app.config.get_settings.cache_clear()
    app.db._engine = None
    app.db._session_factory = None


class TestPipelineAPI:
    def test_list_runs_empty(self, api_client):
        client, _ = api_client
        resp = client.get("/api/pipeline/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_start_run(self, api_client):
        client, sv_id = api_client
        resp = client.post(
            "/api/pipeline/run",
            json={"scan_version_id": sv_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] in ("pending", "running")
        assert data["scan_version_id"] == sv_id

    def test_start_run_invalid_scan(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/api/pipeline/run",
            json={"scan_version_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

    def test_get_run_detail(self, api_client):
        client, sv_id = api_client
        # Start a run first
        start_resp = client.post(
            "/api/pipeline/run",
            json={"scan_version_id": sv_id},
        )
        run_id = start_resp.json()["id"]

        # Give the async loop a moment to start
        import time
        time.sleep(0.1)

        resp = client.get(f"/api/pipeline/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stages"]) == 10
        assert data["stages"][0]["stage_number"] == 1

    def test_list_runs_after_start(self, api_client):
        client, sv_id = api_client
        client.post("/api/pipeline/run", json={"scan_version_id": sv_id})
        resp = client.get("/api/pipeline/runs")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_cancel_run(self, api_client):
        client, sv_id = api_client
        start_resp = client.post(
            "/api/pipeline/run",
            json={"scan_version_id": sv_id},
        )
        run_id = start_resp.json()["id"]

        resp = client.post(f"/api/pipeline/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_run(self, api_client):
        client, _ = api_client
        resp = client.post(f"/api/pipeline/runs/{uuid.uuid4()}/cancel")
        assert resp.status_code == 404
