# Local Development Environment

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11.x | backend + pipeline |
| Node.js | ≥ 20 LTS | viewer (and dashboard later) |
| npm | ≥ 10 | ships with Node 20 |
| Git | any recent | |

Docker is required for the backend database (PostgreSQL 16 via `docker compose`).
Optional until Phase M4–6: PDAL (system package), Blender ≥ 4.x (headless), Redis.

## Workflow

1. Changes are committed directly to `main` (no MRs, no feature branches).
2. You `git pull`, run, test and validate locally.
3. Once validated, the corresponding checkbox in `PLAN.md` is checked and the changelog updated.

## Viewer (`viewer/`)

```bash
cd viewer
npm install
npm run dev        # http://localhost:5173
npm run typecheck  # strict TS check
npm run build      # production build
```

The HUD (top-left) shows the active render backend — **WEBGPU** on supported browsers
(Chrome/Edge ≥ 113, recent Firefox), **WEBGL2** otherwise. Verify both paths when possible.

## Backend (`backend/`)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload    # http://127.0.0.1:8000, docs at /docs
pytest                           # run tests
ruff check . && mypy app         # lint + typecheck
```

Settings are environment-driven with the `VS_` prefix (see `app/config.py`), e.g. `VS_ENVIRONMENT=production`.

### Database (PostgreSQL 16)

```bash
docker compose up -d db      # from repo root; starts postgres:16 on :5432
cd backend
alembic upgrade head         # apply migrations
```

Default connection: `postgresql+psycopg://vs:vs@localhost:5432/virtual_staging`
(override with `VS_DATABASE_URL`). Unit tests run on in-memory SQLite and need no database.

### Raw asset storage

Dev storage is the local filesystem under `VS_STORAGE_ROOT` (default `backend/data/storage`),
using the same immutable key layout as the production S3 bucket:
`raw/{property_id}/v{version}/{filename}`. Existing keys are never overwritten.

### Trying the ingestion API

Open Swagger UI at http://127.0.0.1:8000/docs and, in order:

1. `POST /organizations` — create your org
2. `POST /properties` — create a property under it
3. `POST /properties/{property_id}/scan-versions` — attach your `.e57`
   (+ optional Matterpak / cubemap files) and execute

The response contains version number, SHA-256 checksums, storage keys, status
`ingested` and a completeness report.

## Pipeline (`pipeline/`)

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Note: the `pointcloud` extra (`pip install -e ".[pointcloud]"`) requires PDAL installed
at the system level. Not needed until Phase M4–6 (stage G1).

## Conventions

- Python: `ruff` (line length 100), `mypy` strict-ish, `pytest` for everything testable.
- TypeScript: `strict: true`, no `any`, tree-shaken `@babylonjs/core/*` scoped imports only.
- All world-space math in the runtime Y-up frame; conversion from Matterport Z-up happens exactly once, at ingestion (Coordinate Lock Rule).
- No heavy binary assets in git — scan data and processed assets live in object storage.
