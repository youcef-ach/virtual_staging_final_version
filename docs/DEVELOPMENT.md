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

### 6. Model Output Inspection

Test scripts often generate debug renders or segmentation overlays into directories like `tests/debug_out`. These are invaluable for diagnosing algorithm failures without running the full viewer.


## 7. Deployment & Scaling

The pipeline orchestrator uses a local subprocess executor for development. For production deployment, you'll need:

### Infrastructure
- **API/Dashboard:** Docker container running Uvicorn behind Nginx/ALB.
- **Database:** PostgreSQL 16+.
- **Pipeline Workers:** Celery workers backed by Redis, running on GPU-enabled instances (e.g. AWS g5.xlarge or Kubernetes pods with NVIDIA L4/A10G).
- **Storage:** S3 bucket for published artifacts (GLB, KTX2, JSON).

### CDN Setup
Set `VS_S3_BUCKET`, `VS_CDN_URL`, and `VS_AWS_PROFILE` in your `.env`. 
The `compress.py` stage (Stage 10) will automatically upload the proxy meshes and manifests to S3. Configure CloudFront (or equivalent CDN) to serve this bucket with aggressive caching (`Cache-Control: max-age=31536000`) for the GLB files.

### Monitoring
Start the monitoring stack using `docker-compose -f docker-compose.monitoring.yml up -d`.
- **Prometheus:** `localhost:9090`
- **Grafana:** `localhost:3000` (admin/admin). Import the pipeline metrics dashboard to track DAG completion times and failure rates.
- **Locust (Load Testing):** Run `cd backend && locust -f tests/load_test.py` to simulate dashboard polling.

## 8. Cross-Device QA Checklist

Before releasing major viewer updates, verify on these targets:
- [ ] **Desktop Chrome (Windows/macOS):** Baseline. WebGPU should be active.
- [ ] **Desktop Safari (macOS):** WebGL2 fallback. Check for texture caching bugs.
- [ ] **iOS Safari (iPhone 13+):** Strict VRAM limits. Verify that rapid sweep jumps don't crash the tab.
- [ ] **Android Chrome (Pixel 6+):** Verify pointer events (touch + drag) for the staging editor.

### Tour data

The viewer reads pipeline outputs from `viewer/public/data/` (git-ignored):

```powershell
# from the repo root, after running the pose + panorama stages in pipeline/
New-Item -ItemType Directory -Force viewer\public\data | Out-Null
Copy-Item pipeline\extracted_cubemaps\* viewer\public\data\ -Recurse -Force
Copy-Item pipeline\poses.json viewer\public\data\ -Force
```

Expected layout: `viewer/public/data/manifest.json`, `viewer/public/data/poses.json`,
and one `{sweep_uuid}/` folder per sweep containing `px/nx/py/ny/pz/nz.jpg`.

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
