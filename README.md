# Virtual Staging Platform

Photorealistic, add-only virtual staging on Matterport/E57 LiDAR scans.
Three synchronized runtime layers (photographic cubemaps / metric proxy geometry /
dynamic PBR furniture) rendered with Babylon.js (WebGPU-native, WebGL2 fallback).

**The single source of truth for scope, decisions and progress is [`PLAN.md`](PLAN.md).**

## Repository layout

| Path | Purpose |
|------|---------|
| `backend/` | FastAPI REST API: ingestion, versioning, catalog, staging, auth |
| `pipeline/` | Offline processing pipeline (per-property DAG of idempotent stages) |
| `viewer/` | Runtime web viewer (TypeScript, Vite, React 19, Zustand, Babylon.js) |
| `dashboard/` | Operator dashboard (pose QA, mask fixing, publish workflow) — scaffolded in F6 |
| `docs/` | Developer documentation |

## Quick start

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full local setup.

```bash
# Viewer
cd viewer && npm install && npm run dev

# Backend API
cd backend && pip install -e ".[dev]" && uvicorn app.main:app --reload

# Pipeline tests
cd pipeline && pip install -e ".[dev]" && pytest
```

## Hard invariant — Coordinate Lock Rule

The E57 origin `(0,0,0)` is preserved unchanged through every stage. One fixed
conversion is applied exactly once at ingestion (Matterport Z-up → runtime Y-up):

```
X_rt = X_mp,   Y_rt = Z_mp,   Z_rt = -Y_mp
```
