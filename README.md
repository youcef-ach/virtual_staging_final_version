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

## Presentation demo

From `viewer/`, run `npm ci` and `npm run dev -- --host 127.0.0.1`, then open the printed address with `?demo=1`. Explore the synthetic living room, switch viewpoints, compare the panorama with and without the staged chair, or open the 3D cutaway. **Inspect model outputs** compares the same image with real precomputed Mask2Former and SAM results and exposes the run record.

`npm run build:showcase` writes a standalone demo to `viewer/dist-showcase/`. It verifies the generated fixture inventory and packages only the named synthetic assets. The demo uses studio lighting, runs without a backend, and keeps furniture changes in the current session. The default local viewer still loads your data from `viewer/public/data/`.

## Hard invariant — Coordinate Lock Rule

The E57 origin `(0,0,0)` is preserved unchanged through every stage. One fixed
conversion is applied exactly once at ingestion (Matterport Z-up → runtime Y-up):

```
X_rt = X_mp,   Y_rt = Z_mp,   Z_rt = -Y_mp
```
