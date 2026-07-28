# Virtual Staging Platform API Reference

The REST API is built with FastAPI and runs on `http://localhost:8000`. 
All JSON responses use `snake_case` keys.

## Authentication

Most endpoints require a JWT bearer token. Get a token via `/api/auth/token`.

```bash
curl -X POST "http://localhost:8000/api/auth/token" \
     -d "username=admin@test.com&password=secret"
```

## Endpoints

### Health & Monitoring

- **`GET /health`**
  Check API status, DB connection, and storage access.
- **`GET /metrics`**
  Prometheus metrics exposition (includes pipeline timings and counts).

### Pipeline Orchestration

- **`POST /api/pipeline/run`**
  Start a new pipeline run for a scan version.
  *Body:* `{"scan_version_id": "uuid"}`
- **`GET /api/pipeline/runs`**
  List all pipeline runs.
- **`GET /api/pipeline/runs/active`**
  List runs currently in PENDING or RUNNING state.
- **`GET /api/pipeline/runs/{run_id}`**
  Get detailed status of a run, including all stage executions.
- **`GET /api/pipeline/runs/{run_id}/events`**
  **Server-Sent Events (SSE)** endpoint streaming real-time stage status updates.
- **`POST /api/pipeline/runs/{run_id}/stages/{stage_number}/retry`**
  Retry a failed stage (resets downstream dependents).
- **`POST /api/pipeline/runs/{run_id}/cancel`**
  Cancel a pending or running pipeline.

### Catalog

- **`GET /api/catalog`**
  List available furniture assets (supports `?category=` and `?search=` filters).
- **`POST /api/catalog/upload`**
  Upload and normalize a new GLB asset (requires multipart/form-data).

### Properties & Scans

- **`GET /properties`**
  List all properties.
- **`GET /properties/{id}/scans`**
  List scan versions for a property.
