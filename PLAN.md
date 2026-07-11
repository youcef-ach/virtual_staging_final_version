# Virtual Staging Platform — Master Plan & Progress Tracker

> **How this file works:** every step has a checkbox. When a step is implemented on GitLab,
> pulled locally, run, tested and **validated by you**, it gets checked `[x]`.
> Sub-steps may be added as we go, but the structure and decisions below are locked.
>
> **Workflow:** simple, step by step. Duo implements & commits directly to `main` →
> you pull, run, test, validate → we check the box → next step. No MRs, no branches.

---

## 0. Locked Decisions (Reference — do not reopen)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Renderer | Babylon.js 9.0, WebGPU-native, WebGL2 fallback |
| 2 | v1 staging | Add-only (no material swaps, no room emptying) |
| 3 | Inpainting | Year 2 fast-follow |
| 4 | Segmentation | 2D-first (Mask2Former + SAM), fused with 3D, runs in v1 |
| 5 | BIM | Year 2 module; supporting invariants active from day 1 |
| 6 | Proxy mesh | Plane-intersection shell + OBJ-clipped non-planar details |
| 7 | Transitions | Projective morphing with adaptive stretching + velocity blur |
| 8 | Catalog | Curated in-house glTF, monetization-ready metadata schema |
| 9 | Dollhouse | Textured Matterpak OBJ |
| 10 | Pose QA | Automated edge-overlap score, human review on alert + spot checks |
| 11 | Physics | Havok WASM (Babylon plugin) |
| 12 | Timeline | 12 months, HSSE acceptance metrics |

---

## 1. Architecture Overview (Reference)

Three synchronized runtime layers:

1. **Photographic layer** — per-sweep 4K cubemap (6 KTX2 faces), rendered as skybox, never altered. *Visual truth.*
2. **Geometric layer** — invisible proxy mesh (depth-write occlusion, collision, shadow catching) + parametric planes (`structural_planes` table). *Metric truth from E57.*
3. **Dynamic asset layer** — Draco-compressed glTF 2.0 furniture, PBR, lit by per-sweep IBL.

**Coordinate Lock Rule (hard invariant):** the E57 origin `(0,0,0)` is preserved unchanged
through every stage. All poses, planes, and placements are expressed in this single frame.
One fixed conversion applied at ingestion, Matterport Z-up → runtime Y-up:

```
X_rt = X_mp,   Y_rt = Z_mp,   Z_rt = -Y_mp
```

**Split:** heavy compute offline (Python GPU workers, idempotent DAG); runtime is a
lightweight streaming web viewer.

---

## 2. Progress Tracker

### Phase M1–3 · Foundation  ⟵ **CURRENT PHASE**

> **Milestone:** pixel-accurate overlay on the two-story-home scan, then the 50-scan set.

- [x] **F1. Project scaffolding** ✅ *validated locally 2026-07-10*
  - [x] F1.1 Repo layout (`backend/`, `pipeline/`, `viewer/`, `dashboard/`, `docs/`)
  - [x] F1.2 Python project setup (Python 3.11, pyproject, deps: pye57, numpy, opencv, trimesh, pdal bindings)
  - [x] F1.3 Viewer project setup (TypeScript + Vite, Babylon.js 9.0, React 19 + Zustand shell)
  - [x] F1.4 Local dev environment docs (how to run everything on your machine)
- [x] **F2. E57 parser + pose extractor** *(first ticket)* ✅ *validated 2026-07-11 on a real 10-sweep E57 (tripod heights on Y ≈ 1.4 m, GUIDs extracted)*
  - [x] F2.1 Read `.e57`: enumerate sweeps, extract per-sweep 4×4 pose matrices (pye57)
  - [x] F2.2 Apply the coordinate conversion (Z-up → Y-up) exactly once, at ingestion
  - [x] F2.3 Emit `poses.json` (sweep UUID → 4×4 matrix, runtime frame)
  - [x] F2.4 Unit tests: round-trip conversion, origin preservation invariant
- [x] **F3. Ingestion & versioning API** ✅ *validated 2026-07-11: real 26 GB E57 uploaded → 201, SHA-256 registered, immutable key `raw/{property_id}/v1/`, authoritative v1, status `ingested`*
  - [x] F3.1 PostgreSQL 16 schema — initial migration (`organizations`, `users`, `properties`, `scan_versions`, `scan_positions`, `structural_planes`, `navigation_graph`, `catalog`, `staging_projects`, `qa_reviews`, `publish_states`)
  - [x] F3.2 FastAPI upload endpoint: receive `.e57` + Matterpak + cubemaps, validate completeness + checksums
  - [x] F3.3 Immutable raw storage layout `/raw/{property_id}/{version}/` (S3-compatible; local FS for dev)
  - [x] F3.4 `scan_versions` registration with `is_authoritative` flag; status → `INGESTED`
- [x] **F4. Cubemap viewer (Babylon.js)** ✅ *validated 2026-07-11 on the 198-sweep apartment scan*
  - [x] F4.0 Extract embedded per-sweep cubemap faces from E57 `images2D` (pye57 low-level API), keyed to sweep UUIDs, manifest.json *(added: source data ships panoramas inside the .e57)* ✅ *validated 2026-07-11: 198/198 sweeps with all 6 faces, Set A canonical transforms applied, 52 tests green*
  - [x] F4.1 WebGPUEngine async init, `powerPreference: high-performance`, WebGL2 fallback
  - [x] F4.2 Load active sweep cubemap as unlit skybox (Layer 1)
  - [x] F4.3 Sweep switching via `scan_positions` hotspot coordinates (simple jump via sweep panel + prev/next; morphing comes in M7–9)
  - [x] F4.4 Face-orientation calibration ✅ **locked: `mirrorX · rotX-90`** on the cube texture reflection matrix (runtime-only, Layer 1 pixels untouched); pano camera single-code-path orientation (no 180° jump)
- [ ] **F5. Automated pose-QA scoring (Gate 1)**
  - [ ] F5.1 Headless Blender (bpy): render Matterpak OBJ wireframe from each sweep pose
  - [ ] F5.2 Canny edge map of panorama vs projected mesh edges → edge-overlap score
  - [ ] F5.3 Threshold: score < 0.85 → flag sweep to operator dashboard
  - [ ] F5.4 Random 10% spot-check selection regardless of score
- [ ] **F6. Operator dashboard (pose QA tool)**
  - [ ] F6.1 React admin shell, property list + pipeline stage status
  - [ ] F6.2 Side-by-side overlay view (pano vs wireframe)
  - [ ] F6.3 Micro-adjust sliders (±1 mm / ±0.1°); corrected poses become ground truth → `poses.json`
- [ ] **F7. Milestone validation**
  - [ ] F7.1 Pixel-accurate overlay on the two-story-home scan
  - [ ] F7.2 Pixel-accurate on the 50-scan test set, zero visible drift

### Phase M4–6 · Geometry & Lighting

> **Milestone:** proxy occludes a test object correctly; window reflects on a chrome sphere; sofa looks grounded.

- [ ] **G1. Point cloud preprocessing** (Stage 3: PDAL tile-wise — never load 26 GB in memory; SOR filter 30 neighbors σ=2.0; voxel downsample 15 mm; normals 5 cm radius; archive full-res untouched; compressed PLY)
- [ ] **G2. Plane fitting & boundary extraction** (Stage 5: sequential RANSAC floor/ceiling/walls, alpha-shape hulls, Douglas-Peucker 1 mm, opening detection, Clipper subtraction, sealed watertight corners < 1 mm, `structural_planes` rows with real-scale UV frames 1 m = 1.0 UV)
- [ ] **G3. Proxy mesh synthesis** (Stage 6: watertight shell < 5,000 tris/room, labeled sub-meshes, OBJ-clipped non-planar details merged, `proxy.glb` Draco: pos 14 / normal 10 / UV 11 bits)
- [ ] **G4. G-buffer baking** (Stage 7: per-sweep pixel-aligned depth/world-pos/normal, 2048² KTX2 — validates alignment now, enables Year-2 re-texturing)
- [ ] **G5. IBL pre-computation** (Stage 8: linearize sRGB; interim blind boost `C_hdr = C_lin × (1 + 12·max(0, Y−0.85)^1.5)` → replaced by segmentation-guided highlight reconstruction before ship; diffuse irradiance + 5-level GGX specular chain via cmft; KTX2 UASTC HDR; emit `lighting_seeds.json`)
- [ ] **G6. Semantic segmentation pipeline + Gate 2 tooling** (Stage 4: Mask2Former ADE20K + SAM refinement + 2D-3D fusion; classes: floor, wall_NN, ceiling, window, door, furniture, other; click-fix mask editor; masks + 3D furniture boxes stored for Year-2 inpainting — zero schema rework)
- [ ] **G7. First furniture piece** (occlusion via depth-only proxy, per-sweep IBL, contact shadow via shadow-catcher WGSL: `final = background × (0.35 + 0.65 × shadowVisibility)`)
- [ ] **G8. Milestone validation** (occlusion correct; chrome-sphere window reflection; sofa grounded)

### Phase M7–9 · Tour & Staging

> **Milestone:** complete multi-room tour of a test property with a staged room, 60 FPS.

- [ ] **T1. Navigation graph** (Stage 9: nodes = sweeps; edges = unoccluded line-of-sight through proxy ≤ ~10 m; door-opening edges flagged as portals; `navigation_graph.json`)
- [ ] **T2. Projective morphing transitions** (project current pano onto Matterpak OBJ via ShaderMaterial texture projector; cubic Bézier A→B ~800 ms; adaptive stretching + velocity blur if destination KTX2 not ready; final 150 ms cross-fade; neighbor prefetch; VRAM discipline — only active sweep resident)
- [ ] **T3. Dollhouse view** (textured Matterpak OBJ, orbit camera, clickable sweep markers)
- [ ] **T4. Minimap** (Matterpak floorplan + position indicator + hotspots)
- [ ] **T5. Measurement tools** (point-to-point + area, computed against LiDAR-accurate planes/poses)
- [ ] **T6. Drag-drop staging editor** (raycast to floor sub-meshes, Y locked to floor; Havok WASM colliders vs proxy walls; wall-bound assets snap to wall normal)
- [ ] **T7. Staging project save/load** (JSON diff to `staging_projects`; shareable view-only links resolve `active_sweep_uuid` + payload)
- [ ] **T8. Milestone validation** (full tour + staged room at 60 FPS)

### Phase M10–12 · Platform & Scale

> **Milestone:** upload scan → auto-process → publish → stage → share, end-to-end under 15 minutes.

- [ ] **P1. Catalog admin + normalization pipeline** (headless Blender + gltf-transform: validate → meters → bottom-center pivot → decimate 40k/15k tris → Draco → KTX2 → AABB + thumbnail + LODs → S3 + catalog row; 100+ curated assets, all served from own CDN)
- [ ] **P2. Multi-tenancy, auth, share links** (JWT, org → user → role)
- [ ] **P3. DAG dashboard + autoscaling GPU workers** (Celery + Redis DAG state; Kubernetes, NVIDIA L4/A10G)
- [ ] **P4. Compression & publish stage** (Stage 10: KTX2/Basis UASTC everywhere, Draco meshes, S3 + CloudFront, multi-res progressive cubemaps; status → `PUBLISHED`)
- [ ] **P5. Monitoring** (Prometheus + Grafana, Loki logs)
- [ ] **P6. CDN + load testing, cross-device QA, docs**
- [ ] **P7. Milestone validation** (end-to-end < 15 min)

---

## 3. Offline Pipeline Stage Reference (per-property DAG)

Every stage: **idempotent, input-hash versioned, per-stage cached, resumable, re-runnable independently.**
Celery tasks on autoscaling GPU pods, status dashboard per property.

| Stage | Name | Key tech | Phase |
|-------|------|----------|-------|
| 1 | Ingest & Validate | pye57, PDAL, boto3, FastAPI | M1–3 (F3) |
| 2 | Pose Extraction & QA (Gate 1) | pye57, NumPy, OpenCV, trimesh, bpy | M1–3 (F2/F5/F6) |
| 3 | Point Cloud Preprocessing | PDAL tile-wise, Open3D | M4–6 (G1) |
| 4 | Semantic Segmentation (Gate 2) | PyTorch, Mask2Former, SAM | M4–6 (G6) |
| 5 | Plane Fitting & Boundaries | Open3D, Shapely, Clipper, CGAL | M4–6 (G2) |
| 6 | Proxy Mesh Synthesis | headless Blender, trimesh, Draco | M4–6 (G3) |
| 7 | G-Buffer Baking | headless Blender offscreen | M4–6 (G4) |
| 8 | IBL Pre-computation | OpenCV, OpenImageIO, cmft, toktx | M4–6 (G5) |
| 9 | Navigation Graph | trimesh ray casting, NetworkX | M7–9 (T1) |
| 10 | Compress & Publish | KTX2, Draco, S3 + CloudFront | M10–12 (P4) |

---

## 4. Data Model Reference (PostgreSQL 16 + JSONB)

- `organizations`, `users` — JWT auth, org → user → role
- `properties`
- `scan_versions` — immutable, `is_authoritative` flag, raw asset URLs + checksums
- `scan_positions` — sweep UUID, 4×4 pose as `REAL[16]`, pano/IBL/G-buffer KTX2 URLs, hotspot coordinates
- `structural_planes` — semantic class, plane coefficients `REAL[4]`, boundary polygon GeoJSON, `openings_json`
- `navigation_graph` — nodes/edges/portal flags
- `catalog` — SKU, manufacturer, product name, category, price, currency, purchase URL, materials JSONB, dimensions_mm JSONB, weight, GLB/thumbnail S3 URLs, bounding box, polygon count, LOD URLs
- `staging_projects` — property FK, name, `placement_payload` JSONB of `{sku, position [x,y,z], rotation quaternion}`, `is_published`, `active_sweep_uuid`
- `qa_reviews`, `publish_states`

Editing = lightweight JSON diff on `staging_projects`; heavy assets static and shared.
Raw exports immutable in `/raw/{property_id}/{version}/`.

---

## 5. Acceptance Criteria (enterprise-grade, measurable)

- [ ] Pose alignment: pixel-accurate on 50-scan test set, zero visible drift
- [ ] Geometry: plane corner tolerance < 1 mm; proxy < 5k tris/room, watertight
- [ ] Runtime: locked 60 FPS desktop + mobile web; < 3 s initial load
- [ ] Pipeline: < 15 min end-to-end per property; every stage independently re-runnable
- [ ] Visual QA: staged furniture passes side-by-side review; automated SSIM screenshot regression on reference scenes

---

## 6. Year 2 Modules (pre-wired now, zero rework later)

- **Material re-texturing (floors first):** projective WGSL shader — ray from pose to fitted plane → real-scale UV → sample PBR material → multiply by intrinsic shading from photo. Consumes existing planes, masks, G-buffers.
- **Room emptying:** stored 3D furniture boxes → SAM prompts → pixel-perfect masks → depth-guided diffusion inpainting (LaMa / SD + ControlNet), inpaint on plane textures, reproject for cross-sweep consistency, human QA.
- **BIM export:** `structural_planes` + openings → IfcOpenShell → IFC4 DTV (IfcWallStandardCase with measured thickness, IfcSlab, IfcOpeningElement + IfcWindow/IfcDoor). Industrial scaling on archived full-res E57 (IfcPipeSegment, IfcColumn, IfcBeam).
- **Free-roam:** LiDAR-seeded 3D Gaussian Splatting with geometric regularization, coordinate-locked.
- **HDR capture upgrade, cinematic video export (headless Chrome + FFmpeg), mobile/VR.**

---

## 7. Top Risks & Guardrails

1. **Pose registration drift** → automated scoring on every sweep, Gate 1, obsessive Phase-1 verification. *Everything downstream depends on it.*
2. **IBL believability from LDR** → segmentation-guided reconstruction, chrome-sphere visual test in M4–6 milestone.
3. **Proxy occlusion errors on non-planar geometry** → OBJ-detail merge (Decision 6) + visual QA checklist (clipping, occlusion correctness) before publish.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-10 | Plan committed. Phase M1–3 Foundation started. Next step: **F1 project scaffolding**, then **F2 E57 parser + pose extractor**. |
| 2026-07-10 | **F1 implemented** — repo layout (`backend/`, `pipeline/`, `viewer/`, `dashboard/` placeholder, `docs/`), FastAPI app + health endpoint + tests, `vs_pipeline` package skeleton, Vite/React 19/Zustand/Babylon.js viewer with WebGPU→WebGL2 fallback and FPS HUD, dev docs. **Awaiting local validation** before F1 boxes are checked. Next: **F2**. |
| 2026-07-10 | **F1 validated locally** (viewer dev server, backend pytest + uvicorn, pipeline pytest all green). F1 boxes checked. |
| 2026-07-10 | **F2 implemented** — `vs_pipeline/coordinates.py` (Coordinate Lock Rule: single change-of-basis, point + rigid-pose conversion, exact inverse), `vs_pipeline/stages/pose.py` (E57 pose extraction via pye57, stable sweep UUIDs with deterministic fallback, `poses.json` schema v1 with source SHA-256, CLI), 20+ unit tests (axis mapping, origin invariant, round-trip, rigidity, serialization). **Awaiting local validation.** Next: **F3 ingestion & versioning API**. |
| 2026-07-11 | **F2 validated** on a real 10-sweep Matterport E57 (`cloud_0.e57`): all 40 tests green, tripod heights land on Y (≈1.38–1.41 m), origin preserved, real scan GUIDs extracted. F2 boxes checked. |
| 2026-07-11 | **F3 implemented** — full PostgreSQL 16 schema (11 tables, SQLAlchemy 2.0 + Alembic migration 0001, docker-compose), immutable raw storage (`raw/{property_id}/v{version}/`, streaming SHA-256, write-once guard), FastAPI ingestion: orgs/properties CRUD + multipart scan-version upload with completeness report and authoritative-version promotion. Tests on in-memory SQLite. **Awaiting local validation.** Next: **F4 cubemap viewer**. |
| 2026-07-11 | **F3 validated** (real 26 GB E57 → 201, checksummed, immutable, authoritative, `ingested`). **F4.0 implemented + validated** — embedded cubemap extraction with the proven Set A canonical transforms as a shared lossless module (`cubemap_transforms.py`, parity-tested byte-identical against the original algorithm): 198/198 sweeps, all 6 faces, manifest with transform spec. |
| 2026-07-11 | **F4.1–F4.3 implemented** — tour viewer: manifest+poses join by sweep UUID, right-handed Y-up scene, unlit 6-face skybox (Layer 1), camera at pose translation with rotate-in-place controls, sweep panel + prev/next simple-jump switching, VRAM discipline on cubemap swaps, loading/error overlays. Placeholder scene removed. **Awaiting local validation, then F4.4 face-orientation calibration.** |
| 2026-07-11 | Visual checks passed (seams, up/down, mirroring, continuity → Set A transforms hold in Babylon). Fixed two viewer bugs: cube texture cache collision (all sweeps shared rootUrl "" → same GPU texture; now keyed by sweep UUID) and orbit-style controls replaced with a true inside-out pano camera (FreeCamera + drag handler, pitch clamp, yaw/pitch continuity). **Awaiting re-validation.** |
| 2026-07-11 | Sweep switching confirmed working. Remaining issue identified as the old pipeline's “global inversion handled in frontend”: cubemap content is scanner Z-up, viewer world is Y-up → yaw appears to spin around a sideways axis. Added **F4.4 calibration harness**: global reflection-matrix rotation on the cube texture (runtime-only, never baked), O key cycles 9 candidate orientations, M toggles mirror, state in HUD. Default candidate: rotX+90. **Awaiting calibration result to lock the matrix.** |
| 2026-07-11 | Fixed 180° camera jump on first drag (RH TargetCamera path discrepancy → single `applyLook()` orientation code path). Calibration result: **`rotX-90 + mirror [2]`** → locked as constant `SKYBOX_ORIENTATION = mirrorX · rotX-90`, harness removed. **F4 block complete and validated** — full 198-sweep tour browsable with correct orientation, seams, and continuity. Next: **F5 automated pose-QA scoring (Gate 1)** — requires the Matterpak OBJ. |
| 2026-07-11 | Matterpak confirmed complete for the two-story home (OBJ 43 MB + MTL + 222 textures + cloud.xyz 3.4 GB + color/ceiling plans → plans reserved for the M7–9 minimap). **F5 implemented** — `pose_qa.py` (prepare: per-face 90° cameras in the Matterport frame from poses.json via exact inverse conversion; score: Canny edge-overlap per face, 0.85 flag threshold, deterministic 10% spot-check, review composites) + `blender/render_edges.py` (headless Cycles 1-sample normalized-depth faces, `--limit` quick pass, `--obj-up`). Render orientation is **auto-calibrated** (8 ops × argmax on sample sweeps) instead of hand-tuned. **Awaiting local validation.** |
| 2026-07-11 | First F5 run: incoherent per-face calibration (means 0.5–0.7, all 10 sweeps flagged) → diagnosed as **face-mapping mismatch** (extracted faces carry the Set A Z-swap + viewer mirror, so extracted N ≠ rendered N). Calibration widened to the **full bijective face-mapping search** (6 rendered faces × 8 ops per extracted face, greedy assignment, cached edge maps). Blender-5 API + absolute-path fixes for `render_edges.py` pending (user's local working version to be committed). **Awaiting re-score.** |
| 2026-07-11 | MatterPak readme confirms: **OBJ is right-handed Z-up, 1 unit = 1 m — identical frame to the E57**. The Blender default Y-up OBJ import was therefore rotating the mesh 90°; `--obj-up` default flipped to **Z**. User's Blender-5 compositor patch (dual API 4.x/5.x) + absolute output path fix committed. Color plans are per-floor (reserved for M7–9 minimap); cloud.xyz is decimated → E57 remains the metric source. **Re-render 10 sweeps + re-score pending.** |
| 2026-07-11 | Re-score with Z-up import + full mapping: **coherent global mapping found** (±X swapped, ±Y swapped, Z fixed + flips — the 2D signature of the viewer's mirror), face means 0.71–0.97, sweep mean 0.819, 4/10 flagged for review (correct Gate-1 behavior). Added **persistent calibration** (`calibration.json`: computed once, locked, reloaded on re-runs) and per-face detail in flag output. **Composite inspection of flagged sweeps pending** (pose drift vs content artifacts: floors/mirrors/window mesh walls). |
| 2026-07-11 | Composite inspection: blank composites on worst-flagged sweeps traced to **empty depth renders** (cameras facing sky/outside the mesh — expected on a facade-inclusive scan) scoring 0.0. Fix: faces under `MIN_RENDER_EDGE_PIXELS` are excluded as *uninformative* (calibration skips them too, ≥ 2 informative faces required per sweep); flagged sweeps now produce a **labeled 6-face grid composite** for review. **Awaiting re-score + full 198-sweep run.** |
| 2026-07-11 | Re-score after no-signal exclusion: calibration means 0.87–0.97 (locked), sweep mean 0.871, **2/10 flagged** (009 = 0.500 with an informative py face at 0.00 — under composite review; 007 = 0.820 borderline). **F6 first slice implemented** — operator dashboard (React admin on :5174): QA summary, filterable worst-first sweep queue, face-score table with no-signal faces, grid-composite review pane. Micro-adjust (F6.3) pending confirmation that flags are genuine pose drift. **Full 198-sweep render+score in progress.** |
| 2026-07-11 | Added **alignment verification tooling** (user requested proof beyond the score): `render_edges.py --mode textured` (unlit Workbench textured renders — MatterPak textures derive from the panos, so correct pose ≈ photo), `pose_qa verify` (per-face pano / 50-50 blend / diff-heatmap grids + NCC — pose error appears as ghosting), and `prepare --perturb-*` (deliberate pose corruption proving the metric detects errors). |
