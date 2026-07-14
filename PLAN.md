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
- [x] **F5. Automated pose-QA scoring (Gate 1)** ✅ *validated 2026-07-11: perturbation self-test (5 cm + 2°) flags via `low_margin`; correct poses pass; 47/198 review queue with flag reasons*
  - [x] F5.1 Headless Blender (bpy): render Matterpak OBJ wireframe from each sweep pose
  - [x] F5.2 Canny edge map of panorama vs projected mesh edges → chance-corrected edge-overlap score (raw overlap normalized by pano edge-mask density)
  - [x] F5.3 Two-signal flag rule *(amended from “score < 0.85” after full-run analysis)*: normalized score < 0.20 (gross misregistration) OR mean orientation margin < 0.03 (subtle drift) → flag sweep to operator dashboard
  - [x] F5.4 Random 10% spot-check selection regardless of score
- [ ] **F6. Operator dashboard (pose QA tool)**
  - [ ] F6.1 React admin shell, property list + pipeline stage status *(QA review shell live on :5174; property list + stage status need backend wiring)*
  - [x] F6.2 Side-by-side overlay view ✅ *validated 2026-07-11: 3-column composites (pano | OBJ render | red-edge overlay), schema-v4 queue with margins + flag reasons; full 47-flag review → zero pose errors, all flags content-driven (glass/windows, mirrors, OBJ sparsity)*
  - [ ] F6.3 Micro-adjust sliders (±1 mm / ±0.1°); corrected poses become ground truth → `poses.json` *(deferred by design: build only if a genuinely drifted pose ever appears — zero evidence to date)*
- [ ] **F7. Milestone validation**
  - [x] F7.1 Pixel-accurate overlay on the two-story-home scan ✅ *validated 2026-07-11: ghosting verification blends cleanly on structure (walls/room edges); residual detail offsets (furniture) = MatterPak OBJ decimation, not pose — zero pose errors across the full 198-sweep review; OBJ precision not load-bearing downstream (E57 drives planes/proxy/G-buffers/measurements)*
  - [ ] F7.2 Pixel-accurate on the 50-scan test set, zero visible drift

### Phase M4–6 · Geometry & Lighting

> **Milestone:** proxy occludes a test object correctly; window reflects on a chrome sphere; sofa looks grounded.

- [x] **G1. Point cloud preprocessing** ✅ *validated 2026-07-12: full 198-sweep run → 53,963,860-point working cloud (15 mm, runtime Y-up), SOR 2–4%/sweep, sensor-oriented normals, per-sweep cache retained; visual check passed (both floors + facade)* (Stage 3: per-sweep chunked via pye57 — the E57's native tiles — never load 26 GB in memory; SOR filter 30 neighbors σ=2.0; voxel downsample 15 mm; normals 5 cm radius oriented to sensor; archive full-res untouched; compressed PLY; PDAL deferred until non-E57 sources)
- [x] **G2. Plane fitting & boundary extraction** ✅ *validated 2026-07-12: full 198-sweep run — 824 planes, 126 wall-thickness pairs (9–290 mm), storey levels v3 (elevation bins + tripod evidence), all storeys correct incl. restored second-storey floor, walls top-down reads as the full floor plan; opening detection / Clipper subtraction / exact corner sealing deferred to G3 by design* (Stage 5: sequential RANSAC floor/ceiling/walls, occupancy-grid + Douglas-Peucker boundaries, `structural_planes` rows; real-scale UV frames 1 m = 1.0 UV)
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
| 2026-07-11 | **Ghosting verification PASSED**: textured renders confirmed textured, blend shows edges once (no double vision), heatmap correct → **pose alignment confirmed pixel-accurate**. Low NCC attributed to tone-mapping/windows (metric noise), blend softness to the MatterPak OBJ's documented low density — tolerated by design (E57 = metric truth, panoramas = visual truth; OBJ only serves QA rendering, detail clipping, dollhouse, morphing target). Textured mode now renders with Standard view transform. **Remaining for F5 closure: full 198-sweep depth render + score.** |
| 2026-07-11 | **F5 full-run diagnosis + metric fix.** Full 198-sweep run flagged 73/198 → root causes found. (1) **Stale-batch calibration**: nz orientation differed between the 10 quick-pass renders and the 188 full-run renders; calibration had been locked on the old batch only. Old renders deleted, 10 sweeps re-rendered, recalibration locked nz → `rot180_flip`; alignment visually confirmed on composites (residual softness = MatterPak OBJ decimation, tolerated by design). (2) **Chance-level saturation**: perturbation self-test (5 cm + 2° on sweep 001) moved the raw score only 0.970 → 0.933 — dilated pano edge masks on texture-dense faces reach ~0.9 density, so raw overlap tracks content, not pose. Face score replaced with chance-corrected normalized overlap `(raw − density)/(1 − density)` + pano saturation guard (density > 0.97 → uninformative); raw scores kept in `pose_qa.json` (schema v3); calibration unaffected (monotonic). Normalized separation confirmed: correct pose +0.826 vs 5 cm/2° perturbed +0.618. **Next: full re-score, threshold recalibration from the normalized distribution, then F5 closure.** |
| 2026-07-11 | **F5 two-signal flagging (Gate 1 final form).** Full-run normalized distribution (p10 0.20, p50 0.45, p90 0.75) proved no absolute threshold catches subtle drift: 5 cm + 2° on a texture-dense sweep scores 0.62 — above p75 of correct sweeps. Added the content-invariant **orientation margin** (per face: locked orientation's normalized overlap minus the best wrong orientation's; measured on sweep 001: +0.14 correct vs −0.02 perturbed — sign flips under drift). Flag rule: score < 0.20 (gross misregistration) OR mean margin < 0.03 (subtle drift). Flag reasons, margin, and per-face margins in `pose_qa.json` (schema v4). **Validation pending: (a) perturbed sweep must flag via `low_margin`; (b) full-run false-flag review.** |
| 2026-07-11 | **F5 VALIDATED — Gate 1 closed.** Perturbed sweep [001] (5 cm + 2°): score 0.618, margin −0.021 → flagged `low_margin` ✓. Correct-pose full run: mean score 0.448, mean margin 0.129, 47/198 flagged (27 `low_margin` = ambiguous/symmetric content → human review by design; 16 both = worst outliers; 4 `low_score`). F5.3 wording amended to the validated two-signal rule. **Documented follow-ups (non-blocking):** (1) extend perturbation coverage (2–3 sweeps incl. texture-sparse) during F7.2; (2) tune `MARGIN_THRESHOLD` with 50-scan data if review rate too high; (3) add render-batch homogeneity guard (stamp renderer version per render dir, refuse mixed-batch scoring — root cause of today's stale-calibration bug); (4) sub-cm drift below detection floor (OBJ decimation, ~5 px @ 512) — acceptable, F7 visual validation + F6.3 cover it. Next: **F6 dashboard update for schema v4** (margin, flag reasons), then F7. |
| 2026-07-11 | **F6 dashboard updated for QA schema v4** — queue rows show normalized score + orientation margin with per-reason badges (S/M); filters for `low_score` / `low_margin` / spot-check; margin filter sorts worst-margin-first; detail pane adds score/margin summary with flag reasons and a per-face table (score, margin, raw rows; thresholds read from the report). Repo hygiene: regenerable pipeline artifacts gitignored, depth-map reference doc moved to `docs/`. **Awaiting local validation: review the 47-sweep queue in the dashboard (validates F6.1/F6.2).** |
| 2026-07-11 | **F6.2 VALIDATED.** Composites upgraded to 3 columns (pano \| OBJ render \| red-edge overlay) so mesh sparsity is visually distinguishable from pose error. Full 47-flag queue reviewed in the dashboard: **zero pose errors** — every flag explained by content (glass/windows, mirrors, MatterPak OBJ sparsity). F6.3 remains deferred (no drift evidence). Noted for M4–6: G6 segmentation masks (window/mirror) can be fed back into Gate 1 to exclude glass regions and shrink the review queue. Next: **F7.1 sign-off**, then **F7.2 (50-scan set)**. |
| 2026-07-11 | **F7.1 VALIDATED** (operator sign-off: ghosting blends cleanly on structure; residual detail offsets = MatterPak OBJ decimation, zero pose errors across the 198-sweep review). F7.2 stays open pending multi-property data — acceptance milestone, not a development blocker. **Phase M4–6 started: G1 implemented** — `vs_pipeline/stages/pointcloud.py`: per-sweep chunked preprocessing (the E57's native tiles are its sweeps; pye57 per-scan reads keep memory bounded on the 26 GB source and give sensor origins for normal orientation — chosen over PDAL tiling for E57 sources, PDAL deferred until non-E57 inputs); SOR 30/σ 2.0 → 15 mm voxel → 5 cm hybrid-radius normals oriented to sensor → per-sweep PLY cache (idempotent, resumable, `--limit` quick pass) → progressive fold merge (bounded memory) → compressed binary PLY working cloud + `preprocess_meta.json`, all in the runtime Y-up frame via the shared Coordinate Lock module. 8 unit tests (frame conversion pure-numpy; Open3D tests skip if extra not installed). Open3D under the `pointcloud` extra. **Awaiting local validation.** |
| 2026-07-12 | **G1 VALIDATED** — full 198-sweep run on a Python 3.11 venv (Open3D has no 3.13 wheels): SOR 2–4%/sweep, ~1.7–1.8M pts/sweep at 15 mm, merged working cloud **53,963,860 points**, visual check passed (both floors, facade, Y-up, clean walls). **G2 design brief locked** after multi-reviewer analysis: (1) **local-to-global** plane extraction — per-sweep plane proposals from the cached sweep clouds (sensor-oriented normals) → cluster plane parameters + spatial bounds into global authoritative planes (prevents distant-coplanar “plane bleeding” and the global-RANSAC compute wall); (2) **coarse-to-fine**: detect on a ~5 cm copy, refine coefficients on 15 mm inliers; (3) **furniture robustness geometrically** (floor = lowest dominant horizontal, ceiling = highest, normal gating, min plane area) — segmentation masks plug in later via an idempotent G2 re-run (G6 is NOT a blocker); (4) **wall entities with thickness**: pair parallel planes < 300 mm apart with opposing normals (feeds runtime occlusion at doorways + Year-2 IfcWallStandardCase); (5) **watertight shell via general 2D wall-line graph** extruded floor→ceiling (no Manhattan constraint), corners sealed by plane intersection; (6) **byproduct outputs**: residual (non-planar) point clusters + DBSCAN bounding boxes — consumed by G3 detail meshes now, SAM prompts (G6) and Year-2 room emptying later; (7) **one-room validation slice before full-property run**. Ops to-dos: standardize repo on Python 3.11/3.12; pin Blender version for headless renders. Next: **G2 implementation**. |
| 2026-07-12 | **G2 implemented** — `vs_pipeline/stages/planes.py` per the locked brief: per-sweep proposals (sequential RANSAC on 5 cm coarse copies, point-normal gating, sensor-side orientation, floor/ceiling/wall classification), union-find local-to-global clustering (normal angle + offset + padded-bbox overlap = plane-bleeding guard; opposing wall sides stay separate), coarse-to-fine SVD refit on 15 mm member-sweep inliers (150k-pt cap/plane), wall-thickness pairing (< 300 mm, opposing normals), occupancy-grid + Douglas-Peucker boundaries in per-plane (u, v) frames (first-pass; exact corner sealing lands in G3), residual DBSCAN clusters + boxes. Outputs: `structural_planes.json`, `residual_clusters.json`, color-coded `planes_review.ply`. `--sweeps "30-40"` selection for the one-room slice. 9 unit tests incl. synthetic-room end-to-end and the bleeding-guard case. **Awaiting local validation: one-room slice first, then full property.** |
| 2026-07-12 | **G2 slice run (living room, sweeps 46–58, glass-heavy) reviewed** — true floor found (y = +0.003 m, 31.5 m², 9 sweeps), realistic wall-thickness pairs (187/181/105 mm), bleeding guard held; but 92 planes = under-merged duplicates (noisy per-sweep RANSAC vs tight proposal tolerances) + furniture tops classified “floor”. **Patched**: second union-find merge pass after the 15 mm refit (3° / 2.5 cm, same padded-bbox guard, pooled refit); storey-level reclassification (up-facing planes > 25 cm above a floor level → class `surface`, mirrored for ceilings); **2D review renders** (`planes_review_topdown.png` plan view — under-merge shows as double lines — and `planes_review_front.png` elevation) so review needs no 3D viewer. **Awaiting slice re-run + review.** |
| 2026-07-12 | **Slice re-run diagnosis**: 83 planes, review images unreadable — root cause is *scoping*, not fitting: the glass-heavy living room captures outdoor ground (up-facing plane at y = −1.36 m), the upstairs ceiling through the stairwell (+6.6 m), and neighboring rooms — a slice was never just the room. Also: large furniture/landings (≥ 8 m²) could define spurious storey levels, shielding fake floors from reclassification. **Patched**: `--crop-margin` room-crop mode (with `--poses`) crops each sweep cloud to the selected positions' bbox + margin; storey-level candidates must also be ≥ 30% of the largest floor area; sweep positions drawn as white crosses in the review renders. **Awaiting cropped slice re-run.** |
| 2026-07-12 | **G2 full-run review (198 sweeps, 824 planes)** — walls/ceilings healthy (126 thickness pairs; ceiling storeys at +2.9/+6.6 m), but floors failed to cluster at storey levels: the facade-inclusive scan yields ~940 m² of outdoor-terrain "floors" at y = −0.95…−1.69 which inflated the 30%-of-largest-floor bar to 26 m², so the fragmented second-storey room floors (10–13 m² each at y ≈ +3.60) were demoted to `surface_*` (the ground floor survived only via floor_12, 33.4 m² at −0.109). Diagnosed with the new read-only `inspect` subcommand. **Patched**: storey-level candidacy now requires interiority (ceilings 1.8–4.5 m above covering ≥ 50 % of the floor's plan bbox), with the 30 % bar computed over interior floors only; up-facing planes below the lowest storey level demote to `other` (terrain), not `surface`; new `reclassify` subcommand re-applies the rule to an existing `structural_planes.json` in seconds (classes reset from stored normals — idempotent, no refit; review renders not regenerated). **Awaiting: `reclassify` on planes_full + `inspect` re-check (expect floors only at the two storey bins, terrain → other, zero suspicious demotions).** |
| 2026-07-12 | **Interiority v1 failed on the full run** — terrain still defined all 16 storey levels: the ceiling-cover test *summed* bbox intersections over ~44 overlapping first-storey ceiling patches (massive double-count), and the 1.8–4.5 m window admitted ceiling-over-terrain gaps (3.4–4.3 m). **Interiority v2**: (1) ceiling cover is now a plan-view *union* on a 0.25 m grid (no double-count), bar tightened to 60 %; (2) *tripod evidence* — a level-defining floor must have a sweep position 1.0–2.0 m above it (Matterport tripods stand ≈ 1.4 m above the floor they rest on; terrain has no such sweep) — supplied via `--poses` on `fit`/`reclassify`/`inspect`, skipped when absent. **Awaiting: `reclassify --poses` + `inspect --poses` re-check.** |
| 2026-07-12 | **Audit verdict — the “terrain” reframe.** The level-candidate audit showed 57 sweeps standing 1.0–1.3 m above the y = −1.2…−1.7 planes: that band is a real walkable, scanned level (garden and/or walk-out lower level around a raised main floor), not terrain to exclude — and ceiling cover reads 100 % for every candidate (under-merged ceiling bbox unions blanket the plan), so it discriminates nothing. The one genuine G2 defect was the *relative* 30 %-of-largest bar suppressing the fragmented 10–13 m² upper-storey floors. **Storey levels v3**: elevation bins (0.15 m gap); a bin is a storey level when its biggest patch ≥ 8 m² AND ≥ 3 sweeps stand 1.0–2.0 m above it (furniture bins fail both: biggest 5.27 m², 2 stray stair sweeps); no relative bar; ceiling cover kept only as a no-poses fallback. **Awaiting: `reclassify --poses` (expect surface_00–05 → floor, furniture bin → surface, walkable low bins stay floor) + pipeline pytest.** |
| 2026-07-12 | **Storey levels v3 validated on the full run**: `reclassify --poses` produced 17 correct promotions — the entire second-storey floor restored (13 patches at y ≈ +3.59…+3.62), the −0.828 garden step and the +0.152 podest back to floor, furniture bins untouched — four clean level bins (−1.7, −1.3…−0.8, −0.16…+0.15, +3.6); classes now 77 floor / 25 surface / 52 ceiling / 37 other / 633 wall. Two unit tests failed because they exercise the *no-poses* path with no ceilings, where v3's fallback could never find levels; **fixed**: the no-poses fallback restores the legacy rule (relative 30 % bar + ceiling cover when ceilings exist), tripod evidence stays the primary poses-path rule. **Awaiting: pytest green + final `inspect --poses` (zero suspicious demotions) → G2 sign-off.** |
| 2026-07-12 | **G2 VALIDATED — box checked.** 11/11 plane tests green; `inspect --poses` clean (zero suspicious demotions); walls top-down confirmed as a coherent full floor plan (straight double-line walls, closed rooms; residual candidate noise is G3's input by design). Property layout confirmed from the tour: house raised ~5 steps above a walkable garden/yard at y = −1.2…−1.7 (57 outdoor sweeps), main storey at ~0, second storey at +3.6. *Non-blocking follow-ups*: (1) one `test_pose_qa` assertion fails locally (edge-overlap score) — investigate during F7.2; (2) review renders in `planes_full` predate the reclassification — regenerate on the next `fit` run. **G3 design brief locked**: (1) **wall-line graph per storey** — project wall planes to plan view (paired walls → centerline + thickness, unpaired → single line), snap/intersect into a planar graph (Shapely), no Manhattan constraint; (2) **room polygonization** — closed faces of the graph become room candidates; **wall-enclosure test**: floor patches without an enclosing wall loop (garden/yard) are excluded from shell synthesis — this is where interior/exterior separation properly lives; (3) **authoritative plane selection per room** — pick the floor/ceiling/wall planes covering each room polygon, discarding duplicate/fragment candidates (resolves G2 under-merge); (4) **shell synthesis** — extrude room polygon floor→ceiling, corners sealed exactly by plane intersection (< 1 mm), openings cut via Clipper (door/window rects from wall-boundary holes now, G6 masks later), labeled sub-meshes (floor/wall_NN/ceiling), < 5,000 tris/room; (5) **non-planar details** — residual clusters + OBJ-clipped detail meshes merged (Decision 6); (6) **outputs** — `proxy.glb` (Draco: pos 14 / normal 10 / UV 11 bits) + per-room metadata into `structural_planes`/rooms; (7) **validation path** — one-room slice first, then full property, with 2D review renders per storey. Next: **G3 implementation**. |
| 2026-07-12 | **G3 slice 1 implemented** — `vs_pipeline/stages/rooms.py` (brief steps 1–3): storey detection (floor elevation bins + tripod evidence, G2 v3 rule family; per-storey ceiling from the dominant ceiling band 1.8–4.5 m above); per-storey wall-line graph (wall planes → plan segments via endpoint PCA; paired walls collapse to a centerline with measured thickness; canonical merge of collinear duplicates = dedup of G2 under-merge); room polygonization (segments extended 0.3 m to close corners, Shapely `unary_union` + `polygonize`; face accepted as room when floor cover ≥ 30 % or sweep-supported with cover ≥ 10 % — kills garden-storey ghost faces); per-room election (floor/ceiling patch with maximal plan overlap wins, walls by boundary proximity); **wall-enclosure test**: floor patches mostly outside all rooms → `exterior_floors` (the garden). Pure JSON → JSON (`structural_planes.json` + `poses.json`), seconds, no point clouds. Outputs `rooms.json` + `rooms_review_storey<k>.png`. Shapely 2 added to pipeline deps (`pip install -e pipeline` after pull). 4 unit tests (two-room polygonize + election, garden exterior, duplicate merge, pair centerline). **Awaiting local validation: pytest, then full run on planes_full + storey render review.** |
| 2026-07-12 | **G3 slice 1 first full run**: 3 storeys detected (garden 1 “room”, ground 8, upper 9), 1,136 m² of exterior floors correctly excluded via the wall-enclosure test; renders read as partitioned floor plans. First-run defects: (a) two missing rooms (wall gaps wider than the 0.3 m corner snap — doorways and/or genuinely open sides), (b) sliver “rooms” (1.8–2.9 m² wedges between unpaired opposing wall faces), (c) cosmetic corner overshoot of PCA extents. **Patched**: dangling-end closure (endpoints with no wall within 0.35 m ray-extend to the first wall hit ≤ 1.2 m — seals doorways; the lintel continues the wall in reality) and sliver rejection (faces narrower than 0.6 m via negative buffer). 2 new tests (17 total). **Awaiting: re-run + render comparison; verify in the tour what the voids are (garage/porch with an open side cannot and should not be sealed by snapping — needs ceiling-outline closure later) and whether s0_room_00 (38.9 m², garden storey) is a street-level garage.** |
