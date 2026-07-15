# Open Problems, Deferred Work & Enhancement Candidates

> Living catalog of everything we hit and did NOT fully solve, plus known
> approximations that need hardening. Each entry carries the context and
> pointers a research agent needs. Ordered by pipeline stage.
> Status legend: **OPEN** (unsolved), **DEFERRED** (consciously postponed),
> **MITIGATED** (workaround in place, root cause remains), **TUNE** (works,
> thresholds unvalidated beyond one property).

Property used for all findings so far: 198-sweep Matterport E57, two-storey
house raised ~5 steps above a walkable garden, garage under the house at
garden level, heavy glazing in several rooms. Frame: runtime Y-up, E57
origin locked (see PLAN.md §1).

---

## G2 — Plane fitting (`pipeline/vs_pipeline/stages/planes.py`)

### 1. Under-merged duplicate plane candidates — MITIGATED
The same physical wall/floor/ceiling appears as several coplanar patches
(e.g. ground-floor bin: 279 m² of “up” area for a ~120 m² footprint; 633
wall candidates). Root cause: per-sweep RANSAC proposals + conservative
union-find merging (angle/offset/padded-bbox); the second refit-merge pass
(3° / 2.5 cm) still leaves duplicates whose offsets differ by 3–10 cm.
Downstream we compensate: rooms.py canonical segment merge (collinear
within 8 cm), per-room plane election, interval-union coverage tests.
The duplicates repeatedly caused secondary bugs (double-counted coverage
metrics — see #4 and #12). A cleaner G2 would shrink a lot of downstream
complexity.
- Research: region-growing plane segmentation vs sequential RANSAC;
  Schnabel “Efficient RANSAC”; CGAL Shape Detection / Polygonal Surface
  Reconstruction; global plane regularization (parallelism/orthogonality
  priors); “plane merging point cloud segmentation”.

### 2. Glass & mirrors corrupt LiDAR geometry — OPEN (biggest recurring theme)
Glass-heavy rooms have almost no wall inliers (two rooms produced no room
polygon at all: sweeps 58–61 storey 1, sweep 175 storey 2). Mirrors/windows
also caused most Gate-1 pose-QA review flags (47/198, all content-driven).
Current mitigations: closure tooling (#8), two-signal QA flagging.
No detection/removal of glass artifacts at the point-cloud level.
NEW OBSERVATION (2026-07-12): mirrors also INJECT geometry — the S2
bathroom (two mirrors + glass window) produced short DIAGONAL phantom
wall stubs (reflected returns fitted as planes at bogus angles) that
initially mimicked stairwell geometry and locally corrupt the wall-line
graph. Mirror-artifact detection would clean both G2 candidates and the
room graph. Search: “mirror artifact removal laser scanning indoor”,
“specular reflection point cloud phantom geometry”.
- Research: “glass detection terrestrial laser scanning”, “specular/mirror
  artifact removal LiDAR indoor”, “transparent surface reconstruction”;
  also G6 (Mask2Former window class) can feed masks back into Gate 1 and G2.

### 3. Windows are invisible to boundary extraction — DEFERRED
`extract_boundary` uses `cv2.RETR_EXTERNAL`, so interior holes (windows)
are lost; only floor-reaching notches (doors) survive. Fix path already
agreed: RETR_CCOMP hole extraction + `openings_json` per plane (schema has
the column reserved), and/or G6 window masks. Needed for window openings in
the proxy, BIM IfcOpeningElement/IfcWindow, and Year-2 re-texturing.

### 4. Storey-level classification — MITIGATED (v3), fallback weaker
History (full detail in PLAN.md changelog 2026-07-12): relative
30%-of-largest-floor bar let ~940 m² of terrain suppress the fragmented
upper-storey floors; interiority v1 (summed bbox ceiling cover) and v2
(union cover + tripod evidence as AND) both failed on real data — cover
reads 100% everywhere because duplicate ceiling patches blanket the plan.
Final v3 rule (validated): elevation bins; a bin is a storey level iff
biggest patch ≥ 8 m² AND ≥ 3 sweeps stand 1.0–2.0 m above it. Residual
weaknesses: (a) the no-poses fallback reverts to the legacy relative-bar
rule (weak); (b) walkable exterior ground is classed `floor` by design and
only separated later by wall enclosure (#11); (c) no dedicated `terrain` /
`exterior` semantic class — below-level planes get demoted to `other`.
- Research: “storey segmentation indoor point cloud”, “indoor outdoor
  segmentation TLS”, scan-to-BIM storey splitting.

### 5. Sloped surfaces unmodeled — OPEN
37 planes are classed `other` (slopes). Attic/roof-pitch rooms, stair
soffits and ramps cannot be represented by the current horizontal-cap room
prisms (#14). Affects proxy occlusion and Year-2 BIM (IfcRoof).

### 6. G2 runtime — TUNE
Full 198-sweep fit ≈ 15–40 min (proposals + refit + residual DBSCAN).
O(n²) union-find loops over proposals/planes. Fine at house scale; the
50-scan acceptance set and industrial scans may need spatial indexing,
residual downsampling knob, or parallelism.

---

## Gate 1 pose QA (`pose_qa.py`, F5) — all non-blocking, F5 validated

### 7. Metric limitations — TUNE
- Chance-corrected score distribution is content-dependent (p10 0.20 / p50
  0.45 / p90 0.75); gross threshold 0.20 and margin threshold 0.03 were
  hand-calibrated on ONE property. Re-tune on the 50-scan set.
- Perturbation self-test coverage: only sweep 001 (5 cm + 2°). Extend to
  texture-sparse sweeps during F7.2.
- Sub-cm drift is below the detection floor (MatterPak OBJ decimation,
  ~5 px @ 512). Accepted; F6.3 micro-adjust exists only as a design.
- Render-batch homogeneity guard still missing (stamp renderer version per
  render dir; mixed batches caused a stale-calibration bug once).
- 47/198 human-review rate is high; feeding G6 glass/mirror masks into the
  scoring mask is the agreed reduction path.

---

## G3 — Rooms & proxy (`rooms.py`, `openings.py`, `proxy.py`)

### 8. Two glass rooms unclosed — OPEN (fix in flight)
Closure history: v1 zero virtual walls (double-counted coverage + sawtooth
union boundary vs 5 cm simplify); v2 over-triggered 104 segments (ceiling
union includes eaves/porch soffits; boundary sits at the EAVE line, not the
facade); v3 need-driven (orphan sweeps) built a porch room but missed both
voids — orphan test wrongly counted unaccepted faces as housing, and
eave-line virtual walls are simply in the wrong place for facade glass.
v4 (pending validation): orphan test vs ACCEPTED-quality faces + orphan-
scoped dangle extension to 4 m using real wall stubs (sills/lintels
continue across windows) with ceiling outline as secondary source.
Open question worth investigating with the wall candidates directly
(`planes_review_topdown_walls.png`, `structural_planes.json` near the
voids): why are the sill/lintel wall planes around these glass rooms so
fragmented that gaps exceed 1.2 m? Knobs: `CEILING_EDGE_COVERED_FRACTION`
(0.6), `CEILING_EDGE_WALL_TOL_M` (0.3), `CEILING_SIMPLIFY_M` (0.15),
`DANGLE_ORPHAN_EXTEND_M` (4.0), `ORPHAN_RADIUS_M` (6.0).
ROOT CAUSE FOUND (probe, 2026-07-12): both voids have NO plane classed
`ceiling` above them (`ceiling union covers: False`) despite being real
interior rooms — the ceilings are plausibly the down-facing `other` planes
(demoted by the 0.25 m ceiling-level tolerance, e.g. +6.07 vs the +6.59
level; kitchen bulkheads at +1.7). v5 (pending validation) expands the
evidence: overhead union includes down-facing `other` planes, and the
storey FLOOR outline becomes a co-equal virtual-wall source (floors are
solid and end at the glass; floor-evidence orphans must be wall-adjacent
to keep the open yard out). Still open: why G2 misses/demotes these
ceilings at all — ties into #1/#2 and the 0.25 m `LEVEL_TOLERANCE_M`
ceiling rule; the demotion-to-`other` of near-level ceilings deserves its
own revisit.
FINAL STATUS (2026-07-12): v5 also failed — floor outlines are too noisy
(fragmented patches, furniture shadows → 297 virtual segments across the
three storeys, the garage shattered, healthy rooms fragmented, and the
voids STILL open). Auto-closure of glass rooms is hereby PARKED AS
RESEARCH; the shipped production path is operator-authored manual walls
(`rooms build --close-file closures.json`, endpoints read off the probe
renders' 1 m coordinate grid). Research directions: Floor-SP / RoomFormer
/ HEAT-style learned floorplan reconstruction; G6 window masks to recover
glass wall lines from imagery; fixing the G2 near-level ceiling demotion
and under-merge so honest overhead evidence exists in the first place.
- Research: Floor-SP, RoomFormer, HEAT (holistic edge attention
  transformer), “floorplan reconstruction from point clouds”, PolyFit —
  as alternatives or validators for the whole wall-graph→polygonize chain.

### 9. Room-graph heuristics tuned on one property — TUNE
`SNAP_EXTEND_M` 0.3, `DANGLE_MAX_EXTEND_M` 1.2, `MIN_ROOM_WIDTH_M` 0.6,
acceptance (floor cover ≥ 0.3, or sweep-supported ≥ 0.1), storey ceiling
from dominant band 1.8–4.5 m. All validated on this house only. Known
cosmetic issue: canonical wall segments overshoot corners (PCA extents);
some walls render slightly non-straight — unverified whether architecture
or fit noise (compare against panos).

### 10. Election coverage is thin in places — OPEN (observe)
Some accepted rooms elect floors at 9–29% single-patch coverage and several
rooms have `ceiling None` (stairwells/voids/double-height). The prism model
uses a storey-level fallback ceiling there. Stairwell/void rooms need a
proper multi-height treatment eventually (#14).

### 11. Interior/exterior separation lives in room acceptance — MITIGATED
Garden/yard floors (≈ 1,136 m², 53 planes) are excluded via the
wall-enclosure test (floor patches < 30% inside any room). The garage
(open side, garden level, `s0_room_00`) DID polygonize via ray closure with
low cover (13%/12%) — whether an open garage should receive a proxy shell
is an unresolved product decision. No explicit exterior semantic class.

### 12. Door detection limits — DEFERRED items
`openings.py` detects floor-reaching notches only. Known gaps:
- max width 1.4 m → double/patio doors missed (one 1.35 m detection sits
  at the ceiling of the range — unverified whether double door or false
  positive from glass);
- runs touching a wall-plane end are skipped as ambiguous → doors at wall
  ends missed;
- glass doors leave no notch at all;
- 54 detections were sanity-checked in aggregate, not per-door; per-door
  verification deferred to downstream visual gates (G8, staging QA).
- Research: “door window opening detection point cloud scan-to-BIM”.

### 13. Proxy shell approximations — OPEN list
- Caps are flat at the elected elevation; actual planes may tilt slightly
  (not snapped to plane coefficients). Sub-cm effect, revisit for BIM.
- Virtual (ceiling-closure) walls carry default 100 mm thickness.
- `inset_fallback` path (degenerate insets) skips opening cutouts.
- Door cutouts intentionally break per-room watertightness; a global
  cross-room seal (door tunnels between rooms) does not exist yet.
- No Draco compression yet (plan: pos 14 / normal 10 / UV 11 bits);
  trimesh cannot Draco-encode — use gltf-transform or Blender in P4 or a
  G3 finishing pass.

### 14. Non-planar detail geometry — MITIGATED (shipped 2026-07-12)
`details.py` ships: 136/2,970 clusters → 100 detail proxies (27,586 tris),
staircase captured (7.8 × 3.5 × 4.3 m). Remaining questions: (a) is the
MatterPak-decimated silhouette fidelity ENOUGH for occlusion realism —
gates at the G8 milestone (test object occlusion, grounded sofa); (b) the
95.4 % cluster cut is threshold-driven (MIN_CLUSTER_POINTS 100 etc.) and
unvalidated beyond this property; (c) possible garden-clutter leakage in
the garage area (large boxes upper-right of storey 0 — flagged in review,
not yet investigated); (d) detail proxies are box-clipped, not
semantically segmented — G6 masks + 3D furniture boxes will supersede the
selection for Year-2 room emptying.

---

## Upstream/platform items parked earlier

### 14b. G4 follow-ups — OPEN (non-blocking, logged 2026-07-12)
- Detail proxies can ENCLOSE sweep positions: sweep 50's G-buffer shows
  0.04 m depth medians on two faces — a residual-cluster box (furniture
  group) likely swallowed the camera. At runtime this corrupts occlusion
  for that sweep. Fix path: details.py containment check (shrink/split
  boxes that contain a sweep position, or carve a clearance sphere).
- Full-property 2048² G-buffer bake HALTED at 49/198 sweeps — local DISK
  EXHAUSTION (2026-07-15). The bake is resumable; nothing lost. Root cause:
  output footprint — dual EXR + npy at 2048² for depth + world-normal is
  ~300–400 MB/sweep (~60–80 GB full property). Storage-optimization
  candidates before resuming: float16 storage (depth precision check
  first), compressed .npz instead of raw .npy, drop the dual-format
  redundancy (npy only until P4), optional half-res normals. WORKFLOW
  POLICY (2026-07-15): all full-property batch runs are DEFERRED — stages
  validate on sweep slices only; full runs queue until disk budget and/or
  storage optimization lands. Applies to the G5 full IBL bake too (IBL
  output is small, ~3 MB/sweep, but the policy is uniform).
- KTX2 packing (toktx) deferred to P4 alongside Draco.
- Alignment score absolute values are low (+0.04 mean) by nature (smooth
  proxy = few edges); composites are the gate, the score is only a trend
  indicator — revisit if a quantitative proxy-QA gate is ever needed.

### 15. F6 dashboard — property list + pipeline stage status need backend
wiring (F6.1); F6.3 micro-adjust sliders deferred until a genuinely drifted
pose ever appears (zero evidence to date).

### 16. F7.2 — 50-scan pixel-accuracy acceptance pending multi-property data.
Also re-validate all TUNE-tagged thresholds there (#7, #9).

### 17. G5 IBL — interim blind highlight boost is a placeholder
`C_hdr = C_lin × (1 + 12·max(0, Y−0.85)^1.5)` must be replaced by
segmentation-guided highlight reconstruction before ship (chrome-sphere
test is the G8 gate).
- Research: inverse tone mapping / LDR-to-HDR panorama reconstruction
  (StyleLight, EverLight), indoor lighting estimation from panoramas.

### 18. G6 segmentation — STARTED 2026-07-15 (slice 1: per-face 2D inference
shipped). Consumers waiting: Gate-1 mask feedback (#7), window openings (#3),
segmentation-guided highlight boost (#17), Year-2 room emptying. Later
slices: SAM boundary refinement, 2D→3D fusion (G-buffer depth lift → wall
openings + 3D furniture boxes), Gate-2 click-fix mask editor.

### 20. G6 model variants constrained by dev hardware — DEFERRED (worker upgrade)
Dev GPU is a 4 GB GTX 1650-class laptop card. Locked LOCAL configuration:
Mask2Former **Swin-T** ADE20K semantic (HF `transformers`, no detectron2
build) + fp16, faces resized to 640 px for inference, masks stored at that
size; SAM **ViT-B** planned for slice-2 refinement (MobileSAM fallback if
4 GB still chokes). CPU fallback works but is roughly an order of magnitude
slower. DEFERRED to P3 GPU workers (L4/A10G): **Swin-L**, **SAM ViT-H**,
higher mask resolution, batched inference. The variant is a CLI parameter
(`--model`, `--mask-size`) and part of the per-sweep idempotency identity,
so the upgrade is a parameter-swap re-run with zero code change (old masks
re-infer intentionally). Quality delta Swin-T vs Swin-L on our 7-class
mapping is UNMEASURED — measure before trusting Swin-T masks for Gate-1
exclusion (#7) and highlight reconstruction (#17), e.g. on the 50-scan set
or first worker deployment.

### 19. MatterPak OBJ decimation — accepted limitation
Causes QA render softness and the pose-QA detection floor; OBJ is only a
QA/dollhouse/morphing asset (E57 is metric truth), but if a higher-fidelity
mesh source appears, several limits above relax.

---

## Resolved-but-instructive (for the record)
These were fully fixed but the failure patterns are worth remembering:
- Sum-based coverage metrics double-count under-merged duplicates — bit us
  TWICE (G2 interiority v1, G3 closure v1). Any future coverage test must
  union intervals/areas, never sum per-candidate.
- Relative (“% of largest”) thresholds break when an outlier class inflates
  the reference (terrain vs upper-storey floors). Prefer absolute physical
  invariants (tripod height 1.0–2.0 m above a walkable floor).
- `shapely buffer(0)` can return MultiPolygon — handle in every renderer.
- Empty-input guards must sit AFTER normalization too (`pose_qa`
  chance-correction turned a guarded 0.0 into −0.08).
