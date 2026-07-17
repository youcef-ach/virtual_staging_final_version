# Lessons & Critical Fix Records (internal playbook)

> **Purpose & scope:** validated failure patterns, critical fixes, and
> hard-won design rules that future stages MUST respect. This is NOT the
> research catalog — unsolved problems that need deep external research
> live in `OPEN_PROBLEMS.md`. This file is the internal playbook: read it
> before designing any new stage or consumer.
>
> Format: each entry = the rule, the incident that taught it, and where
> the fix lives.

---

## Data semantics & measurement

### L1. Never gate on a data channel whose value semantics are unvalidated
The G6 mirror normal gate was built assuming signed unit vectors and
silently passed everything — ceilings survived a VERTICALITY gate, which
is physically impossible with correct normals. Likely cause: normal maps
stored ENCODED ((n+1)/2 in 0..1); an encoded down-normal (0,0,−1) becomes
(0.5, 0.5, 0) and reads as “vertical”.
**Rule:** the FIRST consumer of any new baked channel prints value-range
diagnostics (min/max/median-length) before any logic is trusted.
**Fix record:** `fusion._decode_normals` auto-detects and decodes
(no negatives AND median length away from 1 ⇒ encoded → 2v−1) + one-time
diagnostic print. PENDING (2026-07-17): if the next run confirms encoding,
fix at SOURCE in `render_gbuffer.py` (write raw signed normals, npy
siblings included) and drop the consumer-side decode — every future
normal consumer (relighting, re-texturing, BIM face orientation) would
otherwise need the same workaround.

### L2. Consumers of PROXY depth must respect its intentional holes
G3 cuts door openings into the proxy, so “lift through depth” sends door
pixels through the hole onto geometry in the room behind (G6 slice 3
first run: 0/54 door match; windows worked ONLY because the proxy has no
window holes yet — a silent coincidence, not correctness).
**Rule:** geometric primitives (ray-plane intersection) are the
measurement; depth is demoted to an OCCLUSION GATE (hit at/beyond the
target passes, hit clearly in front blocks). Every future depth consumer
(T1 line-of-sight, T2 morphing, Year-2 re-texturing) must decide its
hole-behavior explicitly at design time.
**Fix record:** `fusion.ray_plane_lift`.

### L3. Cross-stage comparisons must match GEOMETRY, not entity ids
The G6 door cross-check required plane_id equality and read 1/54: with
G2's under-merged duplicate walls, the same physical door lives on
different plane ids in different stages. Duplicates have now broken
id-or-count-based logic THREE times (sum-based coverage, relative
thresholds, id-equality matching).
**Rule:** any comparison across stages happens in world space (distance /
overlap), never by id, count, or per-candidate summation.
**Fix record:** `fusion.match_doors` (horizontal center distance +
vertical interval overlap, reports both directions).

### L4. Coverage/threshold logic under duplicate entities (recurring family)
- Sum-based coverage metrics double-count under-merged duplicates — bit
  us TWICE (G2 interiority v1, G3 closure v1). Union intervals/areas,
  never sum per-candidate.
- Relative (“% of largest”) thresholds break when an outlier class
  inflates the reference (terrain vs upper-storey floors). Prefer
  absolute physical invariants (e.g. tripod height 1.0–2.0 m above a
  walkable floor).
Until G2 under-merge (#1 in OPEN_PROBLEMS) is solved at the root, EVERY
new consumer of `structural_planes.json` must assume the same physical
surface appears as multiple candidates.

---

## Validation & review discipline

### L5. Review artifacts must be verifiable against the PHOTOGRAPH
The G6 (u,v) occupancy grids were mechanically honest but the operator
could not judge them — and was right to refuse to sign off. Reprojecting
results onto the pano (the F5 ghosting-proof style) made correctness
self-evident AND instantly disproved an orientation-bug hypothesis (rects
framing real windows/doors proves the whole alignment chain end-to-end).
**Rule:** every visual gate renders its evidence over the photography;
abstract evidence maps are secondary at best.
**Fix record:** `fusion._reproject_openings_review`.

### L6. Perturbation self-tests prove a metric can detect what it claims
A metric that scores well on correct data means nothing until it is shown
to DEGRADE on corrupted data (F5: deliberate 5 cm + 2° pose corruption
exposed chance-level saturation of the raw edge-overlap score; the fixed
metric + orientation margin separated cleanly).
**Rule:** new QA metrics ship with a perturbation self-test.

### L7. Renderer/batch homogeneity: stamp versions, refuse mixed batches
The F5 stale-calibration bug came from scoring renders produced by two
different renderer configurations in one directory. Guard still unbuilt
(OPEN_PROBLEMS #7 carries it); until then, delete stale renders before
re-rendering and re-lock calibrations after any renderer change.

### L8. Orientation contracts: one locked op, exact inverses, round-trip tests
Every pixel-space consumer must state which layout it works in (stored
canonical vs upright vs render) and convert via the LOCKED F5 calibration
op or the canonical-transform inverse — with a round-trip unit test.
Incidents: F4.4 calibration itself, the G6 upright fix (models need
gravity-down images), `fusion.invert_orientation`. When in doubt, apply
the op to the DATA ARRAY (lossless permutation) rather than re-deriving
geometry in a different layout.

---

## Environment & tooling

### L9. Blender 4/5 dual-API compositor pattern
`scene.use_nodes`/`node_tree` are gone in Blender 5; the proven pattern
(`compositing_node_group` with NodeGroupOutput on B5, `scene.node_tree`
fallback on B4, passes routed to the scene output with
`write_still=True`) lives in `render_edges.py` and `render_gbuffer.py`.
Any new headless renderer starts from that pattern.

### L10. Windows OpenCV ships without OpenEXR: npy siblings are first-class
All EXR-producing renderers also write `.npy` siblings; all loaders try
npy first (`gbuffer._load_depth`, `fusion._load_normal`). Never rely on
cv2 EXR I/O locally. Also: cv2 loads color EXRs as BGR — vector-valued
channels (normals) must be reversed to XYZ.

### L11. Verify commits actually landed before acting on them
Two payload-less commit failures in one week (an agent crash mid-commit;
a dropped file payload on retry). The fix chain was only safe because the
repo history was checked FIRST.
**Rule:** after any commit, confirm it in history; after any pull,
confirm the expected files changed. The repo is the single source of
truth — an unlanded commit means the truth didn't change.

### L12. Idempotency identity = the full parameter dict
Cache-skip logic must compare the COMPLETE params identity (the G6
`upright` bump taught the contract; regression tests now lock it for
infer/refine). New stages copy that pattern: any semantic change bumps
the params dict, and outputs re-generate without `--force`.

---

## Property-specific facts worth remembering (this test property)

- 198-sweep two-storey house RAISED ~5 steps above a walkable garden
  (y ≈ −1.2…−1.7), garage under the house, storeys at ~0 and +3.6;
  poses.json indices 0–2 are OUTDOOR garden sweeps — do not use them to
  judge interior stages.
- Validation slice: 1 (garden), 50 (living), 60 (kitchen), 152/157
  (bathrooms — mirror acid test), 175 (glass room). G-buffer renders on
  disk are 1024².
- Sweep 50 has a detail-proxy box likely enclosing the camera (0.04 m
  depth medians on two faces) — known occlusion hazard, containment check
  queued (OPEN_PROBLEMS #14b).
- Mirrors segment their REFLECTED content as real room content and the
  raw ADE `mirror` label does not fire; mirror surfaces label as wall
  (G6 acid test) — any mirror logic must come from fusion evidence, not
  the 2D labels alone.
- Slice-only validation policy in force since 2026-07-15: no
  full-property batch runs until disk budget / storage optimization
  lands.
