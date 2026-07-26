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
diagnostic print.
**Outcome (2026-07-17):** UNSETTLED — the heuristic never triggered a
decode AND the one-time diagnostic fired on an empty outdoor face (all
zeros), wasting the read (fixed: diagnostics skip degenerate inputs).
Two sub-lessons banked: (a) auto-detection heuristics on mixed content
can be inconclusive (encoded mixed scenes can median near 1.0) — value
semantics must ultimately be verified at the SOURCE (the
`render_gbuffer.py` write path), not inferred downstream; (b) one-time
diagnostics must validate their sample is non-degenerate before
spending the print. The mirror detector that motivated this was parked
(OPEN_PROBLEMS #2); resolve the encoding question at the renderer
before the NEXT normal consumer (relighting, re-texturing, BIM face
orientation) is built.
**SETTLED (2026-07-26):** the fixed diagnostic fired on a VALID face
during the fusion slice-extension run: min −1.000 / max +1.000 /
median|n| 1.000 → G-buffer normals are stored as SIGNED unit vectors
(no (n+1)/2 encoding); `_decode_normals` correctly never decodes, and
the mirror-detector park stands on PREMISE grounds (#2), not data
grounds. Future normal consumers can trust signed semantics — with
the usual L1 range print on first contact.

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

### L17. Third-party format/API facts: verify at the SHIPPED source, never from search results
The G7.2 `.env` research (web search summary) contradicted the installed
Babylon source on FOUR load-bearing points: the magic bytes (claimed an
ASCII header string; reality is 8 binary bytes), the manifest schema
(claimed `{lod,face,offset,size}`; reality is `{length,position}`
mip-major), the RGBD math (claimed shared-exponent `2^(8(A-1))`, i.e.
RGBE; reality is `D = floor(255/max)/255` premultiply), and gamma
(claimed linear-in-PNG; reality is gamma-encoded rgb, decoded
`pow(rgb,2.2)/a`). Building on the research would have produced files
that fail Babylon's header check on byte 0 — or, worse, load with
uniformly wrong lighting. `viewer/node_modules` IS the ground truth for
the exact installed engine version; four pasted snippets locked the
format in one round.
**Rule:** any format or API consumed across a language boundary gets its
facts from the shipped source of the EXACT installed version, plus a
parser/round-trip test in our codebase (`ibl_export.read_env`). Search
results and LLM summaries are hypothesis generators, never format specs.
Corollary (same family as L1): a plausible, detailed, confident spec is
still unvalidated value semantics.
**Fix record:** `ibl_export.py` (docstring carries the verified facts +
source pointers); `test_ibl_export.py` round-trip tests.

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
- Pano orientation chain (LOCKED 2026-07-17, G7.1): reflection matrix =
  pose_rt rotation filled VERBATIM into Babylon row-major storage (acts as
  R^T = world->local on sampling directions) times SKYBOX_BASE
  (mirrorX . rotX-90); residual yaw 0. Rotation source = E57 SCAN headers
  (legacy 3d-viewer confirmed identical source).

---

## Orientation & frames (G7.1 additions, 2026-07-17)

### L13. World-space corrections PRE-multiply Babylon reflection matrices
Babylon's `A.multiply(B)` applies A THEN B to a row vector. The G7.1
round-1 yaw harness post-multiplied `RotationY` onto the skybox base, so
the "yaw" acted in cube space around a HORIZONTAL axis -- the pano pitched
down instead of yawing, and a whole hypothesis round was wasted testing
garbage.
**Rule:** corrections meant in world space compose BEFORE the world->cube
(or any frame) conversion; every matrix-composition site states its
convention in a comment.
**Fix record:** `tourViewer.applySkyboxOrientation`.

### L14. Per-sweep local-frame data can pass EVERY global consistency check
The extracted pano faces live in each sweep's LOCAL tripod frame. This
passed F4 validation (seams, up/down, readability, continuity -- all
satisfiable per-sweep because tripods are level), F5's GLOBAL face-mapping
calibration (0.87-0.97 across 198 sweeps), G4 verification and G6 fusion --
because every one of those consumers builds its cameras FROM the pose
rotations, i.e. compares in the local frame. The viewer was the FIRST
world-aligned consumer, and it exposed the gap instantly (G7.1): no global
constant could align what needs a per-sweep rotation. The operator's
memory of the legacy project (applying per-sweep E57 quaternions to the
pano) was the decisive clue.
**Rule:** for any per-item dataset, ask explicitly at design time: is this
in the ITEM frame or the WORLD frame? "Validated" only means validated
under the consumers exercised so far; schedule a world-aligned consumer
early when one will eventually exist.
**Fix record:** `loadTour` (pose_rt rotation join) +
`tourViewer.applySkyboxOrientation` (R^T per sweep).

---

## Provenance & physical plausibility (G7.1 triage additions, 2026-07-18)

### L15. Aggregate counts are NOT artifact provenance
The G7.1 "freshness check" matched a rebuilt proxy against the validated
closure-final numbers (23 shells / 810 tris) and passed -- but the
rooms.json underneath was built WITHOUT `--close-file closures.json`:
a different room layout coincidentally produced the same counts, and the
entire G7.1 proxy era ran closure-less unnoticed. The kitchen "void" and
missing floors were partly CONFIG artifacts, not geometry defects. The
probe compounded it by polygonizing without manual walls, issuing
misleading "true void" verdicts.
**Rule:** artifact freshness/provenance is verified by comparing the
BUILD CONFIG (inputs, flags, thresholds), never output aggregates.
Stages stamp their params into their outputs; diagnostic tools consume
the SAME config as production or their verdicts mislead.
**Fix record:** `rooms.build_property` build_params stamp;
`rooms probe --close-file`; `inject_manual_walls` shared by both paths.

### L16. Paper-thin opposing-normal pairs are objects, not walls
A 3.3 mm "wall pair" (wall_389+wall_629) split the living room with a
6.9 m phantom interior wall that occluded staged furniture and captured
floor double-clicks (occluders pick on both faces with backFaceCulling
off). Two opposing-normal planes a few mm apart are a freestanding thin
object seen from both sides (curtain, glass panel, divider) or a
flipped-normal duplicate fit -- never a wall.
**Rule:** geometric entities pass a physical-plausibility gate (walls
have thickness >= ~50 mm) before entering structural graphs; prefer
absolute physical invariants (see L4).
**Fix record:** `rooms.wall_segments` MIN_PAIR_THICKNESS_M +
`test_rooms_thin_pairs.py`; glass-wall tension tracked in
OPEN_PROBLEMS #24.
