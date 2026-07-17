"""Stage 2 (part 3) — automated pose-QA scoring (Gate 1).

Pipeline:
1. ``prepare``: read poses.json (runtime Y-up), convert back to the Matterport
   frame with the exact inverse of the ingestion conversion, and emit
   render_jobs.json — one 90-degree pinhole camera per cube face per sweep
   (camera-to-world matrices for Blender: X=right, Y=up, -Z=view).
2. Blender renders normalized depth per face (pipeline/blender/render_edges.py).
3. ``score``: Canny edges of the rendered depth vs dilated Canny edges of the
   extracted panorama face -> chance-corrected edge-overlap per face: the raw
   overlap is normalized by the pano edge-mask density ((raw - density) /
   (1 - density)), so texture-dense and texture-sparse faces are comparable
   and the score responds to pose error rather than content. Sweep score =
   mean over informative faces. Flagging is two-signal: sweep score below
   the gross-failure floor (misregistration renders wrong geometry -> score
   collapses toward chance ~0), OR mean orientation margin below
   MARGIN_THRESHOLD (per face, the locked orientation's normalized overlap
   must beat the best wrong orientation's -- a correct pose makes the locked
   orientation stand out, drift collapses that gap, and content cancels out).
   A deterministic ~10% of sweeps is always marked for spot-check.

   With ``--seg-dir`` (G6 slice 4.2), window pixels from the 7-class
   segmentation masks (SAM-refined when present) and mirror pixels from the
   RAW ADE map are excluded from BOTH edge maps before scoring: glass shows
   outdoor content and mirrors reflected content that the rendered mesh
   cannot match -- the dominant false-flag source in the full-run review.
   Chance correction normalizes by the non-excluded pixel count, and faces
   whose remaining render edges drop below the informative floor are
   excluded rather than punished. Sweeps without segmentation masks score
   exactly as before (graceful fallback).

Face-mapping auto-calibration: extracted faces carry the canonical Set A
transforms (including the +Z/-Z content swap) plus the viewer-level mirror,
so extracted face N does not necessarily correspond to rendered face N.
Calibration searches ALL 6 rendered faces x 8 in-plane orientations per
extracted face on sample sweeps and locks a bijective assignment — the
overlap metric itself is the calibration signal.

CLI:
    python -m vs_pipeline.stages.pose_qa prepare poses.json -o render_jobs.json
    python -m vs_pipeline.stages.pose_qa score --panos extracted_cubemaps \
        --renders qa_renders --poses poses.json -o qa_output [--seg-dir seg_out]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vs_pipeline.coordinates import convert_pose_yup_to_zup
from vs_pipeline.stages.pose import SweepPose, read_poses_json

QA_SCHEMA_VERSION = 5
#: Gross-failure floor for the chance-corrected sweep score: a misregistered
#: sweep renders wrong geometry, so its normalized score collapses toward 0.
DEFAULT_THRESHOLD = 0.20
SPOT_CHECK_RATE = 0.1
DEFAULT_RENDER_SIZE = 512
DILATE_PX = 5
#: Rendered faces with fewer edge pixels than this carry no alignment signal
#: (camera facing sky / outside the mesh / through a window) and are excluded.
MIN_RENDER_EDGE_PIXELS = 200
#: Minimum informative faces required to score a sweep at all.
MIN_INFORMATIVE_FACES = 2
#: Pano faces whose dilated edge mask covers more than this fraction of the
#: image are saturated (chance overlap ~= 1): no measurable alignment signal.
PANO_MAX_DENSITY = 0.97
#: Subtle-drift signal: mean per-face margin (locked orientation's normalized
#: overlap minus the best wrong orientation's) below this value flags the
#: sweep. Correct pose -> clear positive margin; drift -> margin collapses.
MARGIN_THRESHOLD = 0.03

#: G6 slice 4.2 -- Gate-1 glass/mirror exclusion. 7-class id for 'window'
#: (index into vs_pipeline.stages.segmentation.CLASSES; cross-checked by a
#: unit test instead of an import, keeping pose-QA decoupled from the
#: segmentation module) and the raw ADE20K label id for 'mirror;looking
#: glass' (the 7-class map deliberately sends mirror to 'other', so mirrors
#: are only reachable through the RAW map). Known limitation (OPEN_PROBLEMS
#: #2): REAL mirror surfaces are usually labeled 'wall' by Mask2Former, so
#: mirror exclusion via the raw map is best-effort.
SEG_WINDOW_CLASS_ID = 4
ADE_MIRROR_LABEL_ID = 27

FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")

#: Per-face view directions in the scanner-local Matterport frame (Z-up).
_FACE_DIRS: dict[str, tuple[float, float, float]] = {
    "px": (1.0, 0.0, 0.0),
    "nx": (-1.0, 0.0, 0.0),
    "py": (0.0, 1.0, 0.0),
    "ny": (0.0, -1.0, 0.0),
    "pz": (0.0, 0.0, 1.0),
    "nz": (0.0, 0.0, -1.0),
}

#: Approximate up vectors (in-plane roll is resolved by auto-calibration).
_FACE_UPS: dict[str, tuple[float, float, float]] = {
    "px": (0.0, 0.0, 1.0),
    "nx": (0.0, 0.0, 1.0),
    "py": (0.0, 0.0, 1.0),
    "ny": (0.0, 0.0, 1.0),
    "pz": (0.0, -1.0, 0.0),
    "nz": (0.0, 1.0, 0.0),
}

ORIENTATION_OPS = (
    "rot0",
    "rot90",
    "rot180",
    "rot270",
    "rot0_flip",
    "rot90_flip",
    "rot180_flip",
    "rot270_flip",
)

_ROT_K = {"rot0": 0, "rot90": 1, "rot180": 2, "rot270": 3}


@dataclass
class SweepQAResult:
    sweep_uuid: str
    index: int
    score: float | None
    face_scores: dict[str, float]
    raw_face_scores: dict[str, float]
    margin: float | None
    face_margins: dict[str, float]
    flag_reasons: list[str]
    flagged: bool
    spot_check: bool
    uninformative_faces: list[str]
    #: Per-face fraction of pixels excluded as glass/mirror (seg-dir runs only).
    face_excluded: dict[str, float] = field(default_factory=dict)


def apply_orientation(image: np.ndarray, op: str) -> np.ndarray:
    """Apply one of the 8 lossless orientation ops (rotNN [+ _flip])."""
    base, _, flip = op.partition("_")
    if base not in _ROT_K or (flip and flip != "flip"):
        raise ValueError(f"unknown orientation op: {op}")
    result = np.rot90(image, k=_ROT_K[base])
    if flip:
        result = np.fliplr(result)
    return result


def face_camera_matrix(pose_mp: np.ndarray, face: str) -> np.ndarray:
    """Camera-to-world 4x4 for one cube face (Blender: X=right, Y=up, -Z=view)."""
    rotation = pose_mp[:3, :3]
    position = pose_mp[:3, 3]
    forward = rotation @ np.asarray(_FACE_DIRS[face], dtype=np.float64)
    up_hint = rotation @ np.asarray(_FACE_UPS[face], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    x_axis = np.cross(forward, up_hint)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = -forward
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = x_axis
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = z_axis
    matrix[:3, 3] = position
    return matrix


def _perturb_pose(pose_mp: np.ndarray, offset_cm: float, yaw_deg: float) -> np.ndarray:
    """Deliberately corrupt a pose (sensitivity self-test for the QA metric)."""
    perturbed = pose_mp.copy()
    yaw = np.radians(yaw_deg)
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    perturbed[:3, :3] = rot_z @ perturbed[:3, :3]
    perturbed[0, 3] += offset_cm / 100.0
    return perturbed


def prepare_render_jobs(
    poses_path: Path,
    output_path: Path,
    render_size: int = DEFAULT_RENDER_SIZE,
    perturb_index: int | None = None,
    perturb_cm: float = 0.0,
    perturb_deg: float = 0.0,
) -> int:
    """Emit render_jobs.json for the Blender renderer. Returns sweep count."""
    sweeps = read_poses_json(poses_path)
    sweep_entries = []
    for s in sweeps:
        pose_mp = convert_pose_yup_to_zup(s.pose_rt)
        if perturb_index is not None and s.index == perturb_index:
            pose_mp = _perturb_pose(pose_mp, perturb_cm, perturb_deg)
            print(
                f"PERTURBED sweep [{s.index:03d}] {s.sweep_uuid}: "
                f"+{perturb_cm} cm, +{perturb_deg} deg yaw (sensitivity test)"
            )
        sweep_entries.append(
            {
                "sweep_uuid": s.sweep_uuid,
                "index": s.index,
                "faces": {
                    face: face_camera_matrix(pose_mp, face).tolist() for face in FACE_NAMES
                },
            }
        )
    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "render_size": render_size,
        "sweeps": sweep_entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(sweeps)


def is_spot_check(sweep_uuid: str, rate: float = SPOT_CHECK_RATE) -> bool:
    """Deterministic ~rate selection (stable across re-runs)."""
    bucket = int(hashlib.sha256(sweep_uuid.encode("utf-8")).hexdigest()[:4], 16)
    return bucket < int(65536 * rate)


def _load_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    return image


def _pano_edges_dilated(pano_gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if pano_gray.shape != size:
        pano_gray = cv2.resize(pano_gray, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(pano_gray, 50, 150)
    kernel = np.ones((DILATE_PX, DILATE_PX), np.uint8)
    return cv2.dilate(edges, kernel)


def _render_edges(render_gray: np.ndarray, op: str) -> np.ndarray:
    oriented = np.ascontiguousarray(apply_orientation(render_gray, op))
    return cv2.Canny(oriented, 20, 60)


def _overlap(pano_dilated: np.ndarray, render_edges: np.ndarray) -> float:
    total = int(np.count_nonzero(render_edges))
    if total == 0:
        return 0.0
    hits = int(np.count_nonzero((render_edges > 0) & (pano_dilated > 0)))
    return hits / total


def _mask_density(mask: np.ndarray, valid_px: int | None = None) -> float:
    """Fraction of (valid) pixels covered by the mask.

    ``valid_px`` restricts the denominator to non-excluded pixels when a
    glass/mirror exclusion is active (the mask itself must already be zeroed
    inside the excluded region). A zero/negative valid area reads as fully
    saturated (density 1.0 -> no measurable signal).
    """
    total = float(valid_px) if valid_px is not None else float(mask.size)
    if total <= 0:
        return 1.0
    return float(np.count_nonzero(mask)) / total


def _normalized_overlap(
    pano_dilated: np.ndarray,
    render_edges: np.ndarray,
    valid_px: int | None = None,
) -> float:
    """Chance-corrected overlap: (raw - density) / (1 - density).

    ``density`` is the fraction of the image covered by the dilated pano edge
    mask -- the overlap a randomly placed edge set would achieve by chance.
    The normalized score measures how much better than chance the alignment
    is, making texture-dense and texture-sparse faces comparable. A perfect
    alignment still scores 1.0; a chance-level alignment scores ~0.0; worse
    than chance is negative. With an active glass/mirror exclusion both edge
    maps are zeroed inside the excluded region and ``valid_px`` carries the
    non-excluded pixel count, so chance is computed over scoreable pixels.
    """
    if int(np.count_nonzero(render_edges)) == 0:
        return 0.0  # no rendered edges -> no signal, not worse-than-chance
    density = _mask_density(pano_dilated, valid_px)
    if density >= 1.0:
        return 0.0
    raw = _overlap(pano_dilated, render_edges)
    return (raw - density) / (1.0 - density)


def load_exclusion_mask(
    seg_dir: Path, sweep_uuid: str, ext_face: str, shape: tuple[int, int]
) -> np.ndarray | None:
    """Gate-1 glass/mirror exclusion mask for one extracted face (or None).

    Reads the G6 segmentation outputs, which are stored-layout and therefore
    pixel-aligned with the extracted pano faces (the upright fix's
    guarantee): window pixels from the 7-class map (SAM-refined variant
    preferred) UNION mirror pixels from the RAW ADE map (the 7-class map
    deliberately sends mirror to 'other'). Both regions carry photographic
    content the mesh render cannot match (outdoor scenery through glass,
    reflected room content), so they are removed from BOTH edge maps before
    scoring. Returns a bool mask resized to ``shape`` (NEAREST -- label
    images), or None when no segmentation exists for this face.
    """
    from vs_pipeline.stages.segmentation import resolve_class_map

    sweep_dir = seg_dir / sweep_uuid
    cls, _src = resolve_class_map(seg_dir, sweep_uuid, ext_face)
    if cls is None:
        return None
    mask = cls == SEG_WINDOW_CLASS_ID
    ade = cv2.imread(str(sweep_dir / f"{ext_face}_ade.png"), cv2.IMREAD_GRAYSCALE)
    if ade is not None and ade.shape == cls.shape:
        mask |= ade == ADE_MIRROR_LABEL_ID
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.shape != tuple(shape):
        mask_u8 = cv2.resize(
            mask_u8, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST
        )
    return mask_u8.astype(bool)


def edge_overlap_score(
    pano_gray: np.ndarray,
    render_gray: np.ndarray,
    orientation: str = "rot0",
    dilate_px: int = DILATE_PX,  # noqa: ARG001 (kept for API stability)
) -> float:
    """Chance-corrected fraction of rendered edges landing on (dilated) pano edges."""
    render_edges = _render_edges(render_gray, orientation)
    pano_dilated = _pano_edges_dilated(pano_gray, render_edges.shape)
    return _normalized_overlap(pano_dilated, render_edges)


def _bijective_assignment(
    combo_scores: dict[tuple[str, str, str], float],
) -> dict[str, dict[str, object]]:
    """Greedy best-score bijection: extracted face -> (rendered face, op)."""
    mapping: dict[str, dict[str, object]] = {}
    used_render: set[str] = set()
    for (ext_face, render_face, op), score in sorted(
        combo_scores.items(), key=lambda item: item[1], reverse=True
    ):
        if ext_face in mapping or render_face in used_render:
            continue
        mapping[ext_face] = {"render_face": render_face, "op": op, "mean": round(score, 4)}
        used_render.add(render_face)
        if len(mapping) == len(FACE_NAMES):
            break
    return mapping


def calibrate_face_mapping(
    pano_dir: Path,
    render_dir: Path,
    sweeps: list[SweepPose],
    sample_count: int = 5,
) -> dict[str, dict[str, object]]:
    """Search all 6 rendered faces x 8 orientations per extracted face."""
    step = max(1, len(sweeps) // sample_count)
    samples = sweeps[::step][:sample_count]

    # Cache rendered edge maps per (render face, sample, op).
    render_edge_cache: dict[tuple[str, str, str], np.ndarray] = {}
    render_shape: tuple[int, int] | None = None
    for render_face in FACE_NAMES:
        for sweep in samples:
            gray = _load_gray(render_dir / sweep.sweep_uuid / f"{render_face}.png")
            if gray is None:
                continue
            for op in ORIENTATION_OPS:
                edges = _render_edges(gray, op)
                render_edge_cache[(render_face, sweep.sweep_uuid, op)] = edges
                render_shape = edges.shape
    if render_shape is None:
        raise ValueError(f"no rendered faces found under {render_dir}")

    # Cache dilated pano edges per (extracted face, sample).
    pano_cache: dict[tuple[str, str], np.ndarray] = {}
    for ext_face in FACE_NAMES:
        for sweep in samples:
            gray = _load_gray(pano_dir / sweep.sweep_uuid / f"{ext_face}.jpg")
            if gray is None:
                continue
            pano_cache[(ext_face, sweep.sweep_uuid)] = _pano_edges_dilated(gray, render_shape)

    combo_scores: dict[tuple[str, str, str], float] = {}
    for ext_face in FACE_NAMES:
        for render_face in FACE_NAMES:
            for op in ORIENTATION_OPS:
                values: list[float] = []
                for sweep in samples:
                    pano = pano_cache.get((ext_face, sweep.sweep_uuid))
                    render = render_edge_cache.get((render_face, sweep.sweep_uuid, op))
                    if pano is None or render is None:
                        continue
                    if np.count_nonzero(render) < MIN_RENDER_EDGE_PIXELS:
                        continue  # no alignment signal in this render
                    if _mask_density(pano) > PANO_MAX_DENSITY:
                        continue  # saturated pano: chance overlap ~= 1
                    values.append(_normalized_overlap(pano, render))
                if values:
                    combo_scores[(ext_face, render_face, op)] = float(np.mean(values))

    mapping = _bijective_assignment(combo_scores)
    for ext_face in FACE_NAMES:
        entry = mapping.get(ext_face)
        if entry:
            print(
                f"  calibration {ext_face} <- render {entry['render_face']} "
                f"({entry['op']}), mean overlap {entry['mean']}"
            )
        else:
            print(f"  calibration {ext_face}: NO MATCH (missing data)")
    return mapping


def _write_composite_grid(
    faces: list[tuple[str, float, np.ndarray, np.ndarray, str]],
    out_path: Path,
    exclusions: dict[str, np.ndarray] | None = None,
) -> None:
    """Stacked review image, one row per face:
    pano | rendered OBJ view (oriented normalized depth) | pano + red render edges.

    Excluded glass/mirror regions (seg-dir runs) are tinted yellow in the
    overlay column: those pixels did not participate in the score.
    """
    rows: list[np.ndarray] = []
    for face_name, score, pano_gray, render_gray, orientation in faces:
        render_edges = _render_edges(render_gray, orientation)
        pano_small = cv2.resize(
            pano_gray,
            (render_edges.shape[1], render_edges.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        color = cv2.cvtColor(pano_small, cv2.COLOR_GRAY2BGR)
        oriented_render = np.ascontiguousarray(apply_orientation(render_gray, orientation))
        render_view = cv2.cvtColor(oriented_render, cv2.COLOR_GRAY2BGR)
        overlay = color.copy()
        excl = exclusions.get(face_name) if exclusions else None
        if excl is not None and excl.shape == overlay.shape[:2]:
            tinted = overlay[excl].astype(np.float32)
            overlay[excl] = (
                0.55 * tinted + 0.45 * np.array([0.0, 255.0, 255.0], dtype=np.float32)
            ).astype(np.uint8)
            render_edges = render_edges.copy()
            render_edges[excl] = 0
        overlay[render_edges > 0] = (0, 0, 255)
        row = np.hstack([color, render_view, overlay])
        cv2.putText(
            row,
            f"{face_name}  {score:.2f}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        rows.append(row)
    if not rows:
        return
    composite = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def score_property(
    pano_dir: Path,
    render_dir: Path,
    poses_path: Path,
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    calibration_samples: int = 5,
    calibration_path: Path | None = None,
    seg_dir: Path | None = None,
) -> list[SweepQAResult]:
    """Score every rendered sweep; write pose_qa.json + review composites."""
    sweeps = read_poses_json(poses_path)
    rendered = [s for s in sweeps if (render_dir / s.sweep_uuid).is_dir()]
    if not rendered:
        raise ValueError(f"no rendered sweeps found under {render_dir}")

    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        print(f"loaded locked calibration from {calibration_path}")
        for ext_face in FACE_NAMES:
            entry = calibration.get(ext_face)
            if entry:
                print(
                    f"  {ext_face} <- render {entry['render_face']} ({entry['op']}), "
                    f"mean {entry.get('mean')}"
                )
    else:
        print(
            f"auto-calibrating face mapping on {min(calibration_samples, len(rendered))} "
            "sample sweep(s):"
        )
        calibration = calibrate_face_mapping(pano_dir, render_dir, rendered, calibration_samples)
        if calibration_path is not None:
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(
                json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
            )
            print(f"calibration locked -> {calibration_path}")

    if seg_dir is not None:
        print(f"glass/mirror exclusion active (seg masks: {seg_dir})")

    results: list[SweepQAResult] = []
    composites_dir = output_dir / "composites"
    for sweep in rendered:
        face_scores: dict[str, float] = {}
        raw_face_scores: dict[str, float] = {}
        face_margins: dict[str, float] = {}
        face_excluded: dict[str, float] = {}
        face_exclusions: dict[str, np.ndarray] = {}
        uninformative: list[str] = []
        faces_gray: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        for ext_face in FACE_NAMES:
            entry = calibration.get(ext_face)
            if not entry:
                continue
            render_face = str(entry["render_face"])
            op = str(entry["op"])
            pano = _load_gray(pano_dir / sweep.sweep_uuid / f"{ext_face}.jpg")
            render = _load_gray(render_dir / sweep.sweep_uuid / f"{render_face}.png")
            if pano is None or render is None:
                continue
            render_edges = _render_edges(render, op)
            exclusion: np.ndarray | None = None
            valid_px: int | None = None
            if seg_dir is not None:
                exclusion = load_exclusion_mask(
                    seg_dir, sweep.sweep_uuid, ext_face, render_edges.shape
                )
                if exclusion is not None and not np.any(exclusion):
                    exclusion = None  # mask exists but excludes nothing
            if exclusion is not None:
                excluded_count = int(np.count_nonzero(exclusion))
                face_excluded[ext_face] = round(
                    excluded_count / float(exclusion.size), 4
                )
                face_exclusions[ext_face] = exclusion
                valid_px = int(exclusion.size) - excluded_count
                render_edges[exclusion] = 0  # Canny output: fresh array, safe to mutate
            if int(np.count_nonzero(render_edges)) < MIN_RENDER_EDGE_PIXELS:
                uninformative.append(ext_face)  # no signal: exclude, don't punish
                continue
            pano_dilated = _pano_edges_dilated(pano, render_edges.shape)
            if exclusion is not None:
                pano_dilated[exclusion] = 0  # dilate output: fresh array
            if _mask_density(pano_dilated, valid_px) > PANO_MAX_DENSITY:
                uninformative.append(ext_face)  # saturated pano: no measurable signal
                continue
            locked_norm = _normalized_overlap(pano_dilated, render_edges, valid_px)
            face_scores[ext_face] = locked_norm
            raw_face_scores[ext_face] = _overlap(pano_dilated, render_edges)
            other_norms: list[float] = []
            for other_op in ORIENTATION_OPS:
                if other_op == op:
                    continue
                other_edges = _render_edges(render, other_op)
                if exclusion is not None:
                    other_edges[exclusion] = 0
                other_norms.append(
                    _normalized_overlap(pano_dilated, other_edges, valid_px)
                )
            face_margins[ext_face] = locked_norm - max(other_norms)
            faces_gray[ext_face] = (pano, render, op)
        informative = len(face_scores)
        score = (
            float(np.mean(list(face_scores.values())))
            if informative >= MIN_INFORMATIVE_FACES
            else None
        )
        margin = (
            float(np.mean(list(face_margins.values())))
            if informative >= MIN_INFORMATIVE_FACES
            else None
        )
        flag_reasons: list[str] = []
        if score is None:
            flag_reasons.append("too_few_informative_faces")
        else:
            if score < threshold:
                flag_reasons.append("low_score")
            if margin is not None and margin < MARGIN_THRESHOLD:
                flag_reasons.append("low_margin")
        flagged = bool(flag_reasons)
        spot = is_spot_check(sweep.sweep_uuid)
        results.append(
            SweepQAResult(
                sweep_uuid=sweep.sweep_uuid,
                index=sweep.index,
                score=score,
                face_scores=face_scores,
                raw_face_scores=raw_face_scores,
                margin=margin,
                face_margins=face_margins,
                flag_reasons=flag_reasons,
                flagged=flagged,
                spot_check=spot,
                uninformative_faces=uninformative,
                face_excluded=face_excluded,
            )
        )
        if (flagged or spot) and face_scores:
            grid = [
                (face, face_scores[face], *faces_gray[face])
                for face in FACE_NAMES
                if face in face_scores
            ]
            _write_composite_grid(
                grid,
                composites_dir / f"{sweep.index:03d}_{sweep.sweep_uuid}.jpg",
                exclusions=face_exclusions or None,
            )

    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "threshold": threshold,
        "margin_threshold": MARGIN_THRESHOLD,
        "spot_check_rate": SPOT_CHECK_RATE,
        "calibration": calibration,
        "seg_dir": str(seg_dir) if seg_dir is not None else None,
        "seg_excluded_sweep_count": sum(1 for r in results if r.face_excluded),
        "sweep_count": len(results),
        "flagged_count": sum(1 for r in results if r.flagged),
        "spot_check_count": sum(1 for r in results if r.spot_check),
        "sweeps": [
            {
                "sweep_uuid": r.sweep_uuid,
                "index": r.index,
                "score": r.score,
                "face_scores": r.face_scores,
                "raw_face_scores": r.raw_face_scores,
                "margin": r.margin,
                "face_margins": r.face_margins,
                "flag_reasons": r.flag_reasons,
                "uninformative_faces": r.uninformative_faces,
                "face_excluded": r.face_excluded,
                "flagged": r.flagged,
                "spot_check": r.spot_check,
            }
            for r in results
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pose_qa.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return results


def verify_property(
    pano_dir: Path,
    render_dir: Path,
    poses_path: Path,
    calibration_path: Path,
    output_dir: Path,
    limit: int = 0,
) -> None:
    """Ghosting verification: [pano | 50/50 blend | abs-diff heatmap] per face.

    Uses TEXTURED renders (render_edges.py --mode textured). A correct pose
    makes the blend look sharp; pose error appears as double-vision ghosting
    proportional to the error. Prints normalized cross-correlation per sweep.
    """
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    sweeps = read_poses_json(poses_path)
    rendered = [s for s in sweeps if (render_dir / s.sweep_uuid).is_dir()]
    if limit > 0:
        rendered = rendered[:limit]
    if not rendered:
        raise ValueError(f"no rendered sweeps found under {render_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for sweep in rendered:
        rows: list[np.ndarray] = []
        nccs: list[float] = []
        for face in FACE_NAMES:
            entry = calibration.get(face)
            if not entry:
                continue
            pano = cv2.imread(
                str(pano_dir / sweep.sweep_uuid / f"{face}.jpg"), cv2.IMREAD_COLOR
            )
            render = cv2.imread(
                str(render_dir / sweep.sweep_uuid / f"{entry['render_face']}.png"),
                cv2.IMREAD_COLOR,
            )
            if pano is None or render is None:
                continue
            oriented = np.ascontiguousarray(apply_orientation(render, str(entry["op"])))
            if float(oriented.std()) < 2.0:
                continue  # empty view (sky / outside the mesh)
            pano_small = cv2.resize(
                pano, (oriented.shape[1], oriented.shape[0]), interpolation=cv2.INTER_AREA
            )
            gray_pano = cv2.cvtColor(pano_small, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gray_render = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY).astype(np.float64)
            corr = np.corrcoef(gray_pano.ravel(), gray_render.ravel())[0, 1]
            ncc = float(corr) if np.isfinite(corr) else 0.0
            nccs.append(ncc)

            blend = cv2.addWeighted(pano_small, 0.5, oriented, 0.5, 0)
            diff = cv2.applyColorMap(
                cv2.absdiff(
                    gray_pano.astype(np.uint8), gray_render.astype(np.uint8)
                ),
                cv2.COLORMAP_JET,
            )
            row = np.hstack([pano_small, blend, diff])
            cv2.putText(
                row,
                f"{face}  ncc={ncc:.2f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            rows.append(row)
        if rows:
            out_path = output_dir / f"{sweep.index:03d}_{sweep.sweep_uuid}.jpg"
            cv2.imwrite(
                str(out_path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            print(
                f"[{sweep.index:03d}] {sweep.sweep_uuid}  "
                f"mean ncc = {float(np.mean(nccs)):.3f}  -> {out_path.name}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 1: automated pose-QA scoring.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="emit render_jobs.json for Blender")
    p_prepare.add_argument("poses", type=Path, help="poses.json from the pose stage")
    p_prepare.add_argument("--output", "-o", type=Path, default=Path("render_jobs.json"))
    p_prepare.add_argument("--size", type=int, default=DEFAULT_RENDER_SIZE)
    p_prepare.add_argument("--perturb-index", type=int, default=None, help="sweep index to corrupt (sensitivity test)")
    p_prepare.add_argument("--perturb-cm", type=float, default=5.0)
    p_prepare.add_argument("--perturb-deg", type=float, default=2.0)

    p_verify = sub.add_parser("verify", help="ghosting verification against textured renders")
    p_verify.add_argument("--panos", type=Path, required=True)
    p_verify.add_argument("--renders", type=Path, required=True, help="textured renders dir")
    p_verify.add_argument("--poses", type=Path, required=True)
    p_verify.add_argument("--calibration", type=Path, required=True)
    p_verify.add_argument("--output", "-o", type=Path, default=Path("qa_verify"))
    p_verify.add_argument("--limit", type=int, default=0)

    p_score = sub.add_parser("score", help="score rendered faces against panoramas")
    p_score.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p_score.add_argument("--renders", type=Path, required=True, help="Blender output dir")
    p_score.add_argument("--poses", type=Path, required=True, help="poses.json")
    p_score.add_argument("--output", "-o", type=Path, default=Path("qa_output"))
    p_score.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_score.add_argument("--calibration-samples", type=int, default=5)
    p_score.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration file (default: {output}/calibration.json); loaded if present, "
        "otherwise computed and persisted",
    )
    p_score.add_argument(
        "--seg-dir",
        type=Path,
        default=None,
        help="G6 segmentation output dir (seg_out): exclude window pixels "
        "(7-class masks, SAM-refined preferred) + raw-ADE mirror pixels from "
        "scoring; sweeps without masks score exactly as before",
    )

    args = parser.parse_args(argv)

    if args.command == "prepare":
        count = prepare_render_jobs(
            args.poses,
            args.output,
            args.size,
            perturb_index=args.perturb_index,
            perturb_cm=args.perturb_cm,
            perturb_deg=args.perturb_deg,
        )
        print(f"{count} sweep(s) x {len(FACE_NAMES)} faces -> {args.output}")
        return 0

    if args.command == "verify":
        verify_property(
            args.panos,
            args.renders,
            args.poses,
            args.calibration,
            args.output,
            limit=args.limit,
        )
        return 0

    results = score_property(
        args.panos,
        args.renders,
        args.poses,
        args.output,
        threshold=args.threshold,
        calibration_samples=args.calibration_samples,
        calibration_path=args.calibration or (args.output / "calibration.json"),
        seg_dir=args.seg_dir,
    )
    scored = [r for r in results if r.score is not None]
    flagged = [r for r in results if r.flagged]
    spot = [r for r in results if r.spot_check]
    if scored:
        mean = float(np.mean([r.score for r in scored if r.score is not None]))
        print(f"\n{len(results)} sweep(s) scored, mean edge-overlap {mean:.3f}")
        margins = [r.margin for r in scored if r.margin is not None]
        if margins:
            print(f"mean orientation margin {float(np.mean(margins)):.3f}")
    if args.seg_dir is not None:
        masked = sum(1 for r in results if r.face_excluded)
        print(
            f"glass/mirror exclusion: {masked}/{len(results)} sweep(s) had "
            f"masked pixels (seg: {args.seg_dir})"
        )
    print(
        f"flagged (score < {args.threshold} or margin < {MARGIN_THRESHOLD}): "
        f"{len(flagged)}   spot-check: {len(spot)}"
    )
    for r in sorted(flagged, key=lambda r: (r.score is not None, r.score)):
        worst = sorted(r.face_scores.items(), key=lambda kv: kv[1])[:2]
        detail = ", ".join(f"{face}={value:.2f}" for face, value in worst)
        extra = f"  (worst: {detail})" if detail else ""
        if r.uninformative_faces:
            extra += f"  [no-signal: {','.join(r.uninformative_faces)}]"
        if r.face_excluded:
            top = max(r.face_excluded.values())
            extra += f"  [excl max {top:.0%}]"
        score_str = f"{r.score:.3f}" if r.score is not None else "n/a"
        margin_str = f"{r.margin:.3f}" if r.margin is not None else "n/a"
        print(
            f"  FLAG [{r.index:03d}] {r.sweep_uuid}  "
            f"score = {score_str}  margin = {margin_str}  "
            f"[{','.join(r.flag_reasons)}]"
            + extra
        )
    print(f"report: {args.output / 'pose_qa.json'}   composites: {args.output / 'composites'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
