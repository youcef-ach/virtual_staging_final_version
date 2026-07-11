"""Stage 2 (part 3) — automated pose-QA scoring (Gate 1).

Pipeline:
1. ``prepare``: read poses.json (runtime Y-up), convert back to the Matterport
   frame with the exact inverse of the ingestion conversion, and emit
   render_jobs.json — one 90-degree pinhole camera per cube face per sweep
   (camera-to-world matrices for Blender: X=right, Y=up, -Z=view).
2. Blender renders normalized depth per face (pipeline/blender/render_edges.py).
3. ``score``: Canny edges of the rendered depth vs dilated Canny edges of the
   extracted panorama face -> edge-overlap score in [0, 1] per face; sweep
   score = mean over faces. score < 0.85 -> flagged for operator review.
   A deterministic ~10% of sweeps is always marked for spot-check.

Face-mapping auto-calibration: extracted faces carry the canonical Set A
transforms (including the +Z/-Z content swap) plus the viewer-level mirror,
so extracted face N does not necessarily correspond to rendered face N.
Calibration searches ALL 6 rendered faces x 8 in-plane orientations per
extracted face on sample sweeps and locks a bijective assignment — the
overlap metric itself is the calibration signal.

CLI:
    python -m vs_pipeline.stages.pose_qa prepare poses.json -o render_jobs.json
    python -m vs_pipeline.stages.pose_qa score --panos extracted_cubemaps \
        --renders qa_renders --poses poses.json -o qa_output
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vs_pipeline.coordinates import convert_pose_yup_to_zup
from vs_pipeline.stages.pose import SweepPose, read_poses_json

QA_SCHEMA_VERSION = 2
DEFAULT_THRESHOLD = 0.85
SPOT_CHECK_RATE = 0.1
DEFAULT_RENDER_SIZE = 512
DILATE_PX = 5
#: Rendered faces with fewer edge pixels than this carry no alignment signal
#: (camera facing sky / outside the mesh / through a window) and are excluded.
MIN_RENDER_EDGE_PIXELS = 200
#: Minimum informative faces required to score a sweep at all.
MIN_INFORMATIVE_FACES = 2

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
    flagged: bool
    spot_check: bool
    uninformative_faces: list[str]


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


def edge_overlap_score(
    pano_gray: np.ndarray,
    render_gray: np.ndarray,
    orientation: str = "rot0",
    dilate_px: int = DILATE_PX,  # noqa: ARG001 (kept for API stability)
) -> float:
    """Fraction of rendered geometry edges that land on (dilated) pano edges."""
    render_edges = _render_edges(render_gray, orientation)
    pano_dilated = _pano_edges_dilated(pano_gray, render_edges.shape)
    return _overlap(pano_dilated, render_edges)


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
                    values.append(_overlap(pano, render))
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
    faces: list[tuple[str, float, np.ndarray, np.ndarray, str]], out_path: Path
) -> None:
    """Stacked review image: one row per face = pano | pano with red render edges."""
    rows: list[np.ndarray] = []
    for face_name, score, pano_gray, render_gray, orientation in faces:
        render_edges = _render_edges(render_gray, orientation)
        pano_small = cv2.resize(
            pano_gray,
            (render_edges.shape[1], render_edges.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        color = cv2.cvtColor(pano_small, cv2.COLOR_GRAY2BGR)
        overlay = color.copy()
        overlay[render_edges > 0] = (0, 0, 255)
        row = np.hstack([color, overlay])
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

    results: list[SweepQAResult] = []
    composites_dir = output_dir / "composites"
    for sweep in rendered:
        face_scores: dict[str, float] = {}
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
            if int(np.count_nonzero(render_edges)) < MIN_RENDER_EDGE_PIXELS:
                uninformative.append(ext_face)  # no signal: exclude, don't punish
                continue
            pano_dilated = _pano_edges_dilated(pano, render_edges.shape)
            face_scores[ext_face] = _overlap(pano_dilated, render_edges)
            faces_gray[ext_face] = (pano, render, op)
        informative = len(face_scores)
        score = (
            float(np.mean(list(face_scores.values())))
            if informative >= MIN_INFORMATIVE_FACES
            else None
        )
        flagged = score is None or score < threshold
        spot = is_spot_check(sweep.sweep_uuid)
        results.append(
            SweepQAResult(
                sweep_uuid=sweep.sweep_uuid,
                index=sweep.index,
                score=score,
                face_scores=face_scores,
                flagged=flagged,
                spot_check=spot,
                uninformative_faces=uninformative,
            )
        )
        if (flagged or spot) and face_scores:
            grid = [
                (face, face_scores[face], *faces_gray[face])
                for face in FACE_NAMES
                if face in face_scores
            ]
            _write_composite_grid(
                grid, composites_dir / f"{sweep.index:03d}_{sweep.sweep_uuid}.jpg"
            )

    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "threshold": threshold,
        "spot_check_rate": SPOT_CHECK_RATE,
        "calibration": calibration,
        "sweep_count": len(results),
        "flagged_count": sum(1 for r in results if r.flagged),
        "spot_check_count": sum(1 for r in results if r.spot_check),
        "sweeps": [
            {
                "sweep_uuid": r.sweep_uuid,
                "index": r.index,
                "score": r.score,
                "face_scores": r.face_scores,
                "uninformative_faces": r.uninformative_faces,
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
    )
    scored = [r for r in results if r.score is not None]
    flagged = [r for r in results if r.flagged]
    spot = [r for r in results if r.spot_check]
    if scored:
        mean = float(np.mean([r.score for r in scored if r.score is not None]))
        print(f"\n{len(results)} sweep(s) scored, mean edge-overlap {mean:.3f}")
    print(f"flagged (< {args.threshold}): {len(flagged)}   spot-check: {len(spot)}")
    for r in sorted(flagged, key=lambda r: (r.score is not None, r.score)):
        worst = sorted(r.face_scores.items(), key=lambda kv: kv[1])[:2]
        detail = ", ".join(f"{face}={value:.2f}" for face, value in worst)
        extra = f"  (worst: {detail})" if detail else ""
        if r.uninformative_faces:
            extra += f"  [no-signal: {','.join(r.uninformative_faces)}]"
        print(
            f"  FLAG [{r.index:03d}] {r.sweep_uuid}  "
            f"score = {r.score if r.score is not None else 'n/a'}"
            + extra
        )
    print(f"report: {args.output / 'pose_qa.json'}   composites: {args.output / 'composites'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
