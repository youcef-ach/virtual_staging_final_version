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

Render-orientation auto-calibration: the in-plane orientation of rendered
faces vs extracted faces is unknown a priori (like F4.4). The scorer tries 8
orientations (4 rotations x flip) per face on sample sweeps and locks the
argmax — the overlap metric itself is the calibration signal.

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

QA_SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.85
SPOT_CHECK_RATE = 0.1
DEFAULT_RENDER_SIZE = 512

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


def prepare_render_jobs(
    poses_path: Path, output_path: Path, render_size: int = DEFAULT_RENDER_SIZE
) -> int:
    """Emit render_jobs.json for the Blender renderer. Returns sweep count."""
    sweeps = read_poses_json(poses_path)
    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "render_size": render_size,
        "sweeps": [
            {
                "sweep_uuid": s.sweep_uuid,
                "index": s.index,
                "faces": {
                    face: face_camera_matrix(convert_pose_yup_to_zup(s.pose_rt), face).tolist()
                    for face in FACE_NAMES
                },
            }
            for s in sweeps
        ],
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


def edge_overlap_score(
    pano_gray: np.ndarray,
    render_gray: np.ndarray,
    orientation: str = "rot0",
    dilate_px: int = 5,
) -> float:
    """Fraction of rendered geometry edges that land on (dilated) pano edges."""
    oriented = np.ascontiguousarray(apply_orientation(render_gray, orientation))
    if pano_gray.shape != oriented.shape:
        pano_gray = cv2.resize(
            pano_gray, (oriented.shape[1], oriented.shape[0]), interpolation=cv2.INTER_AREA
        )
    pano_edges = cv2.Canny(pano_gray, 50, 150)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    pano_dilated = cv2.dilate(pano_edges, kernel)
    render_edges = cv2.Canny(oriented, 20, 60)
    total = int(np.count_nonzero(render_edges))
    if total == 0:
        return 0.0
    hits = int(np.count_nonzero((render_edges > 0) & (pano_dilated > 0)))
    return hits / total


def calibrate_orientations(
    pano_dir: Path,
    render_dir: Path,
    sweeps: list[SweepPose],
    sample_count: int = 5,
) -> dict[str, str]:
    """Pick the best in-plane orientation per face (argmax mean overlap)."""
    step = max(1, len(sweeps) // sample_count)
    samples = sweeps[::step][:sample_count]
    calibration: dict[str, str] = {}
    for face in FACE_NAMES:
        best_op, best_score = "rot0", -1.0
        for op in ORIENTATION_OPS:
            scores: list[float] = []
            for sweep in samples:
                pano = _load_gray(pano_dir / sweep.sweep_uuid / f"{face}.jpg")
                render = _load_gray(render_dir / sweep.sweep_uuid / f"{face}.png")
                if pano is None or render is None:
                    continue
                scores.append(edge_overlap_score(pano, render, op))
            if scores:
                mean = float(np.mean(scores))
                if mean > best_score:
                    best_op, best_score = op, mean
        calibration[face] = best_op
        print(f"  calibration {face}: {best_op} (mean overlap {best_score:.3f})")
    return calibration


def _write_composite(
    pano_gray: np.ndarray, render_gray: np.ndarray, orientation: str, out_path: Path
) -> None:
    """Side-by-side review image: pano | pano with rendered edges in red."""
    oriented = np.ascontiguousarray(apply_orientation(render_gray, orientation))
    pano_small = cv2.resize(
        pano_gray, (oriented.shape[1], oriented.shape[0]), interpolation=cv2.INTER_AREA
    )
    render_edges = cv2.Canny(oriented, 20, 60)
    color = cv2.cvtColor(pano_small, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[render_edges > 0] = (0, 0, 255)
    composite = np.hstack([color, overlay])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def score_property(
    pano_dir: Path,
    render_dir: Path,
    poses_path: Path,
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    calibration_samples: int = 5,
) -> list[SweepQAResult]:
    """Score every sweep; write pose_qa.json + review composites."""
    sweeps = read_poses_json(poses_path)
    rendered = [s for s in sweeps if (render_dir / s.sweep_uuid).is_dir()]
    if not rendered:
        raise ValueError(f"no rendered sweeps found under {render_dir}")

    print(f"auto-calibrating render orientation on {min(calibration_samples, len(rendered))} sample sweep(s):")
    calibration = calibrate_orientations(pano_dir, render_dir, rendered, calibration_samples)

    results: list[SweepQAResult] = []
    composites_dir = output_dir / "composites"
    for sweep in rendered:
        face_scores: dict[str, float] = {}
        faces_gray: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for face in FACE_NAMES:
            pano = _load_gray(pano_dir / sweep.sweep_uuid / f"{face}.jpg")
            render = _load_gray(render_dir / sweep.sweep_uuid / f"{face}.png")
            if pano is None or render is None:
                continue
            face_scores[face] = edge_overlap_score(pano, render, calibration[face])
            faces_gray[face] = (pano, render)
        score = float(np.mean(list(face_scores.values()))) if face_scores else None
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
            )
        )
        if (flagged or spot) and face_scores:
            worst_face = min(face_scores, key=lambda f: face_scores[f])
            pano, render = faces_gray[worst_face]
            _write_composite(
                pano,
                render,
                calibration[worst_face],
                composites_dir / f"{sweep.index:03d}_{sweep.sweep_uuid}_{worst_face}.jpg",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 1: automated pose-QA scoring.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="emit render_jobs.json for Blender")
    p_prepare.add_argument("poses", type=Path, help="poses.json from the pose stage")
    p_prepare.add_argument("--output", "-o", type=Path, default=Path("render_jobs.json"))
    p_prepare.add_argument("--size", type=int, default=DEFAULT_RENDER_SIZE)

    p_score = sub.add_parser("score", help="score rendered faces against panoramas")
    p_score.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p_score.add_argument("--renders", type=Path, required=True, help="Blender output dir")
    p_score.add_argument("--poses", type=Path, required=True, help="poses.json")
    p_score.add_argument("--output", "-o", type=Path, default=Path("qa_output"))
    p_score.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_score.add_argument("--calibration-samples", type=int, default=5)

    args = parser.parse_args(argv)

    if args.command == "prepare":
        count = prepare_render_jobs(args.poses, args.output, args.size)
        print(f"{count} sweep(s) x {len(FACE_NAMES)} faces -> {args.output}")
        return 0

    results = score_property(
        args.panos,
        args.renders,
        args.poses,
        args.output,
        threshold=args.threshold,
        calibration_samples=args.calibration_samples,
    )
    scored = [r for r in results if r.score is not None]
    flagged = [r for r in results if r.flagged]
    spot = [r for r in results if r.spot_check]
    if scored:
        mean = float(np.mean([r.score for r in scored if r.score is not None]))
        print(f"\n{len(results)} sweep(s) scored, mean edge-overlap {mean:.3f}")
    print(f"flagged (< {args.threshold}): {len(flagged)}   spot-check: {len(spot)}")
    for r in sorted(flagged, key=lambda r: (r.score is not None, r.score)):
        print(f"  FLAG [{r.index:03d}] {r.sweep_uuid}  score = {r.score if r.score is not None else 'n/a'}")
    print(f"report: {args.output / 'pose_qa.json'}   composites: {args.output / 'composites'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
