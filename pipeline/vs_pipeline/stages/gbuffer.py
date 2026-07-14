"""Stage 7 (G4) -- G-buffer baking: per-sweep pixel-aligned depth/normals.

Frame lock (why this stage reuses the F5 camera path unchanged): Blender's
glTF importer converts our runtime Y-up GLBs to Blender Z-up such that the
proxy lands EXACTLY in the Matterport Z-up frame (X_b = X_mp, Y_b = Y_mp,
Z_b = Z_mp). The per-face cameras from pose_qa (poses converted Y-up ->
Z-up) therefore line up with the F5 render convention pixel-for-pixel, and
the LOCKED F5 face-mapping calibration applies to G-buffers as-is.

Pipeline:
    prepare   poses.json -> gbuffer_jobs.json (6 x 90-deg cameras/sweep)
    (render)  blender -b -P pipeline/blender/render_gbuffer.py -- ...
              bakes 32-bit EXR Depth (ray length) + Normal (world space)
    verify    proxy depth edges vs extracted pano faces (F5 calibration
              mapping) -> composites + chance-corrected alignment scores

Deliberately deferred: KTX2 packing (toktx) joins Draco in the publish
stage; world position is DERIVED from depth + camera (depth_to_worldpos),
not stored -- one fewer texture, zero information loss.

CLI:
    python -m vs_pipeline.stages.gbuffer prepare poses.json -o gbuffer_jobs.json
    python -m vs_pipeline.stages.gbuffer verify --renders gbuffer_renders \
        --panos extracted_cubemaps --calibration qa_output/calibration.json \
        -o gbuffer_review
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

GBUFFER_SCHEMA_VERSION = 1
DEFAULT_SIZE = 2048


def face_ray_directions(size: int) -> np.ndarray:
    """Unit ray directions in CAMERA space for a 90-deg square pinhole face.

    Camera convention (Blender): X = right, Y = up, -Z = view direction.
    Pixel (0, 0) is the top-left corner; rays pass through pixel centers.
    """
    xs = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    ys = 1.0 - (np.arange(size) + 0.5) / size * 2.0
    gx, gy = np.meshgrid(xs, ys)
    dirs = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def depth_to_worldpos(depth: np.ndarray, cam_matrix: np.ndarray) -> np.ndarray:
    """World positions from a Cycles Depth pass (RAY LENGTH) + camera-to-world.

    Positions are in the render frame (Blender Z-up == Matterport frame; see
    module docstring). Convert to runtime Y-up downstream when needed.
    """
    size = depth.shape[0]
    dirs = face_ray_directions(size)
    rot = np.asarray(cam_matrix, dtype=np.float64)[:3, :3]
    pos = np.asarray(cam_matrix, dtype=np.float64)[:3, 3]
    world_dirs = dirs @ rot.T
    return pos + world_dirs * np.asarray(depth, dtype=np.float64)[..., None]


def prepare(poses_path: Path, output_path: Path, size: int) -> int:
    """Emit gbuffer_jobs.json (reuses the validated pose_qa job machinery)."""
    from vs_pipeline.stages.pose_qa import prepare_render_jobs

    count = prepare_render_jobs(poses_path, output_path, render_size=size)
    print(f"{count} sweep(s) -> {output_path} (render size {size})")
    return count


def _load_depth(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return np.asarray(img, dtype=np.float64)


def _depth_vis(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth < 1e6)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if bool(finite.any()):
        d = depth[finite]
        lo, hi = float(d.min()), float(np.percentile(d, 99))
        span = max(hi - lo, 1e-6)
        vis[finite] = np.clip(
            (depth[finite] - lo) / span * 255.0, 0, 255
        ).astype(np.uint8)
    return vis


def _find_pano_face(panos_dir: Path, sweep: str, face: str) -> Path | None:
    base = panos_dir / sweep
    for ext in ("png", "jpg", "jpeg"):
        p = base / f"{face}.{ext}"
        if p.is_file():
            return p
    hits = sorted(base.glob(f"*{face}*")) if base.is_dir() else []
    return hits[0] if hits else None


def verify(
    renders_dir: Path,
    panos_dir: Path,
    calibration_path: Path,
    out_dir: Path,
    limit: int | None = None,
) -> None:
    """Proxy depth edges vs pano faces: composites + alignment scores.

    Uses the LOCKED F5 face-mapping calibration (extracted face ->
    rendered face + orientation op) and the F5 chance-corrected overlap
    metric. High scores + edges hugging walls/doors/stairs = the proxy is
    pixel-aligned with the photography; misplaced geometry shows as red
    edges floating off the pano content.
    """
    from vs_pipeline.stages.pose_qa import (
        FACE_NAMES,
        _normalized_overlap,
        _pano_edges_dilated,
        apply_orientation,
    )

    cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    mapping = cal.get("mapping", cal)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_dirs = sorted(d for d in renders_dir.iterdir() if d.is_dir())
    if limit is not None:
        sweep_dirs = sweep_dirs[:limit]
    all_scores: list[float] = []
    for sweep_dir in sweep_dirs:
        sweep = sweep_dir.name
        rows: list[np.ndarray] = []
        scores: list[float] = []
        for ext_face in FACE_NAMES:
            m = mapping.get(ext_face)
            if not m:
                continue
            depth_path = sweep_dir / f"{m['render_face']}_depth.exr"
            pano_path = _find_pano_face(panos_dir, sweep, ext_face)
            if not depth_path.is_file() or pano_path is None:
                continue
            depth = _load_depth(depth_path)
            if depth is None:
                continue
            vis = apply_orientation(_depth_vis(depth), str(m["op"]))
            vis = np.ascontiguousarray(vis)
            edges = cv2.Canny(vis, 10, 40)
            pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
            if pano is None:
                continue
            pano = cv2.resize(pano, (vis.shape[1], vis.shape[0]))
            pano_gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
            pano_dilated = _pano_edges_dilated(pano_gray, edges.shape)
            score = _normalized_overlap(pano_dilated, edges)
            scores.append(score)
            overlay = pano.copy()
            overlay[edges > 0] = (0, 0, 255)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            row = np.hstack([pano, vis_bgr, overlay])
            cv2.putText(
                row,
                f"{ext_face}  score {score:+.3f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rows.append(row)
        if not rows:
            print(f"[{sweep}] no verifiable faces (missing renders/panos)")
            continue
        width = max(r.shape[1] for r in rows)
        rows = [
            cv2.copyMakeBorder(
                r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT
            )
            for r in rows
        ]
        composite = np.vstack(rows)
        out_path = out_dir / f"gbuffer_verify_{sweep}.png"
        cv2.imwrite(str(out_path), composite)
        mean = float(np.mean(scores))
        all_scores.extend(scores)
        print(
            f"[{sweep}] mean alignment {mean:+.3f} over {len(scores)} face(s) "
            f"-> {out_path.name}"
        )
    if all_scores:
        print(
            f"overall: {np.mean(all_scores):+.3f} "
            f"(min {np.min(all_scores):+.3f}, max {np.max(all_scores):+.3f}) "
            f"across {len(all_scores)} face(s)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 7 (G4): G-buffer baking (prepare / verify)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare", help="poses.json -> gbuffer_jobs.json")
    p_prep.add_argument("poses", type=Path)
    p_prep.add_argument(
        "--output", "-o", type=Path, default=Path("gbuffer_jobs.json")
    )
    p_prep.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p_ver = sub.add_parser(
        "verify", help="proxy depth edges vs pano faces -> composites+scores"
    )
    p_ver.add_argument("--renders", type=Path, required=True)
    p_ver.add_argument("--panos", type=Path, required=True)
    p_ver.add_argument(
        "--calibration", type=Path, required=True, help="F5 calibration.json"
    )
    p_ver.add_argument("--limit", type=int, default=None)
    p_ver.add_argument(
        "--output", "-o", type=Path, default=Path("gbuffer_review")
    )
    args = parser.parse_args(argv)

    if args.command == "prepare":
        if not args.poses.is_file():
            parser.error(f"poses file not found: {args.poses}")
        prepare(args.poses, args.output, args.size)
        return 0
    for path, name in (
        (args.renders, "renders dir"),
        (args.panos, "panos dir"),
        (args.calibration, "calibration file"),
    ):
        if not path.exists():
            parser.error(f"{name} not found: {path}")
    verify(args.renders, args.panos, args.calibration, args.output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
