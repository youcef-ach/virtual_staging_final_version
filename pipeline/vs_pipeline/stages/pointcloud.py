"""Stage 3 (G1) — point cloud preprocessing.

Per-sweep chunked processing of the raw E57 — the full cloud is never loaded
into memory. The E57's native tiling IS its sweeps: each scan is a few
million points, read individually with pye57 (proven in this repo), which
also gives every chunk a known sensor origin for correct normal orientation.
(PDAL tiling is unnecessary for E57 sources; it becomes relevant only if
non-E57 inputs ever appear.)

Per sweep:
1. read global-frame points (pye57 applies the scan pose),
2. convert once to the runtime Y-up frame (Coordinate Lock Rule),
3. statistical outlier removal (SOR: 30 neighbors, sigma = 2.0),
4. voxel downsample to 15 mm,
5. normals (5 cm hybrid radius) oriented toward the sweep's sensor origin,
6. cache as sweep_cache/scan_NNN.ply (idempotent, resumable).

Merge: progressive fold of the cached sweep clouds with periodic voxel
passes (bounded memory), final dedup voxel pass, averaged normals
re-normalized, written as a compressed binary PLY working cloud plus a
preprocess_meta.json provenance record. The raw E57 archive is never
modified — full resolution stays archived untouched.

Requires the 'pointcloud' extra:  pip install -e .[pointcloud]  (Open3D)

CLI:
    python -m vs_pipeline.stages.pointcloud preprocess cloud.e57 -o preprocessed
    python -m vs_pipeline.stages.pointcloud preprocess cloud.e57 --limit 5   # quick pass
    python -m vs_pipeline.stages.pointcloud merge -o preprocessed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vs_pipeline.coordinates import ZUP_TO_YUP

PREPROCESS_SCHEMA_VERSION = 1

DEFAULT_VOXEL_M = 0.015
DEFAULT_SOR_NEIGHBORS = 30
DEFAULT_SOR_STD = 2.0
DEFAULT_NORMAL_RADIUS_M = 0.05
DEFAULT_NORMAL_MAX_NN = 30
#: Fold the running merge down with a voxel pass every N sweeps (bounds memory).
MERGE_FOLD_EVERY = 16

_R_ZUP_TO_YUP = ZUP_TO_YUP[:3, :3]


def convert_points_zup_to_yup(points_mp: np.ndarray) -> np.ndarray:
    """Vectorized Coordinate Lock conversion for an (N, 3) point array."""
    pts = np.asarray(points_mp, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {pts.shape}")
    return pts @ _R_ZUP_TO_YUP.T


def preprocess_scan(
    points_rt: np.ndarray,
    sensor_origin_rt: np.ndarray,
    voxel_m: float = DEFAULT_VOXEL_M,
    sor_neighbors: int = DEFAULT_SOR_NEIGHBORS,
    sor_std: float = DEFAULT_SOR_STD,
    normal_radius_m: float = DEFAULT_NORMAL_RADIUS_M,
    normal_max_nn: int = DEFAULT_NORMAL_MAX_NN,
) -> tuple[Any, dict[str, int]]:
    """SOR -> voxel -> sensor-oriented normals for one sweep (runtime frame).

    Returns the processed Open3D cloud and per-step point counts.
    """
    import open3d as o3d  # local import: heavy optional dependency

    pts = np.asarray(points_rt, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    raw = len(cloud.points)

    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=sor_neighbors, std_ratio=sor_std
    )
    after_sor = len(cloud.points)

    cloud = cloud.voxel_down_sample(voxel_m)
    after_voxel = len(cloud.points)

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_m, max_nn=normal_max_nn)
    )
    cloud.orient_normals_towards_camera_location(
        np.asarray(sensor_origin_rt, dtype=np.float64)
    )
    return cloud, {"raw": raw, "after_sor": after_sor, "after_voxel": after_voxel}


def _write_meta(out_dir: Path, updates: dict[str, Any]) -> None:
    """Merge updates into preprocess_meta.json (created if missing)."""
    meta_path = out_dir / "preprocess_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(updates)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def preprocess_property(
    e57_path: Path,
    out_dir: Path,
    voxel_m: float = DEFAULT_VOXEL_M,
    sor_neighbors: int = DEFAULT_SOR_NEIGHBORS,
    sor_std: float = DEFAULT_SOR_STD,
    normal_radius_m: float = DEFAULT_NORMAL_RADIUS_M,
    limit: int = 0,
    force: bool = False,
) -> int:
    """Clean + downsample every sweep into the per-sweep cache (resumable)."""
    import open3d as o3d  # local import
    import pye57  # local import

    e57 = pye57.E57(str(e57_path))
    total = e57.scan_count
    
    import os
    sweep_range = range(total)
    if os.environ.get("VS_TEST_SWEEP_RANGE"):
        parts = os.environ["VS_TEST_SWEEP_RANGE"].split("-")
        sweep_range = range(int(parts[0]), int(parts[1]) + 1)
    else:
        if limit > 0:
            sweep_range = range(min(limit, total))

    count = len(sweep_range)
    cache = out_dir / "sweep_cache"
    cache.mkdir(parents=True, exist_ok=True)

    new_stats: dict[str, dict[str, int]] = {}
    for index in sweep_range:
        if index >= total:
            continue
        chunk_path = cache / f"scan_{index:03d}.ply"
        if chunk_path.exists() and not force:
            print(f"[{index:03d}] cached -> skip")
            continue
        started = time.perf_counter()
        header = e57.get_header(index)
        data = e57.read_scan(index, ignore_missing_fields=True)
        pts_mp = np.column_stack(
            [data["cartesianX"], data["cartesianY"], data["cartesianZ"]]
        )
        origin_rt = convert_points_zup_to_yup(
            np.asarray(header.translation, dtype=np.float64).reshape(1, 3)
        )[0]
        cloud, stats = preprocess_scan(
            convert_points_zup_to_yup(pts_mp),
            origin_rt,
            voxel_m=voxel_m,
            sor_neighbors=sor_neighbors,
            sor_std=sor_std,
            normal_radius_m=normal_radius_m,
        )
        o3d.io.write_point_cloud(
            str(chunk_path), cloud, write_ascii=False, compressed=True
        )
        new_stats[f"{index:03d}"] = stats
        elapsed = time.perf_counter() - started
        print(
            f"[{index:03d}/{count - 1}] raw {stats['raw']:,} -> SOR {stats['after_sor']:,} "
            f"-> voxel {stats['after_voxel']:,}  ({elapsed:.1f}s)"
        )

    meta_path = out_dir / "preprocess_meta.json"
    scan_stats: dict[str, dict[str, int]] = {}
    if meta_path.exists():
        scan_stats = json.loads(meta_path.read_text(encoding="utf-8")).get("scan_stats", {})
    scan_stats.update(new_stats)
    _write_meta(
        out_dir,
        {
            "schema_version": PREPROCESS_SCHEMA_VERSION,
            "coordinate_frame": "runtime_yup",
            "source_file": e57_path.name,
            "source_bytes": e57_path.stat().st_size,
            "params": {
                "voxel_m": voxel_m,
                "sor_neighbors": sor_neighbors,
                "sor_std": sor_std,
                "normal_radius_m": normal_radius_m,
            },
            "sweep_count": total,
            "processed_sweeps": count,
            "scan_stats": scan_stats,
        },
    )
    return count


def merge_cached(out_dir: Path, voxel_m: float = DEFAULT_VOXEL_M) -> tuple[Path, Any]:
    """Progressively merge cached sweep clouds into the deduplicated working cloud."""
    import open3d as o3d  # local import

    cache = out_dir / "sweep_cache"
    chunks = sorted(cache.glob("scan_*.ply"))
    if not chunks:
        raise ValueError(f"no cached sweep clouds under {cache}")

    merged = o3d.geometry.PointCloud()
    for i, chunk in enumerate(chunks, start=1):
        merged += o3d.io.read_point_cloud(str(chunk))
        if i % MERGE_FOLD_EVERY == 0:
            merged = merged.voxel_down_sample(voxel_m)
            print(f"merge fold {i}/{len(chunks)}: {len(merged.points):,} points")
    merged = merged.voxel_down_sample(voxel_m)

    # Voxel averaging blends normals across overlapping sweeps; re-normalize.
    normals = np.asarray(merged.normals)
    if normals.size:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        merged.normals = o3d.utility.Vector3dVector(normals / norms)

    out_path = out_dir / "working_cloud.ply"
    o3d.io.write_point_cloud(str(out_path), merged, write_ascii=False, compressed=True)
    _write_meta(
        out_dir,
        {
            "merged_chunks": len(chunks),
            "final_points": len(merged.points),
            "bounds_min_rt": np.asarray(merged.get_min_bound()).tolist(),
            "bounds_max_rt": np.asarray(merged.get_max_bound()).tolist(),
            "working_cloud": out_path.name,
        },
    )
    print(
        f"{len(chunks)} sweep cloud(s) merged -> {out_path} "
        f"({len(merged.points):,} points)"
    )
    return out_path, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 (G1): point cloud preprocessing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser(
        "preprocess", help="per-sweep clean + downsample + normals, then merge"
    )
    p_pre.add_argument("input", type=Path, help="path to the .e57 file")
    p_pre.add_argument("--output", "-o", type=Path, default=Path("preprocessed"))
    p_pre.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_M)
    p_pre.add_argument("--sor-neighbors", type=int, default=DEFAULT_SOR_NEIGHBORS)
    p_pre.add_argument("--sor-std", type=float, default=DEFAULT_SOR_STD)
    p_pre.add_argument("--normal-radius", type=float, default=DEFAULT_NORMAL_RADIUS_M)
    p_pre.add_argument(
        "--limit", type=int, default=0, help="process only the first N sweeps (quick pass)"
    )
    p_pre.add_argument("--force", action="store_true", help="reprocess cached sweeps")
    p_pre.add_argument("--no-merge", action="store_true", help="skip the merge step")

    p_merge = sub.add_parser("merge", help="merge cached per-sweep clouds only")
    p_merge.add_argument("--output", "-o", type=Path, default=Path("preprocessed"))
    p_merge.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_M)

    args = parser.parse_args(argv)

    if args.command == "preprocess":
        if not args.input.is_file():
            parser.error(f"input file not found: {args.input}")
        count = preprocess_property(
            args.input,
            args.output,
            voxel_m=args.voxel,
            sor_neighbors=args.sor_neighbors,
            sor_std=args.sor_std,
            normal_radius_m=args.normal_radius,
            limit=args.limit,
            force=args.force,
        )
        print(f"{count} sweep(s) preprocessed -> {args.output / 'sweep_cache'}")
        if not args.no_merge:
            merge_cached(args.output, voxel_m=args.voxel)
        return 0

    merge_cached(args.output, voxel_m=args.voxel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
