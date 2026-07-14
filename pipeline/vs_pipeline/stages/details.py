"""Stage 6 (G3) final slice -- non-planar detail proxies (Decision 6).

The proxy shell handles planar structure; everything non-planar that must
depth-occlude virtual assets (stairs, kitchen counters, built-ins, real
furniture) comes from here:
1. G2's residual clusters mark WHERE non-planar geometry lives. Clusters
   are filtered by size and interiority (plan centroid inside an accepted
   room polygon) and overlapping boxes are merged.
2. The MatterPak OBJ (same frame as the E57; converted Z-up -> Y-up once,
   Coordinate Lock rule) is clipped to each merged box by triangle
   centroid. Occlusion silhouettes only -- the OBJ's decimation is a
   documented, tolerated limitation (E57 stays metric truth).

Outputs:
    proxy_details.glb              labeled detail_NNN meshes
    details_manifest.json          boxes, tri counts, sources
    details_review_storey<k>.png   detail boxes over room outlines

CLI:
    python -m vs_pipeline.stages.details build \
        --clusters planes_full/residual_clusters.json \
        --rooms rooms_out/rooms.json --obj matterpak.obj -o details_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

DETAILS_SCHEMA_VERSION = 1

MIN_CLUSTER_POINTS = 100
MIN_DIM_M = 0.15
MIN_HEIGHT_M = 0.2
MAX_DIM_M = 6.0
BOX_PAD_M = 0.05
MAX_TRIS_PER_DETAIL = 1500


def zup_to_yup_vertices(vertices: np.ndarray) -> np.ndarray:
    """Matterport Z-up -> runtime Y-up (X, Z, -Y); Coordinate Lock rule."""
    v = np.asarray(vertices, dtype=np.float64)
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def filter_clusters(
    clusters: list[dict[str, Any]], room_polys: list[Polygon]
) -> list[dict[str, Any]]:
    """Size + interiority filter: keep furniture/stair-scale interior boxes."""
    kept = []
    for c in clusters:
        if int(c["point_count"]) < MIN_CLUSTER_POINTS:
            continue
        bmin = np.asarray(c["bbox_min"], dtype=np.float64)
        bmax = np.asarray(c["bbox_max"], dtype=np.float64)
        dims = bmax - bmin
        if float(dims.min()) < MIN_DIM_M or float(dims.max()) > MAX_DIM_M:
            continue
        if float(dims[1]) < MIN_HEIGHT_M:
            continue
        cx, _, cz = c["centroid"]
        pt = Point(float(cx), float(cz))
        if not any(poly.covers(pt) for poly in room_polys):
            continue
        kept.append(c)
    return kept


def merge_cluster_boxes(
    clusters: list[dict[str, Any]],
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Union-find merge of overlapping cluster boxes -> (min, max, count)."""
    n = len(clusters)
    mins = [np.asarray(c["bbox_min"], dtype=np.float64) for c in clusters]
    maxs = [np.asarray(c["bbox_max"], dtype=np.float64) for c in clusters]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if bool(
                np.all(mins[i] - BOX_PAD_M <= maxs[j])
                and np.all(mins[j] - BOX_PAD_M <= maxs[i])
            ):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        gmin = np.min([mins[k] for k in members], axis=0)
        gmax = np.max([maxs[k] for k in members], axis=0)
        out.append((gmin, gmax, len(members)))
    return out


def clip_mesh_to_box(mesh: Any, bmin: np.ndarray, bmax: np.ndarray) -> Any | None:
    """Sub-mesh of triangles whose centroid lies inside the (padded) box."""
    import trimesh

    centroids = mesh.triangles_center
    lo = bmin - BOX_PAD_M
    hi = bmax + BOX_PAD_M
    mask = np.all((centroids >= lo) & (centroids <= hi), axis=1)
    if not bool(mask.any()):
        return None
    sub = trimesh.Trimesh(
        vertices=mesh.vertices.copy(), faces=mesh.faces[mask], process=False
    )
    sub.remove_unreferenced_vertices()
    if len(sub.faces) > MAX_TRIS_PER_DETAIL:
        try:
            sub = sub.simplify_quadric_decimation(MAX_TRIS_PER_DETAIL)
        except BaseException:
            pass  # keep full-res detail; noted via tri count in the manifest
    return sub


def build_details(
    clusters: list[dict[str, Any]],
    rooms_data: dict[str, Any],
    mesh: Any,
) -> tuple[dict[str, Any], list[Any], list[tuple[np.ndarray, np.ndarray]]]:
    """Core pass: filter -> merge -> clip. Returns (manifest, meshes, boxes)."""
    room_polys = [
        Polygon([tuple(p) for p in r["polygon_plan"]])
        for r in rooms_data.get("rooms", [])
        if len(r.get("polygon_plan", [])) >= 3
    ]
    kept = filter_clusters(clusters, room_polys)
    boxes = merge_cluster_boxes(kept)
    entries: list[dict[str, Any]] = []
    meshes: list[Any] = []
    used_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for bmin, bmax, count in sorted(
        boxes, key=lambda b: -float(np.prod(b[1] - b[0]))
    ):
        sub = clip_mesh_to_box(mesh, bmin, bmax)
        if sub is None or len(sub.faces) == 0:
            continue
        detail_id = f"detail_{len(entries):03d}"
        entries.append(
            {
                "detail_id": detail_id,
                "bbox_min": [round(float(v), 4) for v in bmin],
                "bbox_max": [round(float(v), 4) for v in bmax],
                "source_clusters": count,
                "triangles": int(len(sub.faces)),
            }
        )
        meshes.append(sub)
        used_boxes.append((bmin, bmax))
    manifest = {
        "schema_version": DETAILS_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "cluster_count": len(clusters),
        "kept_clusters": len(kept),
        "detail_count": len(entries),
        "total_triangles": int(sum(e["triangles"] for e in entries)),
        "details": entries,
    }
    return manifest, meshes, used_boxes


def _render_review(
    rooms_data: dict[str, Any],
    boxes: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    size_px: int = 1600,
) -> None:
    storeys = rooms_data.get("storeys", [])
    rooms = rooms_data.get("rooms", [])
    for st in storeys:
        sid = st["storey_id"]
        floor_y = float(st["floor_y"])
        top = float(st["ceiling_y"]) if st.get("ceiling_y") is not None else floor_y + 3.0
        st_rooms = [r for r in rooms if r["storey_id"] == sid]
        st_boxes = [
            (bmin, bmax)
            for bmin, bmax in boxes
            if floor_y - 0.3 <= float((bmin[1] + bmax[1]) / 2.0) <= top + 0.3
        ]
        if not st_rooms and not st_boxes:
            continue
        xs: list[float] = []
        zs: list[float] = []
        for r in st_rooms:
            for x, z in r["polygon_plan"]:
                xs.append(float(x))
                zs.append(float(z))
        for bmin, bmax in st_boxes:
            xs.extend([float(bmin[0]), float(bmax[0])])
            zs.extend([float(bmin[2]), float(bmax[2])])
        if not xs:
            continue
        xmin, zmin = min(xs), min(zs)
        extent = max(max(xs) - xmin, max(zs) - zmin, 1e-6)
        scale = (size_px - 40) / extent
        width = int((max(xs) - xmin) * scale) + 40
        height = int((max(zs) - zmin) * scale) + 40

        def px(x: float, z: float) -> tuple[int, int]:
            return int((x - xmin) * scale) + 20, int((z - zmin) * scale) + 20

        image = np.zeros((height, width, 3), dtype=np.uint8)
        for r in st_rooms:
            pts = np.array(
                [px(float(x), float(z)) for x, z in r["polygon_plan"]],
                dtype=np.int32,
            )
            cv2.polylines(image, [pts], True, (200, 200, 200), 1)
        for bmin, bmax in st_boxes:
            cv2.rectangle(
                image,
                px(float(bmin[0]), float(bmin[2])),
                px(float(bmax[0]), float(bmax[2])),
                (0, 80, 255),
                2,
            )
        cv2.imwrite(str(out_dir / f"details_review_storey{sid}.png"), image)


def build_property(
    clusters_path: Path, rooms_path: Path, obj_path: Path, out_dir: Path
) -> dict[str, Any]:
    import trimesh

    cdata = json.loads(clusters_path.read_text(encoding="utf-8"))
    rooms_data = json.loads(rooms_path.read_text(encoding="utf-8"))
    print(f"loading MatterPak OBJ: {obj_path} ...")
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    mesh = trimesh.Trimesh(
        vertices=zup_to_yup_vertices(np.asarray(mesh.vertices)),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    print(f"  {len(mesh.faces):,} triangles (runtime Y-up)")

    manifest, meshes, boxes = build_details(
        cdata.get("clusters", []), rooms_data, mesh
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for entry, sub in zip(manifest["details"], meshes):
        scene.add_geometry(sub, geom_name=entry["detail_id"])
    scene.export(str(out_dir / "proxy_details.glb"))
    (out_dir / "details_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _render_review(rooms_data, boxes, out_dir)

    print(
        f"{manifest['kept_clusters']}/{manifest['cluster_count']} clusters kept "
        f"-> {manifest['detail_count']} detail proxies, "
        f"{manifest['total_triangles']:,} triangles"
    )
    biggest = manifest["details"][:8]
    for e in biggest:
        dims = np.asarray(e["bbox_max"]) - np.asarray(e["bbox_min"])
        print(
            f"  {e['detail_id']}  {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m  "
            f"{e['triangles']:>5} tris  ({e['source_clusters']} cluster(s))"
        )
    print(f"outputs -> {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 6 (G3): non-planar detail proxies (Decision 6)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="clusters + rooms + OBJ -> details")
    p_build.add_argument("--clusters", type=Path, required=True)
    p_build.add_argument("--rooms", type=Path, required=True)
    p_build.add_argument("--obj", type=Path, required=True, help="MatterPak OBJ")
    p_build.add_argument("--output", "-o", type=Path, default=Path("details_out"))
    args = parser.parse_args(argv)

    for path, name in (
        (args.clusters, "clusters"),
        (args.rooms, "rooms"),
        (args.obj, "obj"),
    ):
        if not path.is_file():
            parser.error(f"{name} file not found: {path}")
    build_property(args.clusters, args.rooms, args.obj, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
