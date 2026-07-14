"""Stage 6 (G3) slice 2 -- proxy shell synthesis (rooms -> labeled prisms).

Design (PLAN.md G3 brief step 4, first pass):
- Room polygons from slice 1 sit on wall CENTERLINES; the runtime proxy
  must sit on the interior wall FACES the camera sees. Each edge is offset
  inward by half its matched wall thickness (default 100 mm walls) and the
  corners are rebuilt by intersecting adjacent offset edge lines -- exact
  plane-intersection corner sealing in plan view (< 1 mm by construction).
  If the inset degenerates (tiny/self-intersecting rooms) the centerline
  polygon is used and flagged in the manifest.
- Room prism = floor cap (elected floor elevation) + ceiling cap (elected
  ceiling, storey fallback) + one wall quad per edge; normals face INTO
  the room (the proxy is viewed from inside); labeled sub-meshes
  ``<room_id>/floor|ceiling|wall_NN``. Watertightness is verified per room.
- Openings (doors/windows), non-planar detail merge (OBJ clip) and Draco
  compression land in later slices; export is plain GLB for now.

Outputs:
    proxy_shell.glb              all rooms, labeled sub-meshes
    proxy_manifest.json          per-room tris / watertight / fallbacks
    proxy_review_storey<k>.png   centerline outline vs inset footprint

CLI:
    python -m vs_pipeline.stages.proxy build --rooms rooms_out/rooms.json -o proxy_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon

PROXY_SCHEMA_VERSION = 1

#: Unpaired walls carry no measured thickness; assume a light partition.
DEFAULT_WALL_THICKNESS_M = 0.1
#: Inset polygons losing more area than this are degenerate -> fallback.
MIN_INSET_AREA_RATIO = 0.4
#: Adjacent offset edges more parallel than this cannot intersect cleanly.
PARALLEL_EPS = 1e-3


def inset_room_polygon(
    ring: list[tuple[float, float]] | list[list[float]],
    insets: list[float],
) -> Polygon | None:
    """Offset every edge inward by its inset; corners = line intersections.

    ``ring`` is the OPEN outline (no repeated last point); ``insets[i]``
    belongs to the edge ring[i] -> ring[i+1]. Output vertex i is the
    intersection of offset edges i-1 and i (the sealed corner at ring[i]).
    """
    pts = np.asarray(ring, dtype=np.float64)
    n = pts.shape[0]
    if n < 3 or len(insets) != n:
        return None
    area2 = float(
        np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    )
    ccw = area2 > 0
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            return None
        d = d / length
        n_left = np.array([-d[1], d[0]])
        n_in = n_left if ccw else -n_left
        lines.append((a + n_in * insets[i], d))
    verts: list[tuple[float, float]] = []
    for i in range(n):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < PARALLEL_EPS:
            verts.append((float(p2[0]), float(p2[1])))
            continue
        r = p2 - p1
        t = float(r[0] * d2[1] - r[1] * d2[0]) / cross
        q = p1 + d1 * t
        verts.append((float(q[0]), float(q[1])))
    poly = Polygon(verts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def room_shell_mesh(
    poly: Polygon, floor_y: float, ceiling_y: float
) -> dict[str, Any]:
    """Labeled prism parts for one room; normals face INTO the room."""
    import trimesh

    verts2d, faces2d = trimesh.creation.triangulate_polygon(poly)

    def cap(y: float, face_up: bool) -> Any:
        v = np.column_stack(
            [verts2d[:, 0], np.full(verts2d.shape[0], y), verts2d[:, 1]]
        )
        mesh = trimesh.Trimesh(vertices=v, faces=faces2d, process=False)
        mean_ny = float(mesh.face_normals[:, 1].mean())
        if (face_up and mean_ny < 0) or (not face_up and mean_ny > 0):
            mesh.invert()
        return mesh

    parts: dict[str, Any] = {
        "floor": cap(floor_y, face_up=True),
        "ceiling": cap(ceiling_y, face_up=False),
    }
    rp = poly.representative_point()
    centre = np.array([rp.x, 0.0, rp.y])
    ring = np.asarray(poly.exterior.coords, dtype=np.float64)
    for k in range(ring.shape[0] - 1):
        (x0, z0), (x1, z1) = ring[k], ring[k + 1]
        v = np.array(
            [
                [x0, floor_y, z0],
                [x1, floor_y, z1],
                [x1, ceiling_y, z1],
                [x0, ceiling_y, z0],
            ]
        )
        f = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mid = np.array([(x0 + x1) / 2.0, 0.0, (z0 + z1) / 2.0])
        if float(mesh.face_normals[0] @ (centre - mid)) < 0:
            mesh.invert()
        parts[f"wall_{k:02d}"] = mesh
    return parts


def _palette(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(index * 7919 + 13)
    c = (rng.uniform(0.35, 1.0, size=3) * 255).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def _render_review(
    review: dict[int, list[tuple[Polygon, Polygon]]], out_dir: Path, size_px: int = 1600
) -> None:
    """Per storey: centerline outline (white) vs inset shell footprint (fill)."""
    for storey_id, items in review.items():
        xs: list[float] = []
        zs: list[float] = []
        for orig, _ in items:
            bx0, bz0, bx1, bz1 = orig.bounds
            xs.extend([bx0, bx1])
            zs.extend([bz0, bz1])
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
        overlay = image.copy()
        for k, (_, inset) in enumerate(items):
            pts = np.array(
                [px(x, z) for x, z in inset.exterior.coords], dtype=np.int32
            )
            cv2.fillPoly(overlay, [pts], _palette(k))
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        for orig, _ in items:
            pts = np.array(
                [px(x, z) for x, z in orig.exterior.coords], dtype=np.int32
            )
            cv2.polylines(image, [pts], True, (255, 255, 255), 1)
        cv2.imwrite(
            str(out_dir / f"proxy_review_storey{storey_id}.png"), image
        )


def build_property(rooms_path: Path, out_dir: Path) -> dict[str, Any]:
    import trimesh

    data = json.loads(rooms_path.read_text(encoding="utf-8"))
    if data.get("schema_version", 1) < 2:
        raise SystemExit(
            "rooms.json is schema v1 (no per-edge wall data) -- re-run "
            "`vs_pipeline.stages.rooms build` first"
        )
    storeys = {s["storey_id"]: s for s in data["storeys"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    manifest_rooms: list[dict[str, Any]] = []
    review: dict[int, list[tuple[Polygon, Polygon]]] = {}
    total_tris = 0
    skipped = 0
    for room in data["rooms"]:
        st = storeys[room["storey_id"]]
        floor_y = (
            room["floor_elevation_y"]
            if room["floor_elevation_y"] is not None
            else st["floor_y"]
        )
        ceiling_y = (
            room["ceiling_elevation_y"]
            if room["ceiling_elevation_y"] is not None
            else st["ceiling_y"]
        )
        if floor_y is None or ceiling_y is None:
            print(f"  {room['room_id']}: no floor/ceiling elevation -> skipped")
            skipped += 1
            continue
        ring = [tuple(p) for p in room["polygon_plan"]]
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        edges = room.get("edges") or []
        if len(edges) == len(ring):
            insets = [
                (e["thickness_m"] or DEFAULT_WALL_THICKNESS_M) / 2.0 for e in edges
            ]
        else:
            insets = [DEFAULT_WALL_THICKNESS_M / 2.0] * len(ring)
        orig = Polygon(ring)
        if not orig.is_valid:
            orig = orig.buffer(0)
        inset_poly = inset_room_polygon(ring, insets)
        fallback = False
        if (
            inset_poly is None
            or inset_poly.area < MIN_INSET_AREA_RATIO * orig.area
        ):
            inset_poly = orig
            fallback = True
        parts = room_shell_mesh(inset_poly, float(floor_y), float(ceiling_y))
        merged = trimesh.util.concatenate(list(parts.values()))
        merged.merge_vertices()
        tris = int(sum(len(m.faces) for m in parts.values()))
        total_tris += tris
        for name, mesh in parts.items():
            scene.add_geometry(mesh, geom_name=f"{room['room_id']}/{name}")
        manifest_rooms.append(
            {
                "room_id": room["room_id"],
                "storey_id": room["storey_id"],
                "triangles": tris,
                "watertight": bool(merged.is_watertight),
                "inset_fallback": fallback,
                "floor_y": round(float(floor_y), 4),
                "ceiling_y": round(float(ceiling_y), 4),
                "area_m2": round(inset_poly.area, 3),
            }
        )
        review.setdefault(room["storey_id"], []).append((orig, inset_poly))

    glb_path = out_dir / "proxy_shell.glb"
    scene.export(str(glb_path))
    manifest = {
        "schema_version": PROXY_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "total_triangles": total_tris,
        "rooms": manifest_rooms,
    }
    (out_dir / "proxy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _render_review(review, out_dir)

    for r in manifest_rooms:
        marks = "" if r["watertight"] else "  NOT WATERTIGHT"
        if r["inset_fallback"]:
            marks += "  inset-fallback"
        print(
            f"  {r['room_id']:<14} {r['triangles']:>4} tris  "
            f"y {r['floor_y']:+.3f}..{r['ceiling_y']:+.3f}  "
            f"{r['area_m2']:7.2f} m^2{marks}"
        )
    print(
        f"{len(manifest_rooms)} room shell(s), {total_tris} triangles total"
        + (f", {skipped} skipped" if skipped else "")
    )
    print(f"outputs -> {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 6 (G3) slice 2: proxy shell synthesis."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="rooms.json -> proxy_shell.glb")
    p_build.add_argument("--rooms", type=Path, required=True, help="rooms.json")
    p_build.add_argument("--output", "-o", type=Path, default=Path("proxy_out"))
    args = parser.parse_args(argv)

    if not args.rooms.is_file():
        parser.error(f"rooms file not found: {args.rooms}")
    build_property(args.rooms, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
