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
- Openings: G3 jamb-detected DOORS (--openings, floor-reaching notches)
  and G6 fusion WINDOWS (--windows, interior holes) are subtracted from
  wall faces. Windows are assigned to room edges GEOMETRICALLY (proximity
  only, no plane-id prefilter) because fusion may register a window on a
  duplicate wall plane the room's edge list never references (LESSONS L3).
- Non-planar detail merge lives in details.py; Draco compression lands in
  P4; export is plain GLB for now.

Outputs:
    proxy_shell.glb              all rooms, labeled sub-meshes
    proxy_manifest.json          per-room tris / watertight / fallbacks
    proxy_review_storey<k>.png   centerline outline vs inset footprint

CLI:
    python -m vs_pipeline.stages.proxy build --rooms rooms_out/rooms.json \
        --openings openings_out/openings.json \
        --windows fusion_out/openings_fused.json -o proxy_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon, box

PROXY_SCHEMA_VERSION = 1

#: Unpaired walls carry no measured thickness; assume a light partition.
DEFAULT_WALL_THICKNESS_M = 0.1
#: Inset polygons losing more area than this are degenerate -> fallback.
MIN_INSET_AREA_RATIO = 0.4
#: Adjacent offset edges more parallel than this cannot intersect cleanly.
PARALLEL_EPS = 1e-3
#: Kind-aware opening gates.
DOOR_MIN_WIDTH_M = 0.4
WINDOW_MIN_WIDTH_M = 0.25
OPENING_MIN_HEIGHT_M = 0.2
#: No door/passage rect may exceed this width (a wall-sized false rect
#: must not delete a wall; real double/patio doors + open passages fit).
DOOR_MAX_WIDTH_M = 4.0
#: Fused doors (--doors-fused) must reach this far above the room floor:
#: partial/low rects (#21 evidence splitting) are rejected -- a half-height
#: cutout would trade a phantom wall for a phantom lintel.
FUSED_DOOR_MIN_TOP_ABOVE_FLOOR_M = 1.5
#: Windows keep a minimal sill above the floor (floor-reaching 'windows'
#: are glass doors; a 2 cm sill is harmless for occlusion).
WINDOW_SILL_MIN_M = 0.02


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
    poly: Polygon,
    floor_y: float,
    ceiling_y: float,
    edge_openings: list[list[tuple]] | None = None,
) -> dict[str, Any]:
    """Labeled prism parts for one room; normals face INTO the room.

    ``edge_openings[k]`` holds opening rects for edge k in the edge's own
    arc-length parameter: (s0, s1, y0, y1) full rects (windows = interior
    holes, doors = rects whose y0 dips below the floor), or legacy door
    3-tuples (s0, s1, y_top) which imply y0 = floor - 1 cm. Rects are
    subtracted from the wall face; earcut triangulates faces with holes.
    """
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
        a2 = np.array([x0, z0])
        b2 = np.array([x1, z1])
        length = float(np.linalg.norm(b2 - a2))
        if length < 1e-9:
            continue
        dir2 = (b2 - a2) / length
        rects = (
            edge_openings[k]
            if edge_openings is not None and k < len(edge_openings)
            else []
        )
        if rects:
            face = box(0.0, floor_y, length, ceiling_y)
            for rect in rects:
                if len(rect) == 3:  # legacy door rect (s0, s1, y_top)
                    s0, s1, y0, y1 = rect[0], rect[1], floor_y - 0.01, rect[2]
                else:
                    s0, s1, y0, y1 = rect
                s0c = max(float(s0), 0.02)
                s1c = min(float(s1), length - 0.02)
                if s1c - s0c < 0.05 or float(y1) - float(y0) < 0.05:
                    continue
                face = face.difference(box(s0c, float(y0), s1c, float(y1)))
            pieces = [face] if face.geom_type == "Polygon" else list(face.geoms)
            v_parts: list[np.ndarray] = []
            f_parts: list[np.ndarray] = []
            offset = 0
            for piece in pieces:
                if piece.is_empty or piece.area < 1e-6:
                    continue
                v2, f2 = trimesh.creation.triangulate_polygon(piece)
                v3 = np.column_stack(
                    [
                        a2[0] + dir2[0] * v2[:, 0],
                        v2[:, 1],
                        a2[1] + dir2[1] * v2[:, 0],
                    ]
                )
                v_parts.append(v3)
                f_parts.append(f2 + offset)
                offset += v3.shape[0]
            if not v_parts:
                continue
            mesh = trimesh.Trimesh(
                vertices=np.vstack(v_parts),
                faces=np.vstack(f_parts),
                process=False,
            )
        else:
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
        if float(mesh.face_normals.mean(axis=0) @ (centre - mid)) < 0:
            mesh.invert()
        parts[f"wall_{k:02d}"] = mesh
    return parts


def _merge_rects(
    rects: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Merge overlapping (s0, s1, y0, y1) rects (dedup of pair-face doubles).

    Two rects merge only when BOTH the s-intervals and the y-intervals
    overlap (within 5 cm) -- a door and a high window at the same s stay
    separate.
    """
    if not rects:
        return []
    ordered = sorted(rects)
    out = [list(ordered[0])]
    for s0, s1, y0, y1 in ordered[1:]:
        last = out[-1]
        s_overlap = s0 <= last[1] + 0.05
        y_overlap = (y0 <= last[3] + 0.05) and (y1 >= last[2] - 0.05)
        if s_overlap and y_overlap:
            last[0] = min(last[0], s0)
            last[1] = max(last[1], s1)
            last[2] = min(last[2], y0)
            last[3] = max(last[3], y1)
        else:
            out.append([s0, s1, y0, y1])
    return [tuple(r) for r in out]


def _room_edge_openings(
    inset_poly: Polygon,
    edges: list[dict[str, Any]],
    openings_by_plane: dict[str, list[dict[str, Any]]],
    window_openings: list[dict[str, Any]],
    floor_y: float,
    ceiling_y: float,
) -> tuple[list[list[tuple[float, float, float, float]]], int]:
    """Assign openings to inset-polygon edges as (s0, s1, y0, y1) rects.

    Doors: candidates prefiltered by the room's wall plane ids, then the
    0.3 m proximity gate. Windows: GEOMETRIC assignment only (no plane-id
    prefilter, LESSONS L3) -- the proximity gate is the guard.
    """
    ring = [
        np.asarray(p[:2], dtype=np.float64)
        for p in inset_poly.exterior.coords
    ][:-1]
    m = len(ring)
    out: list[list[tuple[float, float, float, float]]] = [[] for _ in range(m)]
    cand_ids = {pid for e in edges for pid in e.get("wall_planes", [])}
    candidates: list[dict[str, Any]] = [
        o for pid in sorted(cand_ids) for o in openings_by_plane.get(pid, [])
    ]
    candidates.extend(window_openings)
    for o in candidates:
        kind = str(o.get("kind", "door"))
        corners = np.asarray(o["corners_3d"], dtype=np.float64)
        pxz = corners[:, [0, 2]]
        y_top = float(corners[:, 1].max())
        y_bot = float(corners[:, 1].min())
        # Vertical sanity: the rect must intersect this room's height band.
        if y_bot > ceiling_y - 0.1 or y_top < floor_y + 0.1:
            continue
        mid = pxz.mean(axis=0)
        best_k: int | None = None
        best_d = 0.3
        for k in range(m):
            a2, b2 = ring[k], ring[(k + 1) % m]
            length = float(np.linalg.norm(b2 - a2))
            if length < 1e-9:
                continue
            dir2 = (b2 - a2) / length
            t = float(np.clip((mid - a2) @ dir2, 0.0, length))
            d = float(np.linalg.norm(a2 + dir2 * t - mid))
            if d < best_d:
                best_d, best_k = d, k
        if best_k is None:
            continue
        a2, b2 = ring[best_k], ring[(best_k + 1) % m]
        length = float(np.linalg.norm(b2 - a2))
        dir2 = (b2 - a2) / length
        ss = (pxz - a2) @ dir2
        s0 = float(np.clip(ss.min(), 0.0, length))
        s1 = float(np.clip(ss.max(), 0.0, length))
        min_w = WINDOW_MIN_WIDTH_M if kind == "window" else DOOR_MIN_WIDTH_M
        if s1 - s0 < min_w:
            continue
        if kind == "door":
            # A wall-sized false rect must not delete a wall (any source).
            if s1 - s0 > DOOR_MAX_WIDTH_M:
                continue
            # Fused doors become floor-reaching cutouts; a partial/low rect
            # (#21 evidence splitting) would leave a phantom LINTEL occluder
            # -- require a walk-through top height.
            if (
                str(o.get("source", "")) == "fusion"
                and y_top < floor_y + FUSED_DOOR_MIN_TOP_ABOVE_FLOOR_M
            ):
                continue
        y1 = min(y_top, ceiling_y - (0.05 if kind == "window" else 0.1))
        if kind == "window":
            y0 = max(y_bot, floor_y + WINDOW_SILL_MIN_M)
        else:
            y0 = floor_y - 0.01
        if y1 - y0 < OPENING_MIN_HEIGHT_M:
            continue
        out[best_k].append((s0, s1, y0, y1))
    merged = [_merge_rects(r) for r in out]
    return merged, sum(len(r) for r in merged)


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


def build_property(
    rooms_path: Path,
    out_dir: Path,
    openings_path: Path | None = None,
    windows_path: Path | None = None,
    fused_doors_path: Path | None = None,
) -> dict[str, Any]:
    import trimesh

    data = json.loads(rooms_path.read_text(encoding="utf-8"))
    openings_by_plane: dict[str, list[dict[str, Any]]] = {}
    if openings_path is not None:
        odata = json.loads(openings_path.read_text(encoding="utf-8"))
        for o in odata.get("openings", []):
            openings_by_plane.setdefault(o["plane_id"], []).append(o)
    # Geometric-assignment candidates (no plane-id prefilter, LESSONS L3):
    # fused windows AND -- when requested -- fused doors share this path.
    window_openings: list[dict[str, Any]] = []
    if windows_path is not None:
        wdata = json.loads(windows_path.read_text(encoding="utf-8"))
        window_openings = [
            o for o in wdata.get("openings", []) if o.get("kind") == "window"
        ]
        print(f"{len(window_openings)} fused window opening(s) loaded")
    if fused_doors_path is not None:
        ddata = json.loads(fused_doors_path.read_text(encoding="utf-8"))
        fused_doors = [
            o for o in ddata.get("openings", []) if o.get("kind") == "door"
        ]
        window_openings.extend(fused_doors)
        print(
            f"{len(fused_doors)} fused door opening(s) loaded (wide-passage "
            f"cutouts #23/#12; gates: width <= {DOOR_MAX_WIDTH_M} m, top >= "
            f"{FUSED_DOOR_MIN_TOP_ABOVE_FLOOR_M} m above floor; overlapping "
            f"G3 rects dedup on merge)"
        )
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
        edge_rects: list[list[tuple[float, float, float, float]]] | None = None
        opening_count = 0
        if (openings_by_plane or window_openings) and not fallback:
            edge_rects, opening_count = _room_edge_openings(
                inset_poly,
                edges,
                openings_by_plane,
                window_openings,
                float(floor_y),
                float(ceiling_y),
            )
        parts = room_shell_mesh(
            inset_poly, float(floor_y), float(ceiling_y), edge_rects
        )
        merged = trimesh.util.concatenate(list(parts.values()))
        merged.merge_vertices()
        tris = int(sum(len(m.faces) for m in parts.values()))
        total_tris += tris
        seethru = 0
        for name, mesh in parts.items():
            out_name = name
            if name.startswith("wall_") and len(edges) == len(ring):
                k = int(name.split("_")[1])
                if k < len(edges) and edges[k].get("occlude", True) is False:
                    # #23: see-through wall (glass closure / open passage) --
                    # the viewer skips depth-write on *seethru* meshes.
                    out_name = f"{name}_seethru"
                    seethru += 1
            scene.add_geometry(mesh, geom_name=f"{room['room_id']}/{out_name}")
        if seethru:
            print(
                f"  {room['room_id']}: {seethru} see-through wall(s) "
                "(occlude=false)"
            )
        manifest_rooms.append(
            {
                "room_id": room["room_id"],
                "storey_id": room["storey_id"],
                "triangles": tris,
                "watertight": bool(merged.is_watertight),
                "openings": opening_count,
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
        marks = ""
        if not r["watertight"] and r["openings"] == 0:
            marks += "  NOT WATERTIGHT"
        if r["openings"]:
            marks += f"  openings {r['openings']}"
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
    p_build.add_argument(
        "--openings", type=Path, default=None, help="G3 openings.json (door cutouts)"
    )
    p_build.add_argument(
        "--windows",
        type=Path,
        default=None,
        help="fusion openings_fused.json (window holes; door rows ignored)",
    )
    p_build.add_argument("--output", "-o", type=Path, default=Path("proxy_out"))
    args = parser.parse_args(argv)

    if not args.rooms.is_file():
        parser.error(f"rooms file not found: {args.rooms}")
    if args.openings is not None and not args.openings.is_file():
        parser.error(f"openings file not found: {args.openings}")
    if args.windows is not None and not args.windows.is_file():
        parser.error(f"windows file not found: {args.windows}")
    build_property(
        args.rooms,
        args.output,
        openings_path=args.openings,
        windows_path=args.windows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
