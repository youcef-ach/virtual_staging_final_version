"""Stage 9 (T1) -- navigation graph: sweep-to-sweep connectivity.

Nodes = sweeps (UUID + position from poses.json).
Edges = two types:
  * INTRA-ROOM: every pair of sweeps assigned to the same room by rooms.json
    (they trivially see each other inside one room).  Weight = Euclidean dist.
  * INTER-ROOM (portal): for each opening (door/passage/window) that separates
    two rooms, connect the nearest sweep in each room.  Flagged ``portal: true``
    with the opening kind and plane id.

No ray casting: room adjacency through openings IS line-of-sight (the proxy has
holes there).  The PLAN's ≤ 10 m cap applies to portal edges only (intra-room
distances are bounded by room size).

Also emits a per-storey top-down review render (sweep nodes + edges) in the
style of rooms_review PNGs.

CLI:
    python -m vs_pipeline.stages.navigation build \\
        --poses poses.json \\
        --rooms rooms_out/rooms.json \\
        --openings openings_out/openings.json \\
        [--fused fusion_out/openings_fused.json] \\
        -o nav_out/navigation_graph.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
MAX_PORTAL_DISTANCE_M = 10.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _position_from_pose(pose_rt: list[list[float]]) -> tuple[float, float, float]:
    """Extract the translation column from a 4×4 row-major pose matrix."""
    return (pose_rt[0][3], pose_rt[1][3], pose_rt[2][3])


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _centroid_3d(corners: list[list[float]]) -> tuple[float, float, float]:
    arr = np.asarray(corners, dtype=np.float64)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


# ---------------------------------------------------------------------------
# core builder
# ---------------------------------------------------------------------------

def build_navigation_graph(
    poses_path: Path,
    rooms_path: Path,
    openings_path: Path,
    fused_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Build the navigation graph and write to *output_path*."""

    # -- load data -----------------------------------------------------------
    poses_data = json.loads(poses_path.read_text(encoding="utf-8"))
    rooms_data = json.loads(rooms_path.read_text(encoding="utf-8"))
    openings_data = json.loads(openings_path.read_text(encoding="utf-8"))
    fused_data = (
        json.loads(fused_path.read_text(encoding="utf-8"))
        if fused_path and fused_path.is_file()
        else None
    )

    # -- nodes: sweep positions ----------------------------------------------
    sweep_pos: dict[str, tuple[float, float, float]] = {}
    sweep_index: dict[str, int] = {}
    for sw in poses_data["sweeps"]:
        uid = sw["sweep_uuid"]
        sweep_pos[uid] = _position_from_pose(sw["pose_rt"])
        sweep_index[uid] = sw["index"]

    # -- room membership -----------------------------------------------------
    # rooms.json stores sweep INDICES; convert to UUIDs.
    index_to_uuid: dict[int, str] = {v: k for k, v in sweep_index.items()}
    room_of_sweep: dict[str, str] = {}  # uuid -> room_id
    room_sweeps: dict[str, list[str]] = {}  # room_id -> [uuid]
    room_storey: dict[str, int] = {}  # room_id -> storey int

    for room in rooms_data["rooms"]:
        rid = room["room_id"]
        uuids: list[str] = []
        for idx in room.get("sweeps", []):
            uid = index_to_uuid.get(idx)
            if uid:
                room_of_sweep[uid] = rid
                uuids.append(uid)
        room_sweeps[rid] = uuids
        # Parse storey from room_id: "s1_room_03" -> 1
        try:
            room_storey[rid] = int(rid.split("_")[0][1:])
        except (IndexError, ValueError):
            room_storey[rid] = 0

    # Sweeps not in any room → "exterior" group
    exterior_uuids = [u for u in sweep_pos if u not in room_of_sweep]
    if exterior_uuids:
        room_sweeps["exterior"] = exterior_uuids
        for u in exterior_uuids:
            room_of_sweep[u] = "exterior"

    # -- room adjacency through openings ------------------------------------
    # For each room, collect the wall_planes it uses.
    room_walls: dict[str, set[str]] = {}
    for room in rooms_data["rooms"]:
        rid = room["room_id"]
        planes: set[str] = set()
        for edge in room.get("edges", []):
            for pid in edge.get("wall_planes", []):
                planes.add(pid)
        room_walls[rid] = planes

    # Invert: plane_id -> set of room_ids
    plane_to_rooms: dict[str, set[str]] = defaultdict(set)
    for rid, planes in room_walls.items():
        for pid in planes:
            plane_to_rooms[pid].add(rid)

    # -- build edges ---------------------------------------------------------
    edges: list[dict[str, Any]] = []
    edge_set: set[tuple[str, str]] = set()  # dedup (sorted pair)

    def _add_edge(
        u1: str,
        u2: str,
        portal: bool = False,
        room: str | None = None,
        opening_kind: str | None = None,
        opening_plane: str | None = None,
    ) -> None:
        pair = (min(u1, u2), max(u1, u2))
        if pair in edge_set:
            return
        d = _dist(sweep_pos[u1], sweep_pos[u2])
        if portal and d > MAX_PORTAL_DISTANCE_M:
            return
        edge_set.add(pair)
        e: dict[str, Any] = {
            "from": u1,
            "to": u2,
            "distance_m": round(d, 3),
            "portal": portal,
        }
        if room:
            e["room"] = room
        if opening_kind:
            e["opening_kind"] = opening_kind
        if opening_plane:
            e["opening_plane"] = opening_plane
        edges.append(e)

    # 1. Intra-room: all-pairs within each room
    for rid, uuids in room_sweeps.items():
        if rid == "exterior":
            # For exterior sweeps, connect only nearest neighbors (≤ 5 m)
            # to avoid a dense clique of 50+ outdoor sweeps.
            for i, u1 in enumerate(uuids):
                for u2 in uuids[i + 1 :]:
                    if _dist(sweep_pos[u1], sweep_pos[u2]) <= 5.0:
                        _add_edge(u1, u2, room="exterior")
        else:
            for i, u1 in enumerate(uuids):
                for u2 in uuids[i + 1 :]:
                    _add_edge(u1, u2, room=rid)

    # 2. Portal edges: connect rooms through openings
    all_openings: list[dict[str, Any]] = list(openings_data.get("openings", []))
    if fused_data:
        for o in fused_data.get("openings", []):
            all_openings.append(o)

    for opening in all_openings:
        pid = opening.get("plane_id", "")
        kind = opening.get("kind", "unknown")
        corners = opening.get("corners_3d", [])
        if not corners:
            continue

        # Which rooms does this plane separate?
        adjacent_rooms = list(plane_to_rooms.get(pid, set()))
        if len(adjacent_rooms) < 2:
            # This opening is on a wall that bounds only one room (exterior
            # wall, or the other side is unpolygonized).  Still useful:
            # connect interior sweep to the nearest exterior sweep through it.
            if len(adjacent_rooms) == 1 and exterior_uuids:
                centroid = _centroid_3d(corners)
                rid = adjacent_rooms[0]
                interior = room_sweeps.get(rid, [])
                if not interior:
                    continue
                best_in = min(interior, key=lambda u: _dist(sweep_pos[u], centroid))
                best_ext = min(
                    exterior_uuids, key=lambda u: _dist(sweep_pos[u], centroid)
                )
                _add_edge(
                    best_in,
                    best_ext,
                    portal=True,
                    opening_kind=kind,
                    opening_plane=pid,
                )
            continue

        # Connect nearest sweep in each adjacent room pair
        for i, r1 in enumerate(adjacent_rooms):
            for r2 in adjacent_rooms[i + 1 :]:
                sweeps1 = room_sweeps.get(r1, [])
                sweeps2 = room_sweeps.get(r2, [])
                if not sweeps1 or not sweeps2:
                    continue
                centroid = _centroid_3d(corners)
                best1 = min(sweeps1, key=lambda u: _dist(sweep_pos[u], centroid))
                best2 = min(sweeps2, key=lambda u: _dist(sweep_pos[u], centroid))
                _add_edge(
                    best1,
                    best2,
                    portal=True,
                    opening_kind=kind,
                    opening_plane=pid,
                )

    # 3. Stair / multi-storey: connect rooms on different storeys that are
    # spatially close (stairwell sweeps, or rooms whose plan bbox overlaps
    # vertically).  Simple heuristic: for each room pair on different storeys,
    # if the nearest sweep pair is within 6 m, add a portal edge.
    room_ids = [r for r in room_sweeps if r != "exterior"]
    for i, r1 in enumerate(room_ids):
        for r2 in room_ids[i + 1 :]:
            s1 = room_storey.get(r1, 0)
            s2 = room_storey.get(r2, 0)
            if s1 == s2:
                continue
            sweeps1 = room_sweeps.get(r1, [])
            sweeps2 = room_sweeps.get(r2, [])
            if not sweeps1 or not sweeps2:
                continue
            # Find the closest pair across the two rooms
            best_d = float("inf")
            best_u1 = best_u2 = ""
            for u1 in sweeps1:
                for u2 in sweeps2:
                    d = _dist(sweep_pos[u1], sweep_pos[u2])
                    if d < best_d:
                        best_d = d
                        best_u1 = u1
                        best_u2 = u2
            if best_d <= 6.0:
                _add_edge(best_u1, best_u2, portal=True, opening_kind="stair")

    # -- assemble output -----------------------------------------------------
    nodes = []
    for uid in sorted(sweep_pos, key=lambda u: sweep_index[u]):
        pos = sweep_pos[uid]
        n: dict[str, Any] = {
            "uuid": uid,
            "index": sweep_index[uid],
            "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
        }
        room = room_of_sweep.get(uid)
        if room:
            n["room"] = room
        nodes.append(n)

    edges.sort(key=lambda e: e["distance_m"])

    # -- graph connectivity stats -------------------------------------------
    # Build adjacency for reachability check
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])
    visited: set[str] = set()
    components = 0
    isolated: list[str] = []
    for uid in sweep_pos:
        if uid not in visited:
            components += 1
            stack = [uid]
            comp: list[str] = []
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                stack.extend(adj[u])
            if len(comp) == 1:
                isolated.append(uid)

    portal_edges = [e for e in edges if e["portal"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "portal_count": len(portal_edges),
        "connected_components": components,
        "nodes": nodes,
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # -- console report ------------------------------------------------------
    rooms_with_sweeps = sum(1 for uuids in room_sweeps.values() if uuids)
    print(
        f"{len(nodes)} nodes, {len(edges)} edges "
        f"({len(portal_edges)} portal, {len(edges) - len(portal_edges)} intra-room)"
    )
    print(
        f"{rooms_with_sweeps} rooms with sweeps, "
        f"{len(exterior_uuids)} exterior sweeps"
    )
    print(f"{components} connected component(s), {len(isolated)} isolated node(s)")
    if isolated:
        for uid in isolated[:5]:
            print(f"  isolated: #{sweep_index[uid]} {uid[:8]} room={room_of_sweep.get(uid, '?')}")
        if len(isolated) > 5:
            print(f"  ... and {len(isolated) - 5} more")
    if edges:
        dists = [e["distance_m"] for e in edges]
        print(
            f"distance range: {min(dists):.2f} – {max(dists):.2f} m "
            f"(mean {sum(dists)/len(dists):.2f} m)"
        )
    print(f"output -> {output_path}")
    return payload


# ---------------------------------------------------------------------------
# review render
# ---------------------------------------------------------------------------

def _render_review(
    payload: dict[str, Any],
    rooms_path: Path,
    output_dir: Path,
) -> None:
    """Per-storey top-down review renders: nodes + edges on a grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- review renders skipped")
        return

    rooms_data = json.loads(rooms_path.read_text(encoding="utf-8"))

    # Group nodes by storey (from room_id "sN_room_NN" -> storey N)
    storey_nodes: dict[int, list[dict]] = defaultdict(list)
    for node in payload["nodes"]:
        room = node.get("room", "exterior")
        try:
            s = int(room.split("_")[0][1:]) if room != "exterior" else -1
        except (IndexError, ValueError):
            s = -1
        storey_nodes[s].append(node)

    # Collect storeys that have rooms
    storeys = sorted(s for s in storey_nodes if s >= 0)
    if not storeys:
        return

    for si in storeys:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_title(f"Navigation graph -- storey {si}", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.grid(True, alpha=0.3)

        # Draw room polygons (light fill)
        for room in rooms_data["rooms"]:
            rid = room["room_id"]
            try:
                rs = int(rid.split("_")[0][1:])
            except (IndexError, ValueError):
                continue
            if rs != si:
                continue
            poly = room.get("polygon_plan", [])
            if poly:
                xs = [p[0] for p in poly] + [poly[0][0]]
                zs = [p[1] for p in poly] + [poly[0][1]]
                ax.fill(xs, zs, alpha=0.08, color="blue")
                ax.plot(xs, zs, color="blue", alpha=0.3, linewidth=0.8)

        # Build a position lookup
        pos_map = {n["uuid"]: n["position"] for n in payload["nodes"]}
        node_rooms = {n["uuid"]: n.get("room", "") for n in payload["nodes"]}

        # Draw edges
        for edge in payload["edges"]:
            u1 = edge["from"]
            u2 = edge["to"]
            if u1 not in pos_map or u2 not in pos_map:
                continue
            p1 = pos_map[u1]
            p2 = pos_map[u2]
            r1 = node_rooms.get(u1, "")
            r2 = node_rooms.get(u2, "")
            # Only draw edges relevant to this storey
            try:
                s1 = int(r1.split("_")[0][1:]) if r1 and r1 != "exterior" else -1
                s2 = int(r2.split("_")[0][1:]) if r2 and r2 != "exterior" else -1
            except (IndexError, ValueError):
                continue
            if s1 != si and s2 != si:
                continue
            if edge["portal"]:
                ax.plot(
                    [p1[0], p2[0]], [p1[2], p2[2]],
                    color="red", linewidth=1.5, alpha=0.7, zorder=2,
                )
            else:
                ax.plot(
                    [p1[0], p2[0]], [p1[2], p2[2]],
                    color="gray", linewidth=0.6, alpha=0.4, zorder=1,
                )

        # Draw nodes
        for node in storey_nodes.get(si, []):
            p = node["position"]
            ax.plot(p[0], p[2], "o", color="green", markersize=5, zorder=3)
            ax.annotate(
                str(node["index"]),
                (p[0], p[2]),
                fontsize=6,
                ha="center",
                va="bottom",
                color="darkgreen",
            )

        out = output_dir / f"nav_review_storey{si}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"review -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 9 (T1): navigation graph."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="poses + rooms + openings -> navigation_graph.json")
    p_build.add_argument("--poses", type=Path, required=True, help="poses.json")
    p_build.add_argument("--rooms", type=Path, required=True, help="rooms.json")
    p_build.add_argument("--openings", type=Path, required=True, help="openings.json")
    p_build.add_argument(
        "--fused", type=Path, default=None, help="openings_fused.json (optional)"
    )
    p_build.add_argument(
        "--output", "-o", type=Path, default=Path("nav_out/navigation_graph.json")
    )
    args = parser.parse_args(argv)

    for f, label in [(args.poses, "poses"), (args.rooms, "rooms"), (args.openings, "openings")]:
        if not f.is_file():
            parser.error(f"{label} file not found: {f}")
    if args.fused and not args.fused.is_file():
        parser.error(f"fused file not found: {args.fused}")

    payload = build_navigation_graph(
        args.poses, args.rooms, args.openings, args.fused, args.output,
    )
    _render_review(payload, args.rooms, args.output.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
