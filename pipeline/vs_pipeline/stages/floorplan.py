"""Export floor plan data for the viewer minimap (T4).

Reads rooms.json + poses.json + navigation_graph.json and produces
``floorplan.json`` — a compact per-storey plan-view representation:
room polygons, sweep positions, nav edges, and bounding extents.

CLI:
    python -m vs_pipeline.stages.floorplan export \\
        --rooms rooms_out/rooms.json \\
        --poses poses.json \\
        --nav nav_out/navigation_graph.json \\
        -o viewer/public/data/floorplan.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def export_floorplan(
    rooms_path: Path,
    poses_path: Path,
    nav_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rooms_data = json.loads(rooms_path.read_text(encoding="utf-8"))
    poses_data = json.loads(poses_path.read_text(encoding="utf-8"))
    nav_data = json.loads(nav_path.read_text(encoding="utf-8"))

    # Sweep positions (plan-view: x, z from the 3D position).
    sweep_pos: dict[str, tuple[float, float, float]] = {}
    sweep_index: dict[str, int] = {}
    for sw in poses_data["sweeps"]:
        uid = sw["sweep_uuid"]
        m = sw["pose_rt"]
        sweep_pos[uid] = (m[0][3], m[1][3], m[2][3])
        sweep_index[uid] = sw["index"]

    # Room membership (from rooms.json sweep indices).
    index_to_uuid: dict[int, str] = {v: k for k, v in sweep_index.items()}
    room_of_sweep: dict[str, str] = {}
    for room in rooms_data["rooms"]:
        for idx in room.get("sweeps", []):
            uid = index_to_uuid.get(idx)
            if uid:
                room_of_sweep[uid] = room["room_id"]

    # Group rooms and sweeps by storey.
    storeys: dict[int, dict[str, Any]] = {}

    for room in rooms_data["rooms"]:
        rid = room["room_id"]
        try:
            si = int(rid.split("_")[0][1:])
        except (IndexError, ValueError):
            continue
        if si not in storeys:
            storeys[si] = {"storey": si, "rooms": [], "sweeps": [], "edges": []}
        poly = room.get("polygon_plan", [])
        if poly:
            storeys[si]["rooms"].append({
                "room_id": rid,
                "polygon": [[round(p[0], 3), round(p[1], 3)] for p in poly],
            })

    # Assign sweeps to their storey.
    for uid, pos in sweep_pos.items():
        rid = room_of_sweep.get(uid)
        if rid:
            try:
                si = int(rid.split("_")[0][1:])
            except (IndexError, ValueError):
                continue
        else:
            # Exterior sweeps: assign to storey by Y proximity to floor elevations.
            si = 0  # default
            best_dy = float("inf")
            for room in rooms_data["rooms"]:
                fy = room.get("floor_elevation_y")
                if fy is not None:
                    dy = abs(pos[1] - fy)
                    if dy < best_dy:
                        best_dy = dy
                        try:
                            si = int(room["room_id"].split("_")[0][1:])
                        except (IndexError, ValueError):
                            pass

        if si not in storeys:
            storeys[si] = {"storey": si, "rooms": [], "sweeps": [], "edges": []}
        storeys[si]["sweeps"].append({
            "uuid": uid,
            "index": sweep_index[uid],
            "x": round(pos[0], 3),
            "z": round(pos[2], 3),
        })

    # Nav edges (plan-view) per storey.
    for edge in nav_data.get("edges", []):
        u1, u2 = edge["from"], edge["to"]
        if u1 not in sweep_pos or u2 not in sweep_pos:
            continue
        # Assign edge to the storey of the first node.
        rid = room_of_sweep.get(u1, room_of_sweep.get(u2, ""))
        try:
            si = int(rid.split("_")[0][1:]) if rid else 0
        except (IndexError, ValueError):
            si = 0
        if si in storeys:
            storeys[si]["edges"].append({
                "from": u1,
                "to": u2,
                "portal": edge.get("portal", False),
            })

    # Compute per-storey bounding box.
    storey_list = []
    for si in sorted(storeys):
        sd = storeys[si]
        all_x: list[float] = []
        all_z: list[float] = []
        for room in sd["rooms"]:
            for p in room["polygon"]:
                all_x.append(p[0])
                all_z.append(p[1])
        for sw in sd["sweeps"]:
            all_x.append(sw["x"])
            all_z.append(sw["z"])
        if all_x and all_z:
            sd["bounds"] = {
                "min_x": round(min(all_x), 3),
                "max_x": round(max(all_x), 3),
                "min_z": round(min(all_z), 3),
                "max_z": round(max(all_z), 3),
            }
        storey_list.append(sd)

    payload = {
        "schema_version": 1,
        "storey_count": len(storey_list),
        "storeys": storey_list,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    total_rooms = sum(len(s["rooms"]) for s in storey_list)
    total_sweeps = sum(len(s["sweeps"]) for s in storey_list)
    print(f"{len(storey_list)} storeys, {total_rooms} rooms, {total_sweeps} sweeps")
    print(f"output -> {output_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T4: export floor plan for minimap.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_exp = sub.add_parser("export", help="rooms + poses + nav -> floorplan.json")
    p_exp.add_argument("--rooms", type=Path, required=True)
    p_exp.add_argument("--poses", type=Path, required=True)
    p_exp.add_argument("--nav", type=Path, required=True)
    p_exp.add_argument("--output", "-o", type=Path, default=Path("floorplan.json"))
    args = parser.parse_args(argv)

    for f, label in [(args.rooms, "rooms"), (args.poses, "poses"), (args.nav, "nav")]:
        if not f.is_file():
            parser.error(f"{label} file not found: {f}")

    export_floorplan(args.rooms, args.poses, args.nav, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
