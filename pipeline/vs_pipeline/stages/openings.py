"""Stage 6 (G3) slice 3 -- door + passage opening detection from wall boundaries.

Doors reach the floor, so they notch the EXTERNAL contour that G2 stores as
``boundary_2d``: a jamb-bounded run of grid columns whose lowest wall inlier
sits well above the wall bottom, with wall remaining above (the lintel).
Detected rects are emitted per wall plane in the plane's (u, v) frame plus
world-space corners; the proxy stage cuts them from wall faces (runtime
occlusion must not block the view through doorways).

WIDE-GAP PASSAGE MODE (#12/#23/#26, added 2026-07-26): doorless open
passages (e.g. the kitchen<->living entrance) carry NO door-class pixels --
segmentation sees the next room's content through them, so the fused-door
path can never detect them (verified on the property: zero fused rects on
the z~0.4 passage wall). The evidence that DOES exist is geometric: the
passage is a floor-reaching occupancy gap in the wall's own boundary, wider
than a door. Runs 1.4-3.5 m wide pass with the same clear/top gates plus an
EXPLICIT jamb requirement (both neighbor columns must hold wall reaching
within 0.8 m of the wall bottom); they emit as kind=door with
``"passage": true`` so downstream consumption (proxy floor-reaching
cutouts) is unchanged.

Windows are interior holes -- invisible to the external contour -- and land
via G6 fusion masks (see fusion.py / proxy --windows).

CLI:
    python -m vs_pipeline.stages.openings detect \
        --planes planes_full/structural_planes.json -o openings_out/openings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

OPENINGS_SCHEMA_VERSION = 1

GRID_M = 0.05
#: Walls too short/narrow cannot host a door (also filters fragments).
MIN_WALL_HEIGHT_M = 1.8
MIN_WALL_WIDTH_M = 1.2
DOOR_MIN_WIDTH_M = 0.55
DOOR_MAX_WIDTH_M = 1.4
#: A column is door-clear when its lowest inlier sits at least this far
#: above the wall bottom (standard doors clear ~2.0 m).
DOOR_MIN_CLEAR_M = 1.6
DOOR_TOP_MAX_M = 2.6
#: Wide-gap passage mode: doorless openings up to this width qualify when
#: BOTH neighbor columns are real jambs (wall reaching near the bottom).
PASSAGE_MAX_WIDTH_M = 3.5
PASSAGE_JAMB_MAX_CLEAR_M = 0.8


def _opening_dict(
    plane: dict[str, Any],
    ua: float,
    ub: float,
    vb: float,
    vt: float,
    passage: bool = False,
) -> dict[str, Any]:
    frame = plane["frame"]
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u_axis = np.asarray(frame["u"], dtype=np.float64)
    v_axis = np.asarray(frame["v"], dtype=np.float64)
    corners = [
        origin + u_axis * u + v_axis * v
        for u, v in ((ua, vb), (ub, vb), (ub, vt), (ua, vt))
    ]
    out = {
        "plane_id": plane["plane_id"],
        "kind": "door",
        "u_range": [round(ua, 4), round(ub, 4)],
        "v_range": [round(float(vb), 4), round(float(vt), 4)],
        "width_m": round(ub - ua, 3),
        "height_m": round(float(vt - vb), 3),
        "corners_3d": [[round(float(c), 4) for c in p] for p in corners],
    }
    if passage:
        out["passage"] = True
    return out


def detect_wall_openings(plane: dict[str, Any]) -> list[dict[str, Any]]:
    """Door/passage rects for one wall plane from its boundary_2d notches."""
    frame = plane.get("frame")
    b2 = plane.get("boundary_2d") or []
    if not frame or len(b2) < 4:
        return []
    poly = Polygon([tuple(p) for p in b2])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return []
    u0, v0, u1, v1 = poly.bounds
    if v1 - v0 < MIN_WALL_HEIGHT_M or u1 - u0 < MIN_WALL_WIDTH_M:
        return []
    nu = max(int(round((u1 - u0) / GRID_M)), 1)
    nv = max(int(round((v1 - v0) / GRID_M)), 1)
    us = u0 + (np.arange(nu) + 0.5) * GRID_M
    vs = v0 + (np.arange(nv) + 0.5) * GRID_M
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    filled = contains_xy(poly, uu.ravel(), vv.ravel()).reshape(nu, nv)

    lowest = np.full(nu, np.inf)
    for i in range(nu):
        idx = np.flatnonzero(filled[i])
        if idx.size:
            lowest[i] = float(vs[idx[0]] - v0)
    door_col = np.isfinite(lowest) & (lowest >= DOOR_MIN_CLEAR_M)

    def _is_jamb(col: int) -> bool:
        """Real jamb: wall in this column reaches near the wall bottom."""
        return bool(
            np.isfinite(lowest[col]) and lowest[col] <= PASSAGE_JAMB_MAX_CLEAR_M
        )

    openings: list[dict[str, Any]] = []
    i = 0
    while i < nu:
        if not door_col[i]:
            i += 1
            continue
        j = i
        while j + 1 < nu and door_col[j + 1]:
            j += 1
        # Runs touching the wall ends are ambiguous (wall end, not a door).
        if i > 0 and j < nu - 1:
            width = (j - i + 1) * GRID_M
            top = float(lowest[i : j + 1].min())
            if top <= DOOR_TOP_MAX_M:
                if DOOR_MIN_WIDTH_M <= width <= DOOR_MAX_WIDTH_M:
                    openings.append(
                        _opening_dict(
                            plane,
                            float(us[i] - GRID_M / 2),
                            float(us[j] + GRID_M / 2),
                            float(v0),
                            float(v0 + top),
                        )
                    )
                elif DOOR_MAX_WIDTH_M < width <= PASSAGE_MAX_WIDTH_M and _is_jamb(
                    i - 1
                ) and _is_jamb(j + 1):
                    # Wide doorless passage: at this width 'not touching the
                    # wall end' alone is too weak -- demand REAL jambs on
                    # both sides (wall reaching near the bottom).
                    openings.append(
                        _opening_dict(
                            plane,
                            float(us[i] - GRID_M / 2),
                            float(us[j] + GRID_M / 2),
                            float(v0),
                            float(v0 + top),
                            passage=True,
                        )
                    )
        i = j + 1
    return openings


def detect_property(planes_path: Path, output_path: Path) -> dict[str, Any]:
    data = json.loads(planes_path.read_text(encoding="utf-8"))
    openings: list[dict[str, Any]] = []
    walls = 0
    for p in data["planes"]:
        if p["class"] != "wall":
            continue
        walls += 1
        openings.extend(detect_wall_openings(p))
    payload = {
        "schema_version": OPENINGS_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "opening_count": len(openings),
        "openings": openings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    passages = [o for o in openings if o.get("passage")]
    if openings:
        widths = [o["width_m"] for o in openings]
        heights = [o["height_m"] for o in openings]
        print(
            f"{len(openings)} opening(s) across {walls} wall plane(s) -- "
            f"{len(openings) - len(passages)} door(s), "
            f"{len(passages)} wide passage(s)  "
            f"(width {min(widths):.2f}-{max(widths):.2f} m, "
            f"height {min(heights):.2f}-{max(heights):.2f} m)"
        )
        for o in passages:
            c = np.asarray(o["corners_3d"], dtype=np.float64).mean(axis=0)
            print(
                f"  passage on {o['plane_id']}: {o['width_m']:.2f} x "
                f"{o['height_m']:.2f} m at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
            )
    else:
        print(f"no openings detected across {walls} wall plane(s)")
    print(f"output -> {output_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 6 (G3) slice 3: door + passage opening detection."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_detect = sub.add_parser("detect", help="structural_planes.json -> openings.json")
    p_detect.add_argument(
        "--planes", type=Path, required=True, help="structural_planes.json"
    )
    p_detect.add_argument(
        "--output", "-o", type=Path, default=Path("openings_out/openings.json")
    )
    args = parser.parse_args(argv)

    if not args.planes.is_file():
        parser.error(f"planes file not found: {args.planes}")
    detect_property(args.planes, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
