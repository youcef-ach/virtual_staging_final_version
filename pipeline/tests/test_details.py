"""Tests for Stage 6 (G3) final slice: non-planar detail proxies."""

from __future__ import annotations

import numpy as np

from vs_pipeline.stages.details import (
    build_details,
    filter_clusters,
    merge_cluster_boxes,
)


def _cluster(cid, bmin, bmax, points=500):
    bmin = [float(v) for v in bmin]
    bmax = [float(v) for v in bmax]
    return {
        "cluster_id": cid,
        "point_count": points,
        "centroid": [(a + b) / 2 for a, b in zip(bmin, bmax)],
        "bbox_min": bmin,
        "bbox_max": bmax,
    }


def _rooms_payload():
    return {
        "storeys": [{"storey_id": 0, "floor_y": 0.0, "ceiling_y": 2.7}],
        "rooms": [
            {
                "room_id": "s0_room_00",
                "storey_id": 0,
                "polygon_plan": [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            }
        ],
    }


def test_filter_keeps_interior_furniture_scale_only():
    from shapely.geometry import Polygon

    room = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    inside = _cluster("a", [1, 0, 1], [2, 0.8, 2])
    outside = _cluster("b", [10, 0, 10], [11, 0.8, 11])
    too_small = _cluster("c", [1, 0, 1], [1.05, 0.1, 1.05])
    too_sparse = _cluster("d", [1, 0, 1], [2, 0.8, 2], points=10)
    kept = filter_clusters([inside, outside, too_small, too_sparse], room)
    assert [c["cluster_id"] for c in kept] == ["a"]


def test_overlapping_boxes_merge():
    a = _cluster("a", [1, 0, 1], [2, 1, 2])
    b = _cluster("b", [1.9, 0, 1.9], [3, 1, 3])  # overlaps a
    c = _cluster("c", [3.5, 0, 0.5], [3.9, 1, 1.0])  # separate
    boxes = merge_cluster_boxes([a, b, c])
    assert len(boxes) == 2
    merged = max(boxes, key=lambda t: t[2])
    assert merged[2] == 2
    assert np.allclose(merged[0], [1, 0, 1])
    assert np.allclose(merged[1], [3, 1, 3])


def test_clip_end_to_end_produces_detail():
    import trimesh

    # a 1 m box sitting inside the room, embedded in a larger scene mesh
    furniture = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    furniture.apply_translation((1.5, 0.5, 1.5))
    far_away = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    far_away.apply_translation((50.0, 0.5, 50.0))
    scene_mesh = trimesh.util.concatenate([furniture, far_away])

    cluster = _cluster("a", [1.0, 0.0, 1.0], [2.0, 1.0, 2.0])
    manifest, meshes, boxes = build_details(
        [cluster], _rooms_payload(), scene_mesh
    )
    assert manifest["detail_count"] == 1
    assert manifest["details"][0]["triangles"] == 12  # only the furniture box
    assert len(meshes) == 1 and len(boxes) == 1
