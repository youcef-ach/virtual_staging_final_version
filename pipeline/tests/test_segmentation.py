"""Tests for Stage 4 (G6) slice 1: class mapping + remap + idempotency (torch-free)."""

from __future__ import annotations

import json

import numpy as np

from vs_pipeline.stages.segmentation import (
    CLASS_IDS,
    CLASSES,
    FACE_NAMES,
    SegParams,
    _outputs_current,
    build_class_lut,
    class_overlay,
    classify_ade_label,
    remap_labels,
)


def test_classify_ade_label_core_classes():
    assert classify_ade_label("wall") == CLASS_IDS["wall"]
    assert classify_ade_label("floor;flooring") == CLASS_IDS["floor"]
    assert classify_ade_label("rug") == CLASS_IDS["floor"]
    assert classify_ade_label("ceiling") == CLASS_IDS["ceiling"]
    assert classify_ade_label("windowpane;window") == CLASS_IDS["window"]
    assert classify_ade_label("door;double door") == CLASS_IDS["door"]
    assert classify_ade_label("sofa;couch;lounge") == CLASS_IDS["furniture"]
    assert classify_ade_label("kitchen island") == CLASS_IDS["furniture"]
    assert classify_ade_label("sky") == CLASS_IDS["other"]
    # Mirrors deliberately stay 'other' (Gate-1 reads mirror from the RAW map).
    assert classify_ade_label("mirror") == CLASS_IDS["other"]
    assert classify_ade_label("stairs;steps") == CLASS_IDS["other"]


def test_lut_and_remap():
    id2label = {0: "wall", 1: "floor", 2: "windowpane;window", 3: "sofa", 4: "sky"}
    lut = build_class_lut(id2label)
    ade = np.array([[0, 1], [2, 3], [4, 4]], dtype=np.uint8)
    cls = remap_labels(ade, lut)
    expect = np.array(
        [
            [CLASS_IDS["wall"], CLASS_IDS["floor"]],
            [CLASS_IDS["window"], CLASS_IDS["furniture"]],
            [CLASS_IDS["other"], CLASS_IDS["other"]],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cls, expect)


def test_class_overlay_shape_and_dtype():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    cls = np.full((8, 8), CLASS_IDS["window"], dtype=np.uint8)
    out = class_overlay(rgb, cls)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert out[0, 0].sum() > 0  # window color blended in


def test_classes_are_the_locked_seven():
    assert CLASSES == ("other", "floor", "wall", "ceiling", "window", "door", "furniture")


def test_outputs_current_idempotency(tmp_path):
    params = SegParams(model_name="m", mask_size=64)
    sweep_out = tmp_path / "sweep-a"
    sweep_out.mkdir()
    meta_path = sweep_out / "seg_meta.json"
    files = {"faces": {}}
    for face in FACE_NAMES:
        for kind in ("ade", "cls"):
            (sweep_out / f"{face}_{kind}.png").write_bytes(b"x")
        files["faces"][face] = {"ade": f"{face}_ade.png", "cls": f"{face}_cls.png"}
    meta_path.write_text(
        json.dumps({"params": params.to_dict(), "files": files}), encoding="utf-8"
    )
    assert _outputs_current(meta_path, sweep_out, params)
    # Different params (e.g. Swin-L upgrade) -> must re-infer.
    assert not _outputs_current(meta_path, sweep_out, SegParams(model_name="L", mask_size=64))
    # Missing file -> must re-infer.
    (sweep_out / "px_cls.png").unlink()
    assert not _outputs_current(meta_path, sweep_out, params)


def test_module_imports_without_torch():
    # The stage module must stay importable without the [segmentation] extra;
    # torch/transformers/segment-anything are lazy-imported inside the runners.
    import vs_pipeline.stages.segmentation as seg

    assert hasattr(seg, "Mask2FormerRunner")
    assert hasattr(seg, "SamRefiner")


def test_extract_instances_boxes_and_min_area():
    from vs_pipeline.stages.segmentation import extract_instances

    cls = np.zeros((32, 32), dtype=np.uint8)
    cls[2:10, 2:10] = CLASS_IDS["window"]  # 64 px component
    cls[20:22, 20:22] = CLASS_IDS["window"]  # 4 px -> below min_area
    cls[12:18, 14:26] = CLASS_IDS["furniture"]  # 72 px component
    inst = extract_instances(cls, min_area=10)
    assert len(inst) == 2
    by_class = {i["class_id"]: i for i in inst}
    assert by_class[CLASS_IDS["window"]]["box"] == (2, 2, 10, 10)
    assert by_class[CLASS_IDS["furniture"]]["area"] == 72


def test_mask_iou():
    from vs_pipeline.stages.segmentation import mask_iou

    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[0:4, 0:4] = True
    b[2:6, 0:4] = True
    assert abs(mask_iou(a, b) - (8.0 / 24.0)) < 1e-9
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 0.0


def test_merge_refined_paint_order():
    from vs_pipeline.stages.segmentation import merge_refined

    cls = np.zeros((8, 8), dtype=np.uint8)
    big = np.zeros((8, 8), dtype=bool)
    big[0:6, 0:6] = True
    small = np.zeros((8, 8), dtype=bool)
    small[2:4, 2:4] = True
    refined = merge_refined(
        cls,
        [(CLASS_IDS["door"], small), (CLASS_IDS["window"], big)],
    )
    # Large painted first, small stays on top regardless of input order.
    assert refined[0, 0] == CLASS_IDS["window"]
    assert refined[3, 3] == CLASS_IDS["door"]
    assert refined[7, 7] == 0


def test_inverse_transform_face_inverts_forward():
    from vs_pipeline.cubemap_transforms import AXES, inverse_transform_face, transform_face

    rng = np.random.default_rng(7)
    for src in AXES:
        arr = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
        out, out_axis = transform_face(arr, src)
        rec, src_axis = inverse_transform_face(out, out_axis)
        assert src_axis == src
        assert np.array_equal(rec, arr)


def test_upright_round_trip_all_faces():
    from vs_pipeline.stages.segmentation import stored_from_upright, upright_from_stored

    rng = np.random.default_rng(11)
    for face in FACE_NAMES:
        arr = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
        up = upright_from_stored(arr, face)
        back = stored_from_upright(up, face)
        assert np.array_equal(back, arr)
        # Uprighting must actually change rotated faces (not be a no-op everywhere).
    changed = any(
        not np.array_equal(
            upright_from_stored(np.arange(64, dtype=np.uint8).reshape(8, 8), face),
            np.arange(64, dtype=np.uint8).reshape(8, 8),
        )
        for face in FACE_NAMES
    )
    assert changed
