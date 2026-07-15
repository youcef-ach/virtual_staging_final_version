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
    # torch/transformers are lazy-imported inside Mask2FormerRunner only.
    import vs_pipeline.stages.segmentation as seg

    assert hasattr(seg, "Mask2FormerRunner")
