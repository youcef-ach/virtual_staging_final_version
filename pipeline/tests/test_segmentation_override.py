"""G6 slice 4.4: operator-override precedence for class maps.

The Gate-2 click-fix editor was DEFERRED by decision (OPEN_PROBLEMS #22);
the shipped correction path is a hand-authored ``{face}_cls_override.png``
that EVERY consumer must pick up through ``segmentation.resolve_class_map``.
"""

from pathlib import Path

import cv2
import numpy as np

from vs_pipeline.stages.ibl import load_emissive_mask
from vs_pipeline.stages.pose_qa import load_exclusion_mask
from vs_pipeline.stages.segmentation import CLASS_IDS, resolve_class_map


def _write(path: Path, value: int) -> None:
    img = np.full((8, 8), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_resolve_class_map_precedence_chain(tmp_path: Path) -> None:
    sweep = tmp_path / "seg" / "u1"
    sweep.mkdir(parents=True)
    _write(sweep / "px_cls.png", 1)
    _write(sweep / "px_cls_refined.png", 2)
    _write(sweep / "px_cls_override.png", 3)

    cls, src = resolve_class_map(tmp_path / "seg", "u1", "px")
    assert src == "px_cls_override.png" and int(cls[0, 0]) == 3

    (sweep / "px_cls_override.png").unlink()
    cls, src = resolve_class_map(tmp_path / "seg", "u1", "px")
    assert src == "px_cls_refined.png" and int(cls[0, 0]) == 2

    (sweep / "px_cls_refined.png").unlink()
    cls, src = resolve_class_map(tmp_path / "seg", "u1", "px")
    assert src == "px_cls.png" and int(cls[0, 0]) == 1

    (sweep / "px_cls.png").unlink()
    assert resolve_class_map(tmp_path / "seg", "u1", "px") == (None, None)


def test_override_propagates_to_gate1_and_ibl_consumers(tmp_path: Path) -> None:
    seg = tmp_path / "seg"
    sweep = seg / "u1"
    sweep.mkdir(parents=True)
    window = CLASS_IDS["window"]
    cls = np.zeros((8, 8), dtype=np.uint8)
    cls[:4, :] = window  # generated masks say: top half is a window
    assert cv2.imwrite(str(sweep / "px_cls.png"), cls)

    excl = load_exclusion_mask(seg, "u1", "px", (8, 8))
    assert excl is not None and bool(excl[:4].all())
    emis = load_emissive_mask(seg, "u1", "px")
    assert emis is not None and bool(emis[:4].all())

    # Operator override: not a window after all -> BOTH consumers must flip.
    assert cv2.imwrite(str(sweep / "px_cls_override.png"), np.zeros((8, 8), np.uint8))
    excl = load_exclusion_mask(seg, "u1", "px", (8, 8))
    assert excl is not None and not bool(excl.any())
    emis = load_emissive_mask(seg, "u1", "px")
    assert emis is not None and not bool(emis.any())
