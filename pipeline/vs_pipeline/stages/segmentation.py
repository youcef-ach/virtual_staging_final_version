"""Stage 4 (G6) -- semantic segmentation, slice 1: per-face ADE20K -> 7-class masks.

Segmentation runs on the extracted 90-deg pinhole faces -- exactly the
perspective imagery Mask2Former is trained on (6 faces x 198 sweeps ~= 1,188
images). SAM boundary refinement and 2D->3D fusion (G-buffer depth lift ->
window/door openings + 3D furniture boxes) land in later slices.

Hardware-driven variant policy (locked 2026-07-15, docs/OPEN_PROBLEMS.md #20):
the model variant is a PARAMETER. The 4 GB dev GPU (GTX 1650-class) runs
Mask2Former Swin-T + fp16 with faces resized to 640 px; production GPU workers
(P3) swap to Swin-L via ``--model`` with zero code change. Hugging Face
``transformers`` is used deliberately -- no detectron2 build pain on Windows.

Both the RAW ADE20K label map and the remapped 7-class map are stored per
face: remapping is lossless and re-runnable without re-inference, and
downstream consumers can read raw labels the 7-class map collapses (e.g.
Gate-1 glass exclusion wants ``windowpane`` UNION ``mirror``, while the
7-class map deliberately sends ``mirror`` to ``other``).

CLI:
    python -m vs_pipeline.stages.segmentation infer --panos extracted_cubemaps \
        -o seg_out --sweeps "1,50,60,175"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vs_pipeline.cubemap_transforms import (
    FACE_OUTPUT_AXIS,
    inverse_transform_face,
    transform_face,
)

SEG_SCHEMA_VERSION = 1
FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")

#: face name <-> cubemap axis (stored/canonical convention).
_FACE_TO_AXIS = {"px": "+X", "nx": "-X", "py": "+Y", "ny": "-Y", "pz": "+Z", "nz": "-Z"}
_SOURCE_FOR_OUTPUT = {v: k for k, v in FACE_OUTPUT_AXIS.items()}


def upright_from_stored(arr: np.ndarray, face: str) -> np.ndarray:
    """Stored canonical face -> upright (gravity-down) photo orientation.

    The stored faces carry the Set A canonical transforms (a Z-up cubemap
    layout the VIEWER corrects at runtime via the F4.4 reflection matrix) --
    several faces are rotated/flipped and are NOT upright photos.
    Segmentation models assume gravity-down images, so faces are uprighted
    before inference and label maps are mapped back after. Pure rot90/flip
    permutations: lossless both ways, masks stay pixel-aligned with the
    stored extracted faces (required by the F5-calibration fusion path).
    """
    out, _src = inverse_transform_face(np.asarray(arr), _FACE_TO_AXIS[face])
    return np.ascontiguousarray(out)


def stored_from_upright(arr: np.ndarray, face: str) -> np.ndarray:
    """Inverse of :func:`upright_from_stored` (labels back to the stored layout)."""
    src = _SOURCE_FOR_OUTPUT[_FACE_TO_AXIS[face]]
    out, out_axis = transform_face(np.asarray(arr), src)
    assert out_axis == _FACE_TO_AXIS[face]
    return np.ascontiguousarray(out)


DEFAULT_MODEL = "facebook/mask2former-swin-tiny-ade-semantic"
DEFAULT_MASK_SIZE = 640

#: Our 7 classes (PLAN decision 4). Index = stored value in ``{face}_cls.png``.
CLASSES = ("other", "floor", "wall", "ceiling", "window", "door", "furniture")
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

CLASS_COLORS = np.array(
    [
        [128, 128, 128],  # other      gray
        [80, 200, 80],  # floor      green
        [70, 110, 230],  # wall       blue
        [90, 220, 220],  # ceiling    cyan
        [250, 220, 60],  # window     yellow
        [245, 140, 40],  # door       orange
        [220, 80, 200],  # furniture  magenta
    ],
    dtype=np.uint8,
)

# ADE20K label-name -> class sets (matched on the first ';'-separated synonym,
# lowercased). NOTE: 'mirror' stays 'other' ON PURPOSE -- mirrors are not
# windows for occlusion/openings, but Gate-1 exclusion reads mirror+windowpane
# from the RAW ADE map, so nothing is lost.
_FLOOR_NAMES = {"floor", "rug"}
_WALL_NAMES = {"wall"}
_CEILING_NAMES = {"ceiling"}
_WINDOW_NAMES = {"windowpane", "window"}
_DOOR_NAMES = {"door", "screen door"}
_FURNITURE_NAMES = {
    "bed",
    "cabinet",
    "table",
    "chair",
    "sofa",
    "shelf",
    "armchair",
    "seat",
    "desk",
    "wardrobe",
    "lamp",
    "bathtub",
    "cushion",
    "pillow",
    "chest of drawers",
    "counter",
    "countertop",
    "sink",
    "refrigerator",
    "pool table",
    "bookcase",
    "coffee table",
    "toilet",
    "bench",
    "stove",
    "oven",
    "microwave",
    "dishwasher",
    "washer",
    "kitchen island",
    "swivel chair",
    "ottoman",
    "buffet",
    "stool",
    "cradle",
    "television receiver",
    "television",
    "computer",
    "monitor",
    "radiator",
    "fireplace",
    "chandelier",
    "sconce",
}


def classify_ade_label(name: str) -> int:
    """ADE20K label name -> our class id (first synonym, lowercased)."""
    key = name.split(";")[0].strip().lower()
    if key in _FLOOR_NAMES:
        return CLASS_IDS["floor"]
    if key in _WALL_NAMES:
        return CLASS_IDS["wall"]
    if key in _CEILING_NAMES:
        return CLASS_IDS["ceiling"]
    if key in _WINDOW_NAMES:
        return CLASS_IDS["window"]
    if key in _DOOR_NAMES:
        return CLASS_IDS["door"]
    if key in _FURNITURE_NAMES:
        return CLASS_IDS["furniture"]
    return CLASS_IDS["other"]


def build_class_lut(id2label: dict[int, str]) -> np.ndarray:
    """256-entry uint8 LUT: ADE label id -> our class id (unknown ids -> other)."""
    lut = np.zeros(256, dtype=np.uint8)
    for idx, name in id2label.items():
        if 0 <= int(idx) < 256:
            lut[int(idx)] = classify_ade_label(name)
    return lut


def remap_labels(ade_labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[np.asarray(ade_labels, dtype=np.uint8)]


def class_overlay(rgb_u8: np.ndarray, cls: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the class colors over the face image (both HxW[x3], uint8)."""
    colors = CLASS_COLORS[np.asarray(cls, dtype=np.uint8)]
    mixed = (1.0 - alpha) * rgb_u8.astype(np.float32) + alpha * colors.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


#: Gate-2 operator override (G6 slice 4.4 decision): a hand-corrected class
#: map dropped next to the generated ones takes precedence over BOTH the
#: SAM-refined and the raw Mask2Former maps in every consumer. Same
#: philosophy as the G3 closures.json manual walls -- operator-authored
#: artifacts instead of interactive tooling before a defect demands it.
#: Authorable in any image editor: paint 7-class ids (CLASSES indices).
CLS_OVERRIDE_TEMPLATE = "{face}_cls_override.png"


def resolve_class_map(
    seg_dir: Path, sweep_uuid: str, face: str
) -> tuple[np.ndarray | None, str | None]:
    """7-class map for one face with the operator-override precedence.

    Precedence: ``{face}_cls_override.png`` (operator-corrected, Gate-2)
    > ``{face}_cls_refined.png`` (SAM) > ``{face}_cls.png`` (Mask2Former).
    Returns (uint8 label map, source filename) or (None, None).

    ALL mask consumers (Gate-1 glass/mirror exclusion, IBL emissive regions,
    2D->3D fusion) load through this function, so a single hand-authored
    override propagates everywhere. NOTE: the IBL bake caches per sweep
    without hashing mask content -- re-bake affected sweeps with ``--force``
    after authoring an override (OPEN_PROBLEMS #22).
    """
    sweep_dir = seg_dir / sweep_uuid
    for name in (
        CLS_OVERRIDE_TEMPLATE.format(face=face),
        f"{face}_cls_refined.png",
        f"{face}_cls.png",
    ):
        path = sweep_dir / name
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img, name
    return None, None


@dataclass
class SegParams:
    model_name: str = DEFAULT_MODEL
    mask_size: int = DEFAULT_MASK_SIZE
    upright: bool = True  # upright faces before inference (identity bump: re-infers old masks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mask_size": self.mask_size,
            "upright": self.upright,
        }


class Mask2FormerRunner:
    """Lazy torch/transformers wrapper (importable without the extra installed)."""

    def __init__(self, model_name: str, device: str = "auto", fp16: bool = True) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16 and device == "cuda")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model.eval().to(device)
        if self.fp16:
            model.half()
        self.model = model
        self.id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        print(f"model {model_name} on {self.device} (fp16={self.fp16})")

    def infer_face(self, rgb_u8: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> ADE20K label map uint8 (H, W)."""
        torch = self._torch
        inputs = self.processor(images=rgb_u8, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[tuple(rgb_u8.shape[:2])]
        )[0]
        return seg.to("cpu").numpy().astype(np.uint8)


# --------------------------------------------------------------- SAM refinement (slice 2)

#: Classes whose boundaries feed downstream geometry (openings, 3D boxes).
REFINE_CLASS_IDS = (CLASS_IDS["window"], CLASS_IDS["door"], CLASS_IDS["furniture"])

_SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


@dataclass
class RefineParams:
    sam_model: str = "vit_b"  # 4 GB dev GPU; vit_h on P3 workers (OPEN_PROBLEMS #20)
    min_area: int = 300  # px at mask_size; smaller components are left as-is
    iou_min: float = 0.30  # SAM mask vs original component; below = SAM grabbed something else
    upright: bool = True  # refine in upright orientation (identity bump)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sam_model": self.sam_model,
            "min_area": self.min_area,
            "iou_min": self.iou_min,
            "upright": self.upright,
        }


def extract_instances(
    cls: np.ndarray,
    min_area: int,
    class_ids: tuple[int, ...] = REFINE_CLASS_IDS,
) -> list[dict[str, Any]]:
    """Connected components of the refine classes -> box prompts for SAM."""
    out: list[dict[str, Any]] = []
    for cid in class_ids:
        mask = (np.asarray(cls) == cid).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < min_area:
                continue
            out.append(
                {
                    "class_id": int(cid),
                    "box": (x, y, x + w, y + h),
                    "area": area,
                    "mask": labels == i,
                }
            )
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def merge_refined(
    cls: np.ndarray, refinements: list[tuple[int, np.ndarray]]
) -> np.ndarray:
    """Paint refined masks over the class map, largest first (small stay on top)."""
    refined = np.asarray(cls).copy()
    for cid, mask in sorted(refinements, key=lambda t: int(t[1].sum()), reverse=True):
        refined[mask] = cid
    return refined


class SamRefiner:
    """Lazy segment-anything wrapper; checkpoint auto-downloads if missing."""

    def __init__(self, checkpoint: Path, model_type: str = "vit_b", device: str = "auto"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            url = _SAM_CHECKPOINT_URLS.get(model_type)
            if url is None:
                raise ValueError(f"no checkpoint URL known for SAM type {model_type}")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading SAM {model_type} checkpoint -> {checkpoint}")
            torch.hub.download_url_to_file(url, str(checkpoint))
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device)
        self.predictor = SamPredictor(sam)
        print(f"SAM {model_type} on {device}")

    def refine_face(
        self, rgb_u8: np.ndarray, instances: list[dict[str, Any]]
    ) -> list[np.ndarray]:
        self.predictor.set_image(rgb_u8)
        masks: list[np.ndarray] = []
        for inst in instances:
            m, _, _ = self.predictor.predict(
                box=np.asarray(inst["box"], dtype=np.float32), multimask_output=False
            )
            masks.append(np.asarray(m[0], dtype=bool))
        return masks


def _refine_current(meta_path: Path, sweep_out: Path, params: RefineParams) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("params") != params.to_dict():
        return False
    faces = meta.get("files", {}).get("faces", {})
    if sorted(faces) != sorted(FACE_NAMES):
        return False
    return all((sweep_out / name).is_file() for name in faces.values())


def refine(
    panos_dir: Path,
    seg_dir: Path,
    params: RefineParams,
    checkpoint: Path,
    sweep_indices: set[int] | None = None,
    device: str = "auto",
    force: bool = False,
    write_review: bool = True,
) -> dict[str, list[str]]:
    """SAM boundary refinement of window/door/furniture masks; idempotent per sweep."""
    summary: dict[str, list[str]] = {"processed": [], "skipped": [], "failed": []}
    refiner: SamRefiner | None = None
    metas = []
    for meta_path in sorted(seg_dir.glob("*/seg_meta.json")):
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if sweep_indices is not None:
        metas = [m for m in metas if int(m["index"]) in sweep_indices]
    metas.sort(key=lambda m: int(m["index"]))

    for meta in metas:
        uuid = str(meta["sweep_uuid"])
        idx = int(meta["index"])
        mask_size = int(meta["params"]["mask_size"])
        sweep_out = seg_dir / uuid
        rmeta_path = sweep_out / "refine_meta.json"
        if not force and _refine_current(rmeta_path, sweep_out, params):
            print(f"[{idx:03d}] {uuid}: refinement up to date -> skipped")
            summary["skipped"].append(uuid)
            continue

        t0 = time.perf_counter()
        files: dict[str, Any] = {"faces": {}}
        rows: list[np.ndarray] = []
        n_inst = n_kept = n_lowiou = 0
        failed = False
        for face in FACE_NAMES:
            img = cv2.imread(str(panos_dir / uuid / f"{face}.jpg"), cv2.IMREAD_COLOR)
            cls = cv2.imread(str(sweep_out / f"{face}_cls.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or cls is None:
                print(f"[{idx:03d}] {uuid}: missing face/mask {face} -> FAILED")
                failed = True
                break
            rgb = upright_from_stored(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
            rgb = cv2.resize(
                rgb,
                (mask_size, mask_size),
                interpolation=cv2.INTER_AREA,
            )
            cls = upright_from_stored(cls, face)
            instances = extract_instances(cls, params.min_area)
            kept: list[tuple[int, np.ndarray]] = []
            if instances:
                if refiner is None:
                    refiner = SamRefiner(
                        checkpoint, model_type=params.sam_model, device=device
                    )
                sam_masks = refiner.refine_face(rgb, instances)
                for inst, m in zip(instances, sam_masks, strict=True):
                    if mask_iou(m, inst["mask"]) >= params.iou_min:
                        kept.append((inst["class_id"], m))
                    else:
                        n_lowiou += 1
            n_inst += len(instances)
            n_kept += len(kept)
            refined = merge_refined(cls, kept)
            cv2.imwrite(
                str(sweep_out / f"{face}_cls_refined.png"),
                stored_from_upright(refined, face),
            )
            files["faces"][face] = f"{face}_cls_refined.png"
            if write_review:
                rows.append(
                    np.hstack(
                        [
                            _panel(rgb, face),
                            _panel(class_overlay(rgb, cls), "mask2former"),
                            _panel(class_overlay(rgb, refined), "sam refined"),
                        ]
                    )
                )
        if failed:
            summary["failed"].append(uuid)
            continue

        elapsed = time.perf_counter() - t0
        if write_review and rows:
            review_dir = seg_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            body = np.vstack(rows)
            composite = np.vstack([_legend(body.shape[1]), body])
            cv2.imwrite(
                str(review_dir / f"refine_{idx:03d}_{uuid}.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )
        rmeta = {
            "schema_version": SEG_SCHEMA_VERSION,
            "sweep_uuid": uuid,
            "index": idx,
            "params": params.to_dict(),
            "files": files,
            "stats": {
                "instances": n_inst,
                "refined": n_kept,
                "low_iou_skipped": n_lowiou,
                "seconds": elapsed,
            },
        }
        rmeta_path.write_text(json.dumps(rmeta, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{idx:03d}] {uuid}: {n_inst} instance(s), {n_kept} refined, "
            f"{n_lowiou} low-IoU skipped  ({elapsed:.1f} s)"
        )
        summary["processed"].append(uuid)

    print(
        f"processed {len(summary['processed'])}, skipped {len(summary['skipped'])}, "
        f"failed {len(summary['failed'])}"
    )
    return summary


# ---------------------------------------------------------------------------- infer


def _outputs_current(meta_path: Path, sweep_out: Path, params: SegParams) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("params") != params.to_dict():
        return False
    faces = meta.get("files", {}).get("faces", {})
    if sorted(faces) != sorted(FACE_NAMES):
        return False
    return all(
        (sweep_out / entry[kind]).is_file()
        for entry in faces.values()
        for kind in ("ade", "cls")
    )


def _panel(img_u8: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    interp = cv2.INTER_NEAREST if img_u8.shape[0] < size else cv2.INTER_AREA
    img = cv2.resize(np.ascontiguousarray(img_u8), (size, size), interpolation=interp)
    cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def _legend(width: int, height: int = 26) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    seg_w = width // len(CLASSES)
    for i, name in enumerate(CLASSES):
        bar[:, i * seg_w : (i + 1) * seg_w] = CLASS_COLORS[i]
        cv2.putText(
            bar, name, (i * seg_w + 4, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return bar


def infer(
    panos_dir: Path,
    out_dir: Path,
    params: SegParams,
    sweep_indices: set[int] | None = None,
    device: str = "auto",
    fp16: bool = True,
    force: bool = False,
    write_review: bool = True,
) -> dict[str, list[str]]:
    """Run per-face segmentation for the selected sweeps; idempotent per sweep."""
    manifest = json.loads((panos_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = list(manifest.get("sweeps", []))
    if sweep_indices is not None:
        entries = [e for e in entries if int(e["index"]) in sweep_indices]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[str]] = {"processed": [], "skipped": [], "failed": []}
    runner: Mask2FormerRunner | None = None
    lut: np.ndarray | None = None

    for entry in entries:
        uuid = str(entry["sweep_uuid"])
        idx = int(entry["index"])
        faces_rel = entry.get("faces", {})
        if sorted(faces_rel) != sorted(FACE_NAMES):
            print(f"[{idx:03d}] {uuid}: incomplete faces -> FAILED")
            summary["failed"].append(uuid)
            continue
        sweep_out = out_dir / uuid
        meta_path = sweep_out / "seg_meta.json"
        if not force and _outputs_current(meta_path, sweep_out, params):
            print(f"[{idx:03d}] {uuid}: up to date -> skipped")
            summary["skipped"].append(uuid)
            continue

        if runner is None:
            runner = Mask2FormerRunner(params.model_name, device=device, fp16=fp16)
            lut = build_class_lut(runner.id2label)
        assert lut is not None

        t0 = time.perf_counter()
        sweep_out.mkdir(parents=True, exist_ok=True)
        files: dict[str, Any] = {"faces": {}}
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        rows: list[np.ndarray] = []
        failed = False
        for face in FACE_NAMES:
            img = cv2.imread(str(panos_dir / faces_rel[face]), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[{idx:03d}] {uuid}: unreadable face {face} -> FAILED")
                failed = True
                break
            rgb = upright_from_stored(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
            rgb = cv2.resize(
                rgb, (params.mask_size, params.mask_size), interpolation=cv2.INTER_AREA
            )
            ade = runner.infer_face(rgb)
            cls = remap_labels(ade, lut)
            cv2.imwrite(str(sweep_out / f"{face}_ade.png"), stored_from_upright(ade, face))
            cv2.imwrite(str(sweep_out / f"{face}_cls.png"), stored_from_upright(cls, face))
            files["faces"][face] = {"ade": f"{face}_ade.png", "cls": f"{face}_cls.png"}
            counts += np.bincount(cls.reshape(-1), minlength=len(CLASSES))
            if write_review:
                rows.append(
                    np.hstack(
                        [
                            _panel(rgb, face),
                            _panel(class_overlay(rgb, cls), "classes"),
                        ]
                    )
                )
        if failed:
            summary["failed"].append(uuid)
            continue

        fractions = {
            name: float(counts[i]) / float(max(counts.sum(), 1))
            for i, name in enumerate(CLASSES)
        }
        elapsed = time.perf_counter() - t0

        if write_review and rows:
            review_dir = out_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            body = np.vstack(rows)
            composite = np.vstack([_legend(body.shape[1]), body])
            cv2.imwrite(
                str(review_dir / f"seg_{idx:03d}_{uuid}.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )

        meta = {
            "schema_version": SEG_SCHEMA_VERSION,
            "sweep_uuid": uuid,
            "index": idx,
            "params": params.to_dict(),
            "runtime": {"device": runner.device, "fp16": runner.fp16},
            "classes": list(CLASSES),
            "mask_orientation": "stored canonical (pixel-aligned with extracted faces); "
            "inference ran upright",
            "files": files,
            "stats": {"class_fractions": fractions, "seconds": elapsed},
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        frac_str = " ".join(f"{n[:4]} {fractions[n]:5.1%}" for n in CLASSES)
        print(f"[{idx:03d}] {uuid}: {frac_str}  ({elapsed:.1f} s)")
        summary["processed"].append(uuid)

    index_path = assemble_index(out_dir)
    print(
        f"processed {len(summary['processed'])}, skipped {len(summary['skipped'])}, "
        f"failed {len(summary['failed'])} -> {index_path}"
    )
    return summary


def assemble_index(out_dir: Path) -> Path:
    """Rebuild segmentation_index.json from every per-sweep meta on disk."""
    sweeps = []
    for meta_path in sorted(out_dir.glob("*/seg_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sweeps.append(
            {
                "sweep_uuid": meta["sweep_uuid"],
                "index": meta["index"],
                "params": meta["params"],
                "class_fractions": meta["stats"]["class_fractions"],
            }
        )
    sweeps.sort(key=lambda s: int(s["index"]))
    payload = {
        "schema_version": SEG_SCHEMA_VERSION,
        "classes": list(CLASSES),
        "sweep_count": len(sweeps),
        "sweeps": sweeps,
    }
    path = out_dir / "segmentation_index.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4 (G6) slice 1: per-face semantic segmentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_inf = sub.add_parser("infer", help="extracted cubemaps -> ADE + 7-class masks")
    p_inf.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p_inf.add_argument("--output", "-o", type=Path, default=Path("seg_out"))
    p_inf.add_argument(
        "--sweeps", type=str, default=None, help="index selection, e.g. '1,50,60,175'"
    )
    p_inf.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model id (Swin-T default for the 4 GB dev GPU; Swin-L on workers)",
    )
    p_inf.add_argument("--mask-size", type=int, default=DEFAULT_MASK_SIZE)
    p_inf.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p_inf.add_argument("--no-fp16", action="store_true")
    p_inf.add_argument("--force", action="store_true")
    p_inf.add_argument("--no-review", action="store_true")

    p_ref = sub.add_parser("refine", help="SAM boundary refinement of window/door/furniture")
    p_ref.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p_ref.add_argument("--seg-dir", type=Path, default=Path("seg_out"))
    p_ref.add_argument("--sweeps", type=str, default=None)
    p_ref.add_argument(
        "--sam-model", type=str, default="vit_b", help="vit_b (4 GB dev GPU) | vit_l | vit_h"
    )
    p_ref.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "sam_vit_b_01ec64.pth",
        help="SAM checkpoint path (auto-downloads if missing)",
    )
    p_ref.add_argument("--min-area", type=int, default=300)
    p_ref.add_argument("--iou-min", type=float, default=0.30)
    p_ref.add_argument("--device", type=str, default="auto")
    p_ref.add_argument("--force", action="store_true")
    p_ref.add_argument("--no-review", action="store_true")
    args = parser.parse_args(argv)

    from vs_pipeline.stages.ibl import parse_sweep_selection

    selection = parse_sweep_selection(args.sweeps) if args.sweeps else None

    if args.command == "refine":
        if not args.seg_dir.is_dir():
            parser.error(f"seg dir not found: {args.seg_dir}")
        rparams = RefineParams(
            sam_model=args.sam_model, min_area=args.min_area, iou_min=args.iou_min
        )
        refine(
            args.panos,
            args.seg_dir,
            rparams,
            checkpoint=args.checkpoint,
            sweep_indices=selection,
            device=args.device,
            force=args.force,
            write_review=not args.no_review,
        )
        return 0

    if not (args.panos / "manifest.json").is_file():
        parser.error(f"manifest.json not found in panos dir: {args.panos}")
    params = SegParams(model_name=args.model, mask_size=args.mask_size)
    infer(
        args.panos,
        args.output,
        params,
        sweep_indices=selection,
        device=args.device,
        fp16=not args.no_fp16,
        force=args.force,
        write_review=not args.no_review,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
