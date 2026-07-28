"""Stage 2 (part 2) — embedded cubemap extraction from E57 ``images2D``.

Matterport E57 exports embed six cube-face images per sweep. This stage:

1. reads each image2D entry via pye57's low-level libe57 API,
2. classifies which cube face it is (dominant scanner-local view axis, derived
   from the image pose quaternion and the inverse sweep rotation),
3. applies the canonical Set A transforms (see vs_pipeline/cubemap_transforms.py:
   rotation -> flip -> +Z/-Z swap; proven in the previous production pipeline),
4. associates faces with the sweep GUID (same normalization as the pose stage,
   so keys match ``poses.json`` exactly),
5. writes faces to ``{out}/{sweep_uuid}/{face}.jpg`` + ``manifest.json``.

No viewer-specific pose corrections are baked into the output — poses.json
remains the single source of pose truth. Any residual global orientation
adjustment for Babylon.js is resolved at runtime during F4.4 calibration.

CLI:
    python -m vs_pipeline.stages.panorama input.e57 -o extracted_cubemaps/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vs_pipeline.cubemap_transforms import (
    FACE_FLIPS,
    FACE_OUTPUT_AXIS,
    FACE_ROTATIONS_CW,
    transform_face,
)
from vs_pipeline.stages.pose import _read_scan_guid, fallback_sweep_uuid, sha256_file

PANORAMA_SCHEMA_VERSION = 1

FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")

_AXIS_TO_FACE = {"+X": "px", "-X": "nx", "+Y": "py", "-Y": "ny", "+Z": "pz", "-Z": "nz"}

_REPRESENTATIONS = (
    "sphericalRepresentation",
    "pinholeRepresentation",
    "cylindricalRepresentation",
    "visualReferenceRepresentation",
)

_JPEG_QUALITY = 97


@dataclass
class SweepFaces:
    """Extracted cube faces for one sweep (paths relative to the output dir)."""

    sweep_uuid: str
    index: int
    faces: dict[str, str] = field(default_factory=dict)


def quat_rotate(q_xyzw: Any, v: Any) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``(x, y, z, w)``."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q)
    xyz, w = q[:3], q[3]
    vec = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(xyz, vec)
    result: np.ndarray = vec + w * t + np.cross(xyz, t)
    return result


def quat_conjugate(q_xyzw: Any) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def dominant_axis(vec: Any) -> str:
    """'+X' / '-Y' / ... for the axis with the largest absolute component."""
    v = np.asarray(vec, dtype=np.float64)
    idx = int(np.argmax(np.abs(v)))
    sign = "+" if v[idx] > 0 else "-"
    return f"{sign}{'XYZ'[idx]}"


def _read_image_quaternion(img_node: Any) -> np.ndarray | None:
    """Image pose rotation as (x, y, z, w), or None if absent."""
    try:
        rot = img_node["pose"]["rotation"]
        return np.array(
            [rot["x"].value(), rot["y"].value(), rot["z"].value(), rot["w"].value()],
            dtype=np.float64,
        )
    except Exception:
        return None


def _read_blob(img_node: Any) -> tuple[bytes, str] | None:
    """Return (raw bytes, extension) of the first image blob found."""
    for rep_name in _REPRESENTATIONS:
        try:
            rep = img_node[rep_name]
        except Exception:
            continue
        for blob_name, ext in (("jpegImage", ".jpg"), ("pngImage", ".png")):
            try:
                blob = rep[blob_name]
            except Exception:
                continue
            count = blob.byteCount()
            buffer = np.zeros(count, dtype=np.uint8)
            blob.read(buffer, 0, count)
            return buffer.tobytes(), ext
    return None


def _read_assoc_guid(img_node: Any) -> str | None:
    """E57 associatedData3DGuid, normalized identically to the pose stage."""
    try:
        raw = img_node["associatedData3DGuid"].value()
    except Exception:
        return None
    text = str(raw).strip().strip("{}").lower()
    return text or None


def extract_cubemaps(e57_path: Path, output_dir: Path) -> list[SweepFaces]:
    """Extract all embedded cube faces, canonically transformed, per sweep UUID."""
    import pye57  # local import: keeps module importable without libE57 present

    e57 = pye57.E57(str(e57_path))
    try:
        root = e57.image_file.root()
        try:
            images2d = root["images2D"]
        except Exception as exc:
            raise ValueError(
                "no images2D section: this E57 has no embedded panoramas"
            ) from exc

        # Sweep GUID lookup + inverse sweep rotations (world -> scanner-local).
        import os
        sweep_range = range(e57.scan_count)
        if os.environ.get("VS_TEST_SWEEP_RANGE"):
            parts = os.environ["VS_TEST_SWEEP_RANGE"].split("-")
            sweep_range = range(int(parts[0]), int(parts[1]) + 1)
            
        scan_index_by_guid: dict[str, int] = {}
        sweeps: list[SweepFaces] = []
        inv_rotations: list[np.ndarray] = []
        # We must index sweeps based on the actual scan indices, but inv_rotations and sweeps list
        # are mapped by index. We will pad None for skipped indices to maintain indices for later lookup.
        # Actually, `extract_cubemaps` uses `scan_idx >= len(sweeps)` so it expects `sweeps` to match `e57.scan_count`
        # To make it safer, we just append None to `sweeps` for skipped indices.
        for scan_idx in range(e57.scan_count):
            if scan_idx not in sweep_range:
                sweeps.append(None)
                inv_rotations.append(None)
                continue
            header = e57.get_header(scan_idx)
            guid = _read_scan_guid(header) or fallback_sweep_uuid(e57_path.name, scan_idx)
            scan_index_by_guid[guid] = scan_idx
            sweeps.append(SweepFaces(sweep_uuid=guid, index=scan_idx))
            w, x, y, z = (float(v) for v in header.rotation)  # pye57 order: [w, x, y, z]
            inv_rotations.append(quat_conjugate([x, y, z, w]))

        image_count = images2d.childCount()
        for img_idx in range(image_count):
            img_node = images2d[img_idx]

            assoc = _read_assoc_guid(img_node)
            scan_idx = scan_index_by_guid.get(assoc) if assoc is not None else None
            if scan_idx is None:
                scan_idx = img_idx // 6  # Matterport fallback: 6 faces per sweep, in order
            if scan_idx >= len(sweeps) or sweeps[scan_idx] is None:
                continue

            quat = _read_image_quaternion(img_node)
            if quat is None:
                continue
            # Global view direction -> scanner-local -> dominant axis = source face.
            local_fwd = quat_rotate(inv_rotations[scan_idx], quat_rotate(quat, [0.0, 0.0, 1.0]))
            source_axis = dominant_axis(local_fwd)

            blob = _read_blob(img_node)
            if blob is None:
                continue
            raw_bytes, _ext = blob

            decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(
                    f"failed to decode face {source_axis} of sweep {sweeps[scan_idx].sweep_uuid}"
                )

            # Canonical Set A transform (rotation -> flip -> Z swap).
            transformed, output_axis = transform_face(decoded, source_axis)
            face = _AXIS_TO_FACE[output_axis]

            sweep = sweeps[scan_idx]
            face_dir = output_dir / sweep.sweep_uuid
            face_dir.mkdir(parents=True, exist_ok=True)
            out_path = face_dir / f"{face}.jpg"

            ok, encoded = cv2.imencode(
                ".jpg",
                np.ascontiguousarray(transformed),  # rot90/flip return strided views
                [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
            )
            if not ok:
                raise RuntimeError(
                    f"failed to encode face {face} of sweep {sweep.sweep_uuid}"
                )
            out_path.write_bytes(encoded.tobytes())
            sweep.faces[face] = f"{sweep.sweep_uuid}/{face}.jpg"
        return [s for s in sweeps if s is not None]
    finally:
        e57.close()


def write_manifest(
    sweeps: list[SweepFaces],
    output_dir: Path,
    source_file: str,
    source_sha256: str,
) -> Path:
    """Write manifest.json describing all extracted faces (schema v1)."""
    incomplete = [s.sweep_uuid for s in sweeps if len(s.faces) != 6]
    payload = {
        "schema_version": PANORAMA_SCHEMA_VERSION,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "canonical_transforms": {
            "order": ["set_a_rotation_cw", "z_inversion_flips_and_swap"],
            "rotations_cw": dict(FACE_ROTATIONS_CW),
            "flips": dict(FACE_FLIPS),
            "output_axis_remap": dict(FACE_OUTPUT_AXIS),
        },
        "sweep_count": len(sweeps),
        "incomplete_sweeps": incomplete,
        "sweeps": [
            {"sweep_uuid": s.sweep_uuid, "index": s.index, "faces": s.faces} for s in sweeps
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract embedded per-sweep cubemap faces from an .e57 file."
    )
    parser.add_argument("input", type=Path, help="path to the .e57 file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("extracted_cubemaps"),
        help="output directory (default: ./extracted_cubemaps)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    checksum = sha256_file(args.input)
    sweeps = extract_cubemaps(args.input, args.output)
    manifest_path = write_manifest(
        sweeps, args.output, source_file=args.input.name, source_sha256=checksum
    )

    complete = sum(1 for s in sweeps if len(s.faces) == 6)
    print(f"{len(sweeps)} sweep(s), {complete} with all 6 faces -> {args.output}")
    incomplete = [s for s in sweeps if len(s.faces) != 6]
    for sweep in incomplete:
        print(f"  WARNING [{sweep.index:03d}] {sweep.sweep_uuid}: only {sorted(sweep.faces)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
