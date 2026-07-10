"""Stage 2 (part 1) — E57 pose extraction.

Reads per-sweep rigid-body poses from an ``.e57`` file, applies the one-time
Matterport Z-up -> runtime Y-up conversion (Coordinate Lock Rule), and emits
``poses.json`` expressed in the runtime frame.

Idempotent: identical input file -> identical output (stable sweep UUIDs,
source checksum recorded).

CLI:
    python -m vs_pipeline.stages.pose input.e57 --output poses.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vs_pipeline.coordinates import Matrix4, convert_pose_zup_to_yup

POSES_SCHEMA_VERSION = 1

#: Namespace for deterministic fallback sweep UUIDs (uuid5).
_SWEEP_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


@dataclass(frozen=True)
class SweepPose:
    """One sweep's pose, already expressed in the runtime Y-up frame."""

    sweep_uuid: str
    index: int
    pose_rt: Matrix4  # 4x4, row-major, runtime Y-up


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file (works for multi-GB scans)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scan_guid(header: Any) -> str | None:
    """Best-effort read of the E57 per-scan GUID across pye57 versions."""
    guid = getattr(header, "guid", None)
    if guid is None:
        node = getattr(header, "node", None)
        if node is not None:
            try:
                guid = node["guid"].value()
            except Exception:
                guid = None
    if guid is None:
        return None
    text = str(guid).strip().strip("{}").lower()
    return text or None


def fallback_sweep_uuid(source_name: str, index: int) -> str:
    """Deterministic UUID when the E57 scan carries no GUID (idempotency)."""
    return str(uuid.uuid5(_SWEEP_UUID_NAMESPACE, f"{source_name}:scan:{index}"))


def pose_matrix_from_header(header: Any) -> Matrix4:
    """Build the 4x4 sweep->world pose (Matterport Z-up frame) from an E57 header."""
    matrix_mp = np.eye(4, dtype=np.float64)
    matrix_mp[:3, :3] = np.asarray(header.rotation_matrix, dtype=np.float64)
    matrix_mp[:3, 3] = np.asarray(header.translation, dtype=np.float64).reshape(3)
    return matrix_mp


def extract_poses(e57_path: Path) -> list[SweepPose]:
    """Extract all sweep poses from an .e57 file, converted to the runtime frame.

    The Z-up -> Y-up conversion is applied HERE and only here (Coordinate Lock
    Rule): every consumer of the returned poses works in runtime Y-up.
    """
    import pye57  # local import: keeps module importable without libE57 present

    e57 = pye57.E57(str(e57_path))
    sweeps: list[SweepPose] = []
    for index in range(e57.scan_count):
        header = e57.get_header(index)
        pose_mp = pose_matrix_from_header(header)
        pose_rt = convert_pose_zup_to_yup(pose_mp)
        sweep_uuid = _read_scan_guid(header) or fallback_sweep_uuid(e57_path.name, index)
        sweeps.append(SweepPose(sweep_uuid=sweep_uuid, index=index, pose_rt=pose_rt))
    return sweeps


def write_poses_json(
    sweeps: list[SweepPose],
    output_path: Path,
    source_file: str,
    source_sha256: str,
) -> None:
    """Serialize poses to poses.json (schema v1, runtime Y-up frame)."""
    payload = {
        "schema_version": POSES_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "conversion": "matterport_zup_to_runtime_yup",
        "source_file": source_file,
        "source_sha256": source_sha256,
        "sweep_count": len(sweeps),
        "sweeps": [
            {
                "sweep_uuid": s.sweep_uuid,
                "index": s.index,
                "pose_rt": s.pose_rt.tolist(),
            }
            for s in sweeps
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_poses_json(path: Path) -> list[SweepPose]:
    """Load poses.json back into typed records (validates schema version)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != POSES_SCHEMA_VERSION:
        raise ValueError(f"unsupported poses.json schema: {payload.get('schema_version')}")
    if payload.get("coordinate_frame") != "runtime_yup":
        raise ValueError("poses.json is not in the runtime Y-up frame")
    return [
        SweepPose(
            sweep_uuid=item["sweep_uuid"],
            index=item["index"],
            pose_rt=np.asarray(item["pose_rt"], dtype=np.float64),
        )
        for item in payload["sweeps"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-sweep poses from an .e57 file into poses.json "
        "(runtime Y-up frame)."
    )
    parser.add_argument("input", type=Path, help="path to the .e57 file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("poses.json"),
        help="output path (default: ./poses.json)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    checksum = sha256_file(args.input)
    sweeps = extract_poses(args.input)
    write_poses_json(sweeps, args.output, source_file=args.input.name, source_sha256=checksum)

    print(f"{len(sweeps)} sweep pose(s) -> {args.output} (frame: runtime Y-up)")
    for sweep in sweeps:
        t = sweep.pose_rt[:3, 3]
        print(f"  [{sweep.index:03d}] {sweep.sweep_uuid}  t = ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
