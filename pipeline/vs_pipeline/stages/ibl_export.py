"""G7.2 -- Babylon.js `.env` exporter: per-sweep G5 IBL products -> prefiltered environment files.

Consumes the G5 bake (linear-HDR ``.npy`` cubemaps: the GGX specular chain) and
writes one Babylon-native ``.env`` per sweep for the viewer's
``scene.environmentTexture`` swap (G7.2). Interim format until P4 (KTX2).

FORMAT FACTS -- verified against the SHIPPED Babylon source in
``viewer/node_modules/@babylonjs/core`` (environmentTextureTools,
cubemapToSphericalPolynomial, sphericalPolynomial, helperFunctions
fromRGBD/toRGBD), NOT web search (LESSONS L17):

- Container: 8 binary magic bytes ``0x86 16 87 96 F6 D6 96 36``, then a
  NULL-TERMINATED ascii JSON manifest, then the raw payload. Manifest v2::

      {"version": 2, "width": W, "imageType": "image/png",
       "irradiance": {"x","y","z","xx","yy","zz","yz","zx","xy": [r,g,b]},
       "specular": {"mipmaps": [{"length","position"}...],
                     "lodGenerationScale": 0.8}}

  mipmaps are mip-major, face-minor (+X,-X,+Y,-Y,+Z,-Z = our px,nx,py,ny,pz,nz);
  positions are relative to the payload start; the chain runs ALL the way to
  1x1 (``ILog2(width)+1`` levels).
- RGBD (toRGBD shader): ``D = saturate(floor(max(255/maxRGB, 1))/255)`` stored
  in alpha; rgb premultiplied by D then GAMMA-encoded (``pow(x, 1/2.2)`` --
  Babylon's shader approximation, NOT piecewise sRGB); decode is
  ``pow(rgb, 2.2)/D``. HDR range up to 255.
- Irradiance polynomial: project RADIANCE SH9 (exact per-texel solid angles via
  the AreaElement identity, 4pi total correction), then
  convertIncidentRadianceToIrradiance (cosine kernel), then
  convertIrradianceToLambertianRadiance (1/pi), then the polynomial mapping
  (updateFromHarmonics constants) with a final 1/pi. Babylon's
  face->direction tables are IDENTICAL to :func:`ibl.face_directions`
  (checked axis by axis), so the stored canonical faces need no relabeling.

DOUBLE-CONVOLUTION TRAP (banked before implementation): ``FromHarmonics``
expects RADIANCE harmonics and applies the cosine kernel itself. The G5
irradiance map already has the cosine baked in -- projecting SH from it would
convolve twice (uniformly dim diffuse). The polynomial here is therefore
projected from the MIP-0 RADIANCE environment, exactly like Babylon's own
CreateEnvTextureAsync; the G5 irradiance product remains the bake validator,
not the runtime diffuse source.

KNOWN APPROXIMATIONS (OPEN_PROBLEMS #25, judged at the G8 chrome-sphere gate):
- Tail mips below the G5 chain's 16x16 (8,4,2,1) are area-downsampled from the
  roughness-1.0 mip, not GGX-prefiltered (near-uniform by then).
- The G5 roughness schedule (linear m/4 over 5 mips) is re-indexed by Babylon's
  ``lodGenerationScale = 0.8`` shader mapping; mid-roughness blur may sit
  slightly off until the schedules are aligned.

CLI::

    python -m vs_pipeline.stages.ibl_export export --ibl-dir ibl_out -o env_out \
        [--sweeps "1,50,60,152,157,175"] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vs_pipeline.stages.ibl import FACE_NAMES, _resized, face_directions, parse_sweep_selection

ENV_SCHEMA_VERSION = 1

#: environmentTextureTools.js MagicBytes -- verbatim.
MAGIC_BYTES = bytes([0x86, 0x16, 0x87, 0x96, 0xF6, 0xD6, 0x96, 0x36])

#: Babylon shader gamma approximation (toGammaSpace/toLinearSpace), NOT sRGB.
GAMMA = 2.2
#: rgbdMaxRange in the toRGBD shader.
RGBD_MAX_RANGE = 255.0
#: CubeMapToSphericalPolynomial.MAX_HDRI_VALUE (PRESERVE_CLAMPED_COLORS=false).
MAX_HDRI_VALUE = 4096.0
#: Babylon default / GetEnvInfo fallback.
LOD_GENERATION_SCALE = 0.8

#: SH3ylmBasisConstants -- verbatim (order: l00, l1_1, l10, l11, l2_2, l2_1, l20, l21, l22).
_SH_CONSTS = np.array(
    [
        math.sqrt(1.0 / (4.0 * math.pi)),
        -math.sqrt(3.0 / (4.0 * math.pi)),
        math.sqrt(3.0 / (4.0 * math.pi)),
        -math.sqrt(3.0 / (4.0 * math.pi)),
        math.sqrt(15.0 / (4.0 * math.pi)),
        -math.sqrt(15.0 / (4.0 * math.pi)),
        math.sqrt(5.0 / (16.0 * math.pi)),
        -math.sqrt(15.0 / (4.0 * math.pi)),
        math.sqrt(15.0 / (16.0 * math.pi)),
    ],
    dtype=np.float64,
)

#: SHCosKernelConvolution -- verbatim.
_COS_KERNEL = np.array(
    [math.pi] + [2.0 * math.pi / 3.0] * 3 + [math.pi / 4.0] * 5, dtype=np.float64
)

EXPORT_PARAMS: dict[str, Any] = {
    "schema": ENV_SCHEMA_VERSION,
    "gamma": GAMMA,
    "lod_generation_scale": LOD_GENERATION_SCALE,
    "image_type": "image/png",
}


# ---------------------------------------------------------------------------- RGBD


def encode_rgbd(lin_rgb: np.ndarray) -> np.ndarray:
    """Linear HDR (h, w, 3) float -> Babylon RGBD (h, w, 4) uint8.

    Port of the toRGBD shader: shared divisor D in alpha, rgb premultiplied by
    D and stored GAMMA-encoded inside the PNG (helps 8-bit quantization; the
    fromRGBD decode applies toLinearSpace BEFORE dividing by alpha).
    """
    lin = np.clip(np.asarray(lin_rgb, dtype=np.float32), 0.0, None)
    max_rgb = np.maximum(lin.max(axis=-1), 1e-7)  # maxEps
    d = np.maximum(RGBD_MAX_RANGE / max_rgb, 1.0)
    d = np.clip(np.floor(d) / 255.0, 0.0, 1.0)
    rgb = np.clip(lin * d[..., None], 0.0, 1.0) ** (1.0 / GAMMA)
    out = np.empty(lin.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = np.round(d * 255.0).astype(np.uint8)
    return out


def decode_rgbd(rgbd: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_rgbd` (fromRGBD): pow(rgb, 2.2) / alpha."""
    rgbd = np.asarray(rgbd, dtype=np.float32) / 255.0
    rgb = rgbd[..., :3] ** GAMMA
    return (rgb / np.maximum(rgbd[..., 3:4], 1e-7)).astype(np.float32)


def _png_bytes(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]))
    if not ok:
        raise RuntimeError("cv2 PNG encode failed")
    return buf.tobytes()


# ------------------------------------------------------------------ SH / polynomial


def _area_element(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Babylon _AreaElement: atan2(x*y, sqrt(x^2 + y^2 + 1))."""
    return np.arctan2(x * y, np.sqrt(x * x + y * y + 1.0))


def exact_solid_angles(size: int) -> np.ndarray:
    """Exact per-texel solid angle (identical for all faces); 6 faces sum to 4pi.

    Uses the AreaElement identity exactly as CubeMapToSphericalPolynomial does
    (NOT the differential approximation in :func:`ibl.face_solid_angles`).
    """
    a = ((np.arange(size, dtype=np.float64) + 0.5) / size) * 2.0 - 1.0
    u, v = np.meshgrid(a, a)
    h = 1.0 / size  # half texel
    return (
        _area_element(u - h, v - h)
        - _area_element(u - h, v + h)
        - _area_element(u + h, v - h)
        + _area_element(u + h, v + h)
    )


def project_radiance_sh(env_faces: np.ndarray, sample_size: int = 64) -> np.ndarray:
    """Radiance cubemap (6, s, s, 3) linear -> SH9 (9, 3), solid-angle corrected.

    Faces are downsampled to ``sample_size`` first (SH order 2 is low-frequency;
    Babylon added the same targetSize downsample). NaN guard + MAX_HDRI clamp
    per the engine source.
    """
    size = min(sample_size, int(env_faces.shape[1]))
    sa = exact_solid_angles(size).reshape(-1)
    sh = np.zeros((9, 3), dtype=np.float64)
    for i, face in enumerate(FACE_NAMES):
        dirs = face_directions(face, size).reshape(-1, 3).astype(np.float64)
        col = np.nan_to_num(
            _resized(env_faces[i], size).reshape(-1, 3).astype(np.float64),
            nan=0.0,
        )
        col = np.clip(col, 0.0, MAX_HDRI_VALUE)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        terms = np.stack(
            [
                np.ones_like(x),
                y,
                z,
                x,
                x * y,
                y * z,
                3.0 * z * z - 1.0,
                x * z,
                x * x - y * y,
            ]
        )  # (9, N)
        sh += _SH_CONSTS[:, None] * ((terms * sa) @ col)
    total_sa = 6.0 * float(sa.sum())  # identical per face
    sh *= (4.0 * math.pi) / total_sa  # correctionFactor
    return sh


def sh_to_polynomial(sh: np.ndarray) -> dict[str, list[float]]:
    """Full Babylon chain: cosine kernel -> lambertian 1/pi -> polynomial (+1/pi)."""
    sh = np.asarray(sh, dtype=np.float64) * _COS_KERNEL[:, None]  # radiance -> irradiance
    sh = sh / math.pi  # convertIrradianceToLambertianRadiance
    l00, l1_1, l10, l11, l2_2, l2_1, l20, l21, l22 = sh
    poly = {
        "x": -1.02333 * l11,
        "y": -1.02333 * l1_1,
        "z": 1.02333 * l10,
        "xx": 0.886277 * l00 - 0.247708 * l20 + 0.429043 * l22,
        "yy": 0.886277 * l00 - 0.247708 * l20 - 0.429043 * l22,
        "zz": 0.886277 * l00 + 0.495417 * l20,
        "yz": -0.858086 * l2_1,
        "zx": -0.858086 * l21,
        "xy": 0.858086 * l2_2,
    }
    return {k: [float(c) for c in (v / math.pi)] for k, v in poly.items()}


def polynomial_from_env(env_faces: np.ndarray, sample_size: int = 64) -> dict[str, list[float]]:
    return sh_to_polynomial(project_radiance_sh(env_faces, sample_size))


def evaluate_polynomial(poly: dict[str, list[float]], n: tuple[float, float, float]) -> np.ndarray:
    """Shader-side irradiance evaluation for a unit normal (diagnostics/tests)."""
    x, y, z = n
    out = np.zeros(3, dtype=np.float64)
    for key, w in (("x", x), ("y", y), ("z", z), ("xx", x * x), ("yy", y * y),
                   ("zz", z * z), ("yz", y * z), ("zx", z * x), ("xy", x * y)):
        out += np.asarray(poly[key], dtype=np.float64) * w
    return out


# ------------------------------------------------------------------------ container


def extend_mip_chain(mips: list[np.ndarray]) -> list[np.ndarray]:
    """Extend a (6, s, s, 3) chain down to 1x1 by 2x area downsampling (#25)."""
    chain = [np.asarray(m, dtype=np.float32) for m in mips]
    while chain[-1].shape[1] > 1:
        last = chain[-1]
        ns = max(1, last.shape[1] // 2)
        chain.append(
            np.stack(
                [
                    cv2.resize(last[f], (ns, ns), interpolation=cv2.INTER_AREA).reshape(ns, ns, 3)
                    for f in range(6)
                ]
            )
        )
    return chain


def build_env_bytes(mips: list[np.ndarray], polynomial: dict[str, list[float]]) -> bytes:
    """Assemble a Babylon `.env` v2 file from a full 1x1-terminated mip chain."""
    width = int(mips[0].shape[1])
    expected = int(math.log2(width)) + 1
    if len(mips) != expected:
        raise ValueError(
            f"chain must reach 1x1: got {len(mips)} mips for width {width} (need {expected})"
        )
    slices: list[bytes] = []
    for mip in mips:  # mip-major, face-minor (+X,-X,+Y,-Y,+Z,-Z)
        for f in range(6):
            slices.append(_png_bytes(encode_rgbd(mip[f])))
    mipmaps = []
    pos = 0
    for s in slices:
        mipmaps.append({"length": len(s), "position": pos})
        pos += len(s)
    manifest = {
        "version": 2,
        "width": width,
        "imageType": "image/png",
        "irradiance": polynomial,
        "specular": {"mipmaps": mipmaps, "lodGenerationScale": LOD_GENERATION_SCALE},
    }
    js = json.dumps(manifest, separators=(",", ":")).encode("ascii") + b"\x00"
    return MAGIC_BYTES + js + b"".join(slices)


def read_env(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Parse magic + manifest; returns (manifest, payload). Round-trip helper."""
    if data[: len(MAGIC_BYTES)] != MAGIC_BYTES:
        raise ValueError("not a Babylon environment map (magic mismatch)")
    end = data.index(0, len(MAGIC_BYTES))
    manifest = json.loads(data[len(MAGIC_BYTES) : end].decode("ascii"))
    return manifest, data[end + 1 :]


# --------------------------------------------------------------------------- export


def export_sweep(sweep_dir: Path, out_path: Path, sh_sample_size: int = 64) -> dict[str, Any]:
    meta = json.loads((sweep_dir / "ibl_meta.json").read_text(encoding="utf-8"))
    mips = [
        np.load(sweep_dir / name).astype(np.float32)
        for name in meta["files"]["specular_mips"]
    ]
    chain = extend_mip_chain(mips)
    poly = polynomial_from_env(chain[0], sh_sample_size)
    env_bytes = build_env_bytes(chain, poly)
    out_path.write_bytes(env_bytes)
    return {
        "mips": len(chain),
        "width": int(chain[0].shape[1]),
        "bytes": len(env_bytes),
        "poly_xx_luma": float(np.mean(poly["xx"])),
    }


def _identity(sweep_dir: Path, meta: dict[str, Any], sh_sample_size: int) -> dict[str, Any]:
    return {
        "params": {**EXPORT_PARAMS, "sh_sample_size": sh_sample_size},
        "sources": {
            name: (sweep_dir / name).stat().st_mtime
            for name in meta["files"]["specular_mips"]
        },
    }


def export(
    ibl_dir: Path,
    out_dir: Path,
    sweep_indices: set[int] | None = None,
    force: bool = False,
    sh_sample_size: int = 64,
) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_paths = sorted(ibl_dir.glob("*/ibl_meta.json"))
    summary: dict[str, list[str]] = {"exported": [], "skipped": [], "failed": []}
    index: list[dict[str, Any]] = []
    for meta_path in meta_paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        idx = int(meta["index"])
        uuid = str(meta["sweep_uuid"])
        if sweep_indices is not None and idx not in sweep_indices:
            continue
        sweep_dir = meta_path.parent
        out_path = out_dir / f"{uuid}.env"
        emeta_path = out_dir / f"{uuid}.env.meta.json"
        identity = _identity(sweep_dir, meta, sh_sample_size)
        if not force and out_path.is_file() and emeta_path.is_file():
            try:
                if json.loads(emeta_path.read_text(encoding="utf-8")) == identity:
                    print(f"[{idx:03d}] {uuid}: up to date -> skipped")
                    summary["skipped"].append(uuid)
                    index.append({"sweep_uuid": uuid, "index": idx, "env": out_path.name})
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        try:
            stats = export_sweep(sweep_dir, out_path, sh_sample_size)
        except Exception as exc:  # noqa: BLE001 -- per-sweep isolation: report + continue
            print(f"[{idx:03d}] {uuid}: FAILED ({exc})")
            summary["failed"].append(uuid)
            continue
        emeta_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        index.append({"sweep_uuid": uuid, "index": idx, "env": out_path.name})
        print(
            f"[{idx:03d}] {uuid}: {stats['mips']} mips @ {stats['width']}px, "
            f"{stats['bytes'] / 1024:.0f} KB, poly xx luma {stats['poly_xx_luma']:.4f}"
        )
        summary["exported"].append(uuid)
    index.sort(key=lambda e: int(e["index"]))
    (out_dir / "env_index.json").write_text(
        json.dumps({"schema_version": ENV_SCHEMA_VERSION, "sweeps": index}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"exported {len(summary['exported'])}, skipped {len(summary['skipped'])}, "
        f"failed {len(summary['failed'])} -> {out_dir / 'env_index.json'}"
    )
    return summary


# ------------------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G7.2: export per-sweep Babylon .env files from G5 IBL products."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_exp = sub.add_parser("export", help="ibl_out -> {uuid}.env per sweep")
    p_exp.add_argument("--ibl-dir", type=Path, required=True, help="G5 output dir (ibl_out)")
    p_exp.add_argument("--output", "-o", type=Path, default=Path("env_out"))
    p_exp.add_argument("--sweeps", type=str, default=None, help="index selection, e.g. '1,50,60'")
    p_exp.add_argument("--sh-sample-size", type=int, default=64)
    p_exp.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    selection = parse_sweep_selection(args.sweeps) if args.sweeps else None
    export(
        args.ibl_dir,
        args.output,
        sweep_indices=selection,
        force=args.force,
        sh_sample_size=args.sh_sample_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
